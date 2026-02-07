from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.common.job_status import write_job_status
from src.common.keys import (
    default_pointer_key,
    marker_key_for_job,
    marker_key_for_producer,
    parse_context_from_processed_key,
    version_prefix_for_job,
    version_prefix_for_producer,
)
from src.common.s3_io import exists, safe_etag
from src.common.serialize import json_bytes
from src.train.core import train_model
from src.train.manifest import load_manifest_for_job, normalize_manifest


def run_training(s3, bucket: str, processed_key: str) -> Dict[str, Any]:
    ctx = parse_context_from_processed_key(processed_key)
    mode = ctx["mode"]
    job_id = ctx["job_id"]

    if mode == "job" and job_id:
        write_job_status(
            s3=s3,
            bucket=bucket,
            job_id=job_id,
            stage="TRAINING",
            state="RUNNING",
            message="Training started",
            artifacts={"processed_key": processed_key},
        )

    head = s3.head_object(Bucket=bucket, Key=processed_key)
    processed_etag = safe_etag(head.get("ETag", "")) or "no-etag"

    if mode == "job" and job_id:
        marker_key = marker_key_for_job(job_id, processed_etag)
    else:
        marker_key = marker_key_for_producer(processed_etag)

    # idempotenza
    if exists(s3, bucket, marker_key):
        if mode == "job" and job_id:
            v_prefix = version_prefix_for_job(job_id)
            v_model_key = f"{v_prefix}/pipeline.joblib"
            v_metrics_key = f"{v_prefix}/metrics.json"
            v_info_key = f"{v_prefix}/model_info.json"

            write_job_status(
                s3=s3,
                bucket=bucket,
                job_id=job_id,
                stage="DONE",
                state="SUCCEEDED",
                message="Training skipped (already trained for this dataset)",
                artifacts={
                    "processed_key": processed_key,
                    "model_key": v_model_key,
                    "metrics_key": v_metrics_key,
                    "model_info_key": v_info_key,
                    "skipped": True,
                    "processed_etag": processed_etag,
                    "reason": "already_trained_for_processed_etag",
                },
            )

            return {
                "ok": True,
                "skipped": True,
                "reason": "already_trained_for_processed_etag",
                "mode": mode,
                "job_id": job_id,
                "processed_key": processed_key,
                "processed_etag": processed_etag,
                "version_prefix": v_prefix,
                "versioned_model_key": v_model_key,
            }

        return {
            "ok": True,
            "skipped": True,
            "reason": "already_trained_for_processed_etag",
            "mode": mode,
            "job_id": None,
            "processed_key": processed_key,
            "processed_etag": processed_etag,
            "default_pointer_key": default_pointer_key(),
        }

    # read processed.csv
    processed_bytes = s3.get_object(Bucket=bucket, Key=processed_key)["Body"].read()
    df = pd.read_csv(io.BytesIO(processed_bytes), dtype=str)

    manifest_raw = load_manifest_for_job(s3, bucket, job_id) if (mode == "job" and job_id) else {}
    manifest = normalize_manifest(manifest_raw)

    result = train_model(df, manifest=manifest)

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    if mode == "job" and job_id:
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{processed_etag[:12]}"
        v_prefix = version_prefix_for_job(job_id)
    else:
        run_id = f"{now.strftime('%Y%m%d-%H%M')}-pricerunner-producer-{processed_etag[:6]}"
        v_prefix = version_prefix_for_producer(run_id)

    v_model_key = f"{v_prefix}/pipeline.joblib"
    v_metrics_key = f"{v_prefix}/metrics.json"
    v_info_key = f"{v_prefix}/model_info.json"

    metrics = dict(result.metrics)
    metrics.update(
        {
            "timestamp_utc": now_iso,
            "input_bucket": bucket,
            "input_key": processed_key,
            "processed_etag": processed_etag,
            "run_id": run_id,
            "version_prefix": v_prefix,
            "mode": mode,
            "job_id": job_id,
        }
    )

    model_info = dict(result.model_info)
    model_info.update(
        {
            "timestamp_utc": now_iso,
            "processed_key": processed_key,
            "processed_etag": processed_etag,
            "run_id": run_id,
            "version_prefix": v_prefix,
            "mode": mode,
            "job_id": job_id,
        }
    )

    if mode == "job" and job_id:
        model_info["manifest_schema_version"] = manifest.get("schema_version", 0)
        model_info["manifest_raw"] = manifest.get("raw", {})
        model_info["train_algo"] = manifest.get("algo")
        model_info["train_params"] = manifest.get("params")

    # serialize model
    buf = io.BytesIO()
    joblib.dump(result.pipeline, buf)
    model_bytes = buf.getvalue()

    s3.put_object(Bucket=bucket, Key=v_model_key, Body=model_bytes, ContentType="application/octet-stream")
    s3.put_object(Bucket=bucket, Key=v_metrics_key, Body=json_bytes(metrics), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=v_info_key, Body=json_bytes(model_info), ContentType="application/json")

    if mode == "job" and job_id:
        write_job_status(
            s3=s3,
            bucket=bucket,
            job_id=job_id,
            stage="DONE",
            state="SUCCEEDED",
            message="Training completed",
            artifacts={
                "processed_key": processed_key,
                "model_key": v_model_key,
                "metrics_key": v_metrics_key,
                "model_info_key": v_info_key,
                "version_prefix": v_prefix,
            },
        )

    if mode == "producer":
        default_pointer = {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp_utc": now_iso,
            "processed_key": processed_key,
            "processed_etag": processed_etag,
            "model_key": v_model_key,
            "metrics_key": v_metrics_key,
            "model_info_key": v_info_key,
        }
        s3.put_object(
            Bucket=bucket,
            Key=default_pointer_key(),
            Body=json_bytes(default_pointer),
            ContentType="application/json",
        )

    marker = {
        "run_id": run_id,
        "timestamp_utc": now_iso,
        "processed_key": processed_key,
        "processed_etag": processed_etag,
        "version_prefix": v_prefix,
        "mode": mode,
        "job_id": job_id,
    }
    s3.put_object(Bucket=bucket, Key=marker_key, Body=json_bytes(marker), ContentType="application/json")

    return {
        "ok": True,
        "skipped": False,
        "mode": mode,
        "job_id": job_id,
        "run_id": run_id,
        "processed_key": processed_key,
        "processed_etag": processed_etag,
        "version_prefix": v_prefix,
        "versioned_model_key": v_model_key,
        "default_pointer_key": default_pointer_key() if mode == "producer" else None,
    }


def fail_job(s3, bucket: str, job_id: str, stage: str, exc: Exception) -> None:
    write_job_status(
        s3=s3,
        bucket=bucket,
        job_id=job_id,
        stage=stage,
        state="FAILED",
        message=f"{stage} failed",
        error={"type": type(exc).__name__, "detail": str(exc)},
    )
