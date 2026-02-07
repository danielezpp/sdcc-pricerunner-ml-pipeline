from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3
from botocore.config import Config

from src.common.http import api_response, parse_json_body
from src.common.job_status import write_job_status
from src.common.keys import (
    aws_region,
    job_dataset_key,
    job_manifest_key,
    job_status_key,
    model_key_for_job,
)
from src.common.serialize import json_bytes

s3 = boto3.client("s3")

BUCKET = os.environ.get("BUCKET_NAME", "")
STATUS_TTL_SECONDS = int(os.environ.get("STATUS_TTL_SECONDS", "3600"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    if not BUCKET:
        return api_response(500, {"ok": False, "error": "Missing BUCKET_NAME env var"}, allow_methods="OPTIONS,POST")

    ok, body, err = parse_json_body(event)
    if not ok:
        return api_response(400, {"ok": False, "error": err}, allow_methods="OPTIONS,POST")

    algo = (body.get("algo") or "logreg").lower()
    params = body.get("params") or {}

    now_id = datetime.utcnow().strftime("%Y%m%d-%H%M")
    job_id = f"{now_id}-pricerunner-{algo}-{uuid.uuid4().hex[:6]}"

    dataset_key = job_dataset_key(job_id)
    manifest_key = job_manifest_key(job_id)
    status_key = job_status_key(job_id)

    # status iniziale (identico come intent)
    now = _now_iso()
    status_payload = {
        "job_id": job_id,
        "stage": "CREATED",
        "state": "PENDING",
        "updated_at_utc": now,
        "message": "Job created",
        "artifacts": {"dataset_key": dataset_key, "manifest_key": manifest_key},
        "error": None,
        "ttl_seconds_hint": STATUS_TTL_SECONDS,
    }
    s3.put_object(Bucket=BUCKET, Key=status_key, Body=json_bytes(status_payload), ContentType="application/json")

    # --- PRESIGN (ATTENZIONE REGION/URL) ---
    region = aws_region()

    s3_presign = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(signature_version="s3v4"),
    )

    presigned_post = s3_presign.generate_presigned_post(
        Bucket=BUCKET,
        Key=dataset_key,
        Conditions=[["content-length-range", 1, 50_000_000]],
        ExpiresIn=900,
    )
    # URL ESATTA come richiesto (forma che ti tiene in piedi la pipeline)
    presigned_post["url"] = f"https://{BUCKET}.s3.{region}.amazonaws.com/"

    presigned_manifest_post = s3_presign.generate_presigned_post(
        Bucket=BUCKET,
        Key=manifest_key,
        Conditions=[["content-length-range", 1, 1_000_000]],
        ExpiresIn=900,
    )
    presigned_manifest_post["url"] = f"https://{BUCKET}.s3.{region}.amazonaws.com/"

    presigned_status_get = s3_presign.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET, "Key": status_key},
        ExpiresIn=900,
    )

    # expected model_key: centralizzato e coerente col train
    expected_model_key = model_key_for_job(job_id)

    return api_response(
        200,
        {
            "ok": True,
            "job_id": job_id,
            "defaults": {
                "schema_version": 1,
                "job": {"label": "", "created_at_utc": now},
                "train": {"algorithm": algo, "params": params},
            },
            "upload": {
                "dataset": {"type": "presigned_post", "url": presigned_post["url"], "fields": presigned_post["fields"], "key": dataset_key},
                "manifest": {"type": "presigned_post", "url": presigned_manifest_post["url"], "fields": presigned_manifest_post["fields"], "key": manifest_key},
            },
            "polling": {"status": {"type": "presigned_get", "url": presigned_status_get, "key": status_key}},
            "expected": {"model_key": expected_model_key},
        },
        allow_methods="OPTIONS,POST",
    )
