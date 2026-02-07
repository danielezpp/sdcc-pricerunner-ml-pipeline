from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

from src.common.config import (
    S3_DEFAULT_POINTER_KEY,
    S3_INFERENCE_INPUT_PREFIX,
    S3_INFERENCE_OUTPUT_PREFIX,
    S3_MODEL_MARKERS_PREFIX,
    S3_MODEL_VERSIONS_PREFIX,
)


# ---------- JOB KEYS ----------
def job_dataset_key(job_id: str) -> str:
    return f"raw/pricerunner/jobs/{job_id}/dataset.csv"


def job_manifest_key(job_id: str) -> str:
    return f"raw/pricerunner/jobs/{job_id}/manifest.json"


def job_status_key(job_id: str) -> str:
    return f"raw/pricerunner/jobs/{job_id}/status.json"


# ---------- PREPROCESS OUTPUT KEYS ----------
def producer_processed_prefix() -> str:
    return "processed/pricerunner/producer"


def job_processed_prefix(job_id: str) -> str:
    return f"processed/pricerunner/jobs/{job_id}"


def preprocess_outputs_for_input_key(input_key: str) -> Dict[str, Optional[str]]:
    """
    Compatibile con la tua compute_output_keys.
    """
    if input_key.startswith("raw/pricerunner/producer/"):
        base = producer_processed_prefix()
        return {
            "mode": "producer",
            "job_id": None,
            "processed": f"{base}/processed.csv",
            "schema": f"{base}/schema.json",
            "classes": f"{base}/classes.json",
            "stats": f"{base}/stats.json",
        }

    if input_key.startswith("raw/pricerunner/jobs/"):
        parts = input_key.split("/")
        if len(parts) < 5:
            raise ValueError(f"Invalid job input key: {input_key}")
        job_id = parts[3]
        base = job_processed_prefix(job_id)
        return {
            "mode": "job",
            "job_id": job_id,
            "processed": f"{base}/processed.csv",
            "schema": f"{base}/schema.json",
            "classes": f"{base}/classes.json",
            "stats": f"{base}/stats.json",
        }

    raise ValueError(f"Unsupported input key: {input_key}")


def parse_context_from_processed_key(processed_key: str) -> Dict[str, Optional[str]]:
    if processed_key.startswith("processed/pricerunner/producer/"):
        return {"mode": "producer", "job_id": None}

    m = re.match(r"^processed/pricerunner/jobs/([^/]+)/processed\.csv$", processed_key)
    if m:
        return {"mode": "job", "job_id": m.group(1)}

    raise ValueError(f"Unsupported processed key: {processed_key}")


# ---------- TRAIN ARTIFACT KEYS ----------
def version_prefix_for_job(job_id: str) -> str:
    # NOTA: allineato al tuo train_handler attuale: .../versions/jobs/{job_id}
    return f"{S3_MODEL_VERSIONS_PREFIX}/jobs/{job_id}"


def version_prefix_for_producer(run_id: str) -> str:
    return f"{S3_MODEL_VERSIONS_PREFIX}/producer/{run_id}"


def model_key_for_job(job_id: str) -> str:
    return f"{version_prefix_for_job(job_id)}/pipeline.joblib"


def metrics_key_for_job(job_id: str) -> str:
    return f"{version_prefix_for_job(job_id)}/metrics.json"


def model_info_key_for_job(job_id: str) -> str:
    return f"{version_prefix_for_job(job_id)}/model_info.json"


def marker_key_for_job(job_id: str, processed_etag: str) -> str:
    return f"{S3_MODEL_MARKERS_PREFIX}/jobs/{job_id}/{processed_etag}.json"


def marker_key_for_producer(processed_etag: str) -> str:
    return f"{S3_MODEL_MARKERS_PREFIX}/producer/{processed_etag}.json"


def default_pointer_key() -> str:
    return S3_DEFAULT_POINTER_KEY


# ---------- INFERENCE BATCH KEYS ----------
def inference_input_key(filename: str) -> str:
    return f"{S3_INFERENCE_INPUT_PREFIX}/{filename}"


def inference_output_keys(filename: str) -> Dict[str, str]:
    return {
        "json": f"{S3_INFERENCE_OUTPUT_PREFIX}/{filename}_result.json",
        "csv": f"{S3_INFERENCE_OUTPUT_PREFIX}/{filename}_result.csv",
        "summary": f"{S3_INFERENCE_OUTPUT_PREFIX}/{filename}_summary.json",
    }


def aws_region() -> str:
    return os.environ.get("AWS_REGION", "eu-south-1")
