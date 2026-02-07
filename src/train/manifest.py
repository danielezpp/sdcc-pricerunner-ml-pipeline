from __future__ import annotations

import json
from typing import Any, Dict

from botocore.exceptions import ClientError

from src.common.keys import job_manifest_key


def load_manifest_for_job(s3, bucket: str, job_id: str) -> Dict[str, Any]:
    key = job_manifest_key(job_id)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return {}
        raise


def normalize_manifest(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = raw or {}

    if raw.get("schema_version") == 1:
        train = raw.get("train") or {}
        job = raw.get("job") or {}
        algo = (train.get("algorithm") or "logreg").lower()
        params = train.get("params") or {}
        return {
            "schema_version": 1,
            "algo": algo,
            "params": params,
            "job": job,
            "raw": raw,
        }

    algo = (raw.get("algo") or "logreg").lower()
    params = raw.get("params") or {}
    return {
        "schema_version": 0,
        "algo": algo,
        "params": params,
        "job": {},
        "raw": raw,
    }
