from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.common.keys import job_status_key
from src.common.serialize import json_bytes


def write_job_status(
    s3,
    bucket: str,
    job_id: str,
    stage: str,
    state: str,
    message: str,
    artifacts: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "job_id": job_id,
        "stage": stage,
        "state": state,
        "updated_at_utc": now,
        "message": message,
        "artifacts": artifacts or {},
        "error": error,
    }
    s3.put_object(
        Bucket=bucket,
        Key=job_status_key(job_id),
        Body=json_bytes(payload),
        ContentType="application/json",
    )
