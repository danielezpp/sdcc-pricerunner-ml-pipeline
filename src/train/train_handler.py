from __future__ import annotations

from typing import Any, Dict

import boto3

from src.common.config import S3_PROCESSED_KEY
from src.common.keys import parse_context_from_processed_key
from src.train.service import run_training, fail_job

s3 = boto3.client("s3")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    if "Records" in event and event["Records"]:
        rec = event["Records"][0]
        bucket = rec["s3"]["bucket"]["name"]
        key = rec["s3"]["object"]["key"]
    else:
        bucket = event["bucket"]
        key = event.get("key", S3_PROCESSED_KEY)

    ctx = parse_context_from_processed_key(key)
    mode = ctx["mode"]
    job_id = ctx["job_id"]

    try:
        return run_training(s3, bucket=bucket, processed_key=key)
    except Exception as e:
        if mode == "job" and job_id:
            fail_job(s3, bucket=bucket, job_id=job_id, stage="TRAINING", exc=e)
        raise
