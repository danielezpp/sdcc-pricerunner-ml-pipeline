from __future__ import annotations

from typing import Any, Dict

import boto3

from src.preprocess.service import run_preprocess_for_s3_object

s3 = boto3.client("s3")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    try:
        return run_preprocess_for_s3_object(s3, bucket=bucket, key=key)
    except Exception:
        raise
