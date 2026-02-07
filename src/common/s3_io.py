from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


def safe_etag(etag: str) -> str:
    etag = (etag or "").strip().strip('"')
    return re.sub(r"[^a-zA-Z0-9\-]", "", etag)


def exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def read_bytes(s3, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def read_json(s3, bucket: str, key: str) -> Dict[str, Any]:
    raw = read_bytes(s3, bucket, key)
    return json.loads(raw.decode("utf-8"))


def put_bytes(s3, bucket: str, key: str, body: bytes, content_type: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def put_json(s3, bucket: str, key: str, obj: Dict[str, Any]) -> None:
    put_bytes(s3, bucket, key, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")


def s3_client_default():
    return boto3.client("s3")
