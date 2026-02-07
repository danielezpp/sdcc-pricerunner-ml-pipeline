from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict

import boto3
from botocore.config import Config

from src.common.keys import aws_region, inference_input_key, inference_output_keys

s3 = boto3.client("s3")
DEFAULT_BUCKET = os.environ.get("DEFAULT_BUCKET")


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    try:
        bucket = DEFAULT_BUCKET
        if not bucket:
            return {"statusCode": 500, "body": json.dumps({"error": "DEFAULT_BUCKET env var missing"})}

        query = event.get("queryStringParameters") or {}
        filename = query.get("filename") or f"batch_{uuid.uuid4().hex[:8]}.csv"

        input_key = inference_input_key(filename)
        out = inference_output_keys(filename)

        # presigned POST per upload input
        post = s3.generate_presigned_post(Bucket=bucket, Key=input_key, ExpiresIn=300)

        region = aws_region()
        # URL ESATTA richiesta
        post["url"] = f"https://{bucket}.s3.{region}.amazonaws.com/"

        s3_presign = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(signature_version="s3v4"),
        )

        download_url_json = s3_presign.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": out["json"]}, ExpiresIn=3600
        )
        download_url_csv = s3_presign.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": out["csv"]}, ExpiresIn=3600
        )
        download_url_summary = s3_presign.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": out["summary"]}, ExpiresIn=3600
        )

        return {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "post": post,
                    "expected_output_keys": {"json": out["json"], "csv": out["csv"], "summary": out["summary"]},
                    "download_urls": {"json": download_url_json, "csv": download_url_csv, "summary": download_url_summary},
                    "input_key": input_key,
                }
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
