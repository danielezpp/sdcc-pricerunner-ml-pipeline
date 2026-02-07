from __future__ import annotations

import json
import os
from typing import Any, Dict

import boto3
import pandas as pd

from src.common.http import api_response
from src.inference.service import predict_dataframe, process_batch_s3_object

s3 = boto3.client("s3")
DEFAULT_BUCKET = os.environ.get("DEFAULT_BUCKET")


def _error_payload(code: str, message: str, details: Any = None) -> Dict[str, Any]:
    p = {"ok": False, "error": {"code": code, "message": message}}
    if details is not None:
        p["error"]["details"] = details
    return p


def _handle_api_gateway_event(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event
    if "body" in event:
        try:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        except Exception:
            return api_response(400, _error_payload("bad_request", "Invalid JSON body"), allow_methods="OPTIONS,POST")

    records = body.get("records")
    if not records or not isinstance(records, list):
        return api_response(400, _error_payload("bad_request", "Missing records list"), allow_methods="OPTIONS,POST")

    bucket = body.get("bucket") or DEFAULT_BUCKET
    top_k = int(body.get("top_k", 3))

    try:
        df = pd.DataFrame.from_records(records)
        if "Product Title" not in df.columns:
            return api_response(400, _error_payload("bad_request", "Missing 'Product Title' column"), allow_methods="OPTIONS,POST")

        result = predict_dataframe(s3, df, bucket, top_k, body)
        return api_response(200, result, allow_methods="OPTIONS,POST")

    except FileNotFoundError as e:
        return api_response(
            409,
            _error_payload(
                "model_not_ready",
                "Default model not available yet. Train producer model or specify model_key.",
                {"detail": str(e)},
            ),
            allow_methods="OPTIONS,POST",
        )
    except Exception as e:
        return api_response(500, _error_payload("internal_error", str(e)), allow_methods="OPTIONS,POST")


def _handle_s3_batch_event(event: Dict[str, Any]) -> Dict[str, Any]:
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    input_key = record["s3"]["object"]["key"]

    process_batch_s3_object(s3, bucket=bucket, input_key=input_key)
    return {"statusCode": 200, "body": "Batch processing completed"}


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    if "Records" in event and len(event["Records"]) > 0 and "s3" in event["Records"][0]:
        return _handle_s3_batch_event(event)
    return _handle_api_gateway_event(event)
