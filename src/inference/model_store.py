from __future__ import annotations

import io
import json
from typing import Any, Dict, Optional, Tuple

import joblib
from botocore.exceptions import ClientError

from src.common.config import S3_DEFAULT_POINTER_KEY
from src.common.s3_io import read_json

_MODEL = None
_MODEL_ETAG = None
_MODEL_KEY = None


def resolve_model_key(s3, bucket: str, event_context: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    if event_context.get("model_key"):
        return event_context["model_key"], "event_override", None

    try:
        default = read_json(s3, bucket, S3_DEFAULT_POINTER_KEY)
        if default.get("model_key"):
            return default["model_key"], "default_pointer", default
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("NoSuchKey", "404", "NotFound"):
            raise

    raise FileNotFoundError(
        f"Default model pointer not found or invalid: s3://{bucket}/{S3_DEFAULT_POINTER_KEY}. "
        "Run producer training at least once (or pass 'model_key' explicitly)."
    )


def load_model_cached(s3, bucket: str, model_key: str) -> Any:
    global _MODEL, _MODEL_ETAG, _MODEL_KEY
    head = s3.head_object(Bucket=bucket, Key=model_key)
    etag = head.get("ETag")
    if _MODEL is not None and _MODEL_KEY == model_key and _MODEL_ETAG == etag:
        return _MODEL

    obj = s3.get_object(Bucket=bucket, Key=model_key)
    model = joblib.load(io.BytesIO(obj["Body"].read()))
    _MODEL = model
    _MODEL_KEY = model_key
    _MODEL_ETAG = etag
    return model


def get_classes(model: Any):
    try:
        return [str(c) for c in model.named_steps["clf"].classes_]
    except Exception:
        pass
    try:
        return [str(c) for c in model.classes_]
    except Exception:
        return None
