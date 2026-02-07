from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def parse_json_body(event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """
    Ritorna: (ok, body_dict, error_message)
    Supporta:
      - event["body"] string/dict (API Gateway)
      - event già dict (invocazioni dirette)
    """
    if "body" not in event:
        if isinstance(event, dict):
            return True, event, ""
        return False, {}, "Invalid event"

    raw = event.get("body")
    if raw is None or raw == "":
        return True, {}, ""

    try:
        if isinstance(raw, str):
            return True, json.loads(raw), ""
        if isinstance(raw, dict):
            return True, raw, ""
        return False, {}, "Invalid JSON body"
    except Exception:
        return False, {}, "Invalid JSON body"


def api_response(status_code: int, payload: Dict[str, Any], allow_methods: str = "OPTIONS,POST") -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": allow_methods,
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }
