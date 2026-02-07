from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.config import S3_INFERENCE_OUTPUT_PREFIX
from src.inference.model_store import resolve_model_key, load_model_cached, get_classes


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _compute_gap_1_2(pred_row: Dict[str, Any]) -> Optional[float]:
    try:
        topk = pred_row.get("topk") or []
        if len(topk) < 2:
            return None
        p1 = _safe_float(topk[0].get("prob"))
        p2 = _safe_float(topk[1].get("prob"))
        if p1 is None or p2 is None:
            return None
        return float(p1 - p2)
    except Exception:
        return None


def _build_csv_rows_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for p in result.get("predictions", []):
        inp = p.get("input") or {}
        topk = p.get("topk") or []

        conf = _safe_float(p.get("confidence"))
        gap = _compute_gap_1_2(p)

        t2 = topk[1] if len(topk) > 1 else {}
        t3 = topk[2] if len(topk) > 2 else {}

        rows.append({
            "product_title": (inp.get("Product Title") or ""),
            "merchant_id": (inp.get("Merchant ID") or ""),
            "predicted_label": (p.get("predicted_label") or ""),
            "confidence": conf if conf is not None else "",
            "gap_1_2": gap if gap is not None else "",
            "top2_label": (t2.get("label") or ""),
            "top2_prob": _safe_float(t2.get("prob")) if t2 else "",
            "top3_label": (t3.get("label") or ""),
            "top3_prob": _safe_float(t3.get("prob")) if t3 else "",
        })
    return rows


def predict_dataframe(s3, df: pd.DataFrame, bucket: str, top_k: int, event_context: Dict[str, Any]) -> Dict[str, Any]:
    model_key, source, default_ptr = resolve_model_key(s3, bucket, event_context)

    default_run_id = default_ptr.get("run_id") if default_ptr else None
    default_timestamp_utc = default_ptr.get("timestamp_utc") if default_ptr else None

    model = load_model_cached(s3, bucket, model_key)

    if "Merchant ID" not in df.columns:
        df["Merchant ID"] = "0"

    X = df[["Product Title", "Merchant ID"]]
    preds = model.predict(X)

    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    classes = get_classes(model) if proba is not None else None

    predictions_out = []
    records = df.to_dict("records")

    for i, rec in enumerate(records):
        row = {"input": rec, "predicted_label": str(preds[i])}
        if proba is not None and classes:
            probs_i = proba[i]
            idx_sorted = sorted(range(len(probs_i)), key=lambda j: probs_i[j], reverse=True)[:top_k]
            row["confidence"] = float(probs_i[idx_sorted[0]])
            row["topk"] = [{"label": str(classes[j]), "prob": float(probs_i[j])} for j in idx_sorted]
        predictions_out.append(row)

    return {
        "ok": True,
        "model_key": model_key,
        "source": source,
        "default_run_id": default_run_id,
        "default_timestamp_utc": default_timestamp_utc,
        "n_records": len(records),
        "predictions": predictions_out,
    }


def process_batch_s3_object(s3, bucket: str, input_key: str) -> None:
    obj = s3.get_object(Bucket=bucket, Key=input_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str)

    result = predict_dataframe(s3, df, bucket, top_k=3, event_context={})
    result["source_file"] = input_key

    filename = os.path.basename(input_key)
    output_key_json = f"{S3_INFERENCE_OUTPUT_PREFIX}/{filename}_result.json"
    output_key_csv = f"{S3_INFERENCE_OUTPUT_PREFIX}/{filename}_result.csv"
    output_key_summary = f"{S3_INFERENCE_OUTPUT_PREFIX}/{filename}_summary.json"

    preds = result.get("predictions", [])

    labels_dist: Dict[str, int] = {}
    conf_values: List[float] = []
    low_conf_count = 0

    for p in preds:
        lab = str(p.get("predicted_label", ""))
        labels_dist[lab] = labels_dist.get(lab, 0) + 1

        conf = _safe_float(p.get("confidence"))
        gap = _compute_gap_1_2(p)

        if conf is not None:
            conf_values.append(conf)

        is_low = False
        if conf is not None and conf < 0.45:
            is_low = True
        if gap is not None and gap < 0.05:
            is_low = True
        if is_low:
            low_conf_count += 1

    avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else None

    summary = {
        "ok": True,
        "source_file": input_key,
        "n_records": result.get("n_records", len(preds)),
        "model_key": result.get("model_key"),
        "source": result.get("source"),
        "default_run_id": result.get("default_run_id"),
        "default_timestamp_utc": result.get("default_timestamp_utc"),
        "labels_distribution": labels_dist,
        "low_confidence_count": low_conf_count,
        "avg_confidence": avg_conf,
        "output_keys": {"json": output_key_json, "csv": output_key_csv, "summary": output_key_summary},
    }

    rows = _build_csv_rows_from_result(result)
    out_df = pd.DataFrame(rows)

    s3.put_object(Bucket=bucket, Key=output_key_json, Body=json.dumps(result, ensure_ascii=False), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=output_key_summary, Body=json.dumps(summary, ensure_ascii=False), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=output_key_csv, Body=out_df.to_csv(index=False).encode("utf-8"), ContentType="text/csv")
