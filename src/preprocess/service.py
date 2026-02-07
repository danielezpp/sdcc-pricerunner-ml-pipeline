from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from src.common.job_status import write_job_status
from src.common.keys import preprocess_outputs_for_input_key
from src.common.serialize import df_to_csv_bytes, json_bytes
from src.preprocess.preprocess_core import preprocess_dataframe


def run_preprocess_for_s3_object(s3, bucket: str, key: str) -> Dict[str, Any]:
    output = preprocess_outputs_for_input_key(key)

    job_id = output.get("job_id")
    mode = output["mode"]

    if mode == "job" and job_id:
        write_job_status(
            s3=s3,
            bucket=bucket,
            job_id=job_id,
            stage="PREPROCESS",
            state="RUNNING",
            message="Preprocessing started",
            artifacts={"input_key": key},
        )

    # read csv
    raw_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    df_raw = pd.read_csv(io.BytesIO(raw_bytes), dtype=str)

    result = preprocess_dataframe(df_raw)

    now = datetime.now(timezone.utc).isoformat()
    stats = dict(result.stats)
    stats["timestamp_utc"] = now
    stats["input_bucket"] = bucket
    stats["input_key"] = key

    # write outputs
    s3.put_object(Bucket=bucket, Key=output["processed"], Body=df_to_csv_bytes(result.processed_df), ContentType="text/csv")
    s3.put_object(Bucket=bucket, Key=output["schema"], Body=json_bytes(result.schema), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=output["classes"], Body=json_bytes(result.classes), ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=output["stats"], Body=json_bytes(stats), ContentType="application/json")

    if mode == "job" and job_id:
        write_job_status(
            s3=s3,
            bucket=bucket,
            job_id=job_id,
            stage="PREPROCESS",
            state="SUCCEEDED",
            message="Preprocessing completed",
            artifacts={
                "input_key": key,
                "processed_key": output["processed"],
                "schema_key": output["schema"],
                "classes_key": output["classes"],
                "stats_key": output["stats"],
            },
        )

    return {
        "ok": True,
        "processed_key": output["processed"],
        "schema_key": output["schema"],
        "classes_key": output["classes"],
        "stats_key": output["stats"],
        "mode": mode,
        "job_id": job_id,
        "n_rows_processed": int(len(result.processed_df)),
        "timestamp_utc": now,
    }
