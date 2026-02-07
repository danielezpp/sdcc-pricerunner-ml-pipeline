from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.common.config import (
    FEATURE_COLUMNS,
    PROCESSED_COLUMNS,
    RAW_EXPECTED_COLUMNS,
    TARGET_COLUMN,
)
from src.common.io_utils import strip_column_names


@dataclass(frozen=True)
class PreprocessResult:
    processed_df: pd.DataFrame
    schema: Dict
    classes: Dict
    stats: Dict


def _normalize_text(s: pd.Series) -> pd.Series:
    # Lowercase + trim + collapse spazi
    s = s.fillna("")
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _normalize_merchant_id(s: pd.Series) -> pd.Series:
    # Forziamo a stringa per trattarla come categorica (coerente con OneHotEncoder)
    s = s.fillna("unknown")
    s = s.astype(str).str.strip()
    s = s.replace({"": "unknown"})
    return s


def preprocess_dataframe(df_raw: pd.DataFrame) -> PreprocessResult:
    """
    Trasforma il dataset raw in un dataset processed con:
      - nomi colonna ripuliti
      - selezione feature + target
      - normalizzazione minima (testo + merchant id)
      - metadata (schema, classi, stats)
    """
    df = strip_column_names(df_raw)

    # Validazione minima (non blocchiamo se mancano colonne non usate, ma avvisiamo via schema)
    present_cols = set(df.columns)

    # Se mancano feature/target, qui invece è un errore bloccante
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(list(required - present_cols))
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}. Found: {sorted(list(present_cols))}")

    # Selezione e copia
    out = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Normalizzazioni
    out["Product Title"] = _normalize_text(out["Product Title"])
    out["Merchant ID"] = _normalize_merchant_id(out["Merchant ID"])
    out[TARGET_COLUMN] = out[TARGET_COLUMN].fillna("").astype(str).str.strip()
    # (non facciamo lowercase sul target per preservare label “ufficiale”; se vuoi uniformare, lo facciamo)

    # Drop righe senza target (non si può addestrare senza label)
    before = len(out)
    out = out[out[TARGET_COLUMN] != ""].copy()
    dropped_no_target = before - len(out)

    # Riordino colonne (contratto)
    out = out[PROCESSED_COLUMNS]

    # Classi target
    classes: List[str] = sorted(out[TARGET_COLUMN].unique().tolist())

    # Metadata
    schema = {
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "raw_columns_expected": RAW_EXPECTED_COLUMNS,
        "raw_columns_found": sorted(list(df.columns)),
        "processed_columns": PROCESSED_COLUMNS,
    }

    stats = {
        "n_rows_raw": int(len(df_raw)),
        "n_rows_processed": int(len(out)),
        "dropped_rows_missing_target": int(dropped_no_target),
        "n_classes": int(len(classes)),
    }

    classes_payload = {"classes": classes, "n_classes": int(len(classes))}

    return PreprocessResult(
        processed_df=out,
        schema=schema,
        classes=classes_payload,
        stats=stats,
    )
