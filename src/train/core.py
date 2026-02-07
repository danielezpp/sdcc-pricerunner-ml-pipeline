from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.common.config import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    metrics: Dict
    model_info: Dict


def train_model(df: pd.DataFrame, random_state: int = 42, manifest: Dict | None = None) -> TrainResult:
    manifest = manifest or {}
    algo = (manifest.get("algo") or "logreg").lower()
    params = manifest.get("params") or {}

    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in processed dataset.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("title_tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2)), "Product Title"),
            ("merchant_ohe", OneHotEncoder(handle_unknown="ignore"), ["Merchant ID"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    if algo in ("logreg", "logistic_regression", "logistic"):
        clf = LogisticRegression(
            solver=params.get("solver", "saga"),
            max_iter=int(params.get("max_iter", 2000)),
            C=float(params.get("C", 1.0)),
            n_jobs=int(params.get("n_jobs", -1)),
        )
        model_type = "LogisticRegression"
    elif algo in ("random_forest", "rf", "forest"):
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=random_state,
        )
        model_type = "RandomForestClassifier"
    else:
        raise ValueError(f"Unsupported algo '{algo}'. Allowed: logreg, random_forest")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(pd.Series(y).nunique()),
    }

    model_info = {
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_type": f"sklearn Pipeline (TFIDF + OneHot + {model_type})",
        "algo": algo,
        "params": params,
    }

    return TrainResult(pipeline=pipeline, metrics=metrics, model_info=model_info)
