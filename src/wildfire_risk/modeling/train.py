from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump

from wildfire_risk.data.schema import FEATURE_COLUMNS


def build_model(random_state: int = 42) -> Pipeline:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def split_train_valid(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").copy()
    unique_dates = sorted(pd.to_datetime(df["date"]).dt.date.unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = pd.Timestamp(unique_dates[split_idx])

    train_df = df[pd.to_datetime(df["date"]) < split_date].copy()
    valid_df = df[pd.to_datetime(df["date"]) >= split_date].copy()
    return train_df, valid_df


def train_model(df: pd.DataFrame, target_col: str, random_state: int = 42):
    train_df, valid_df = split_train_valid(df, target_col)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_col].astype(int)
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[target_col].astype(int)

    pipeline = build_model(random_state=random_state)
    pipeline.fit(X_train, y_train)

    valid_probs = pipeline.predict_proba(X_valid)[:, 1]
    metrics = {
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_valid": float(y_valid.mean()),
        "average_precision": float(average_precision_score(y_valid, valid_probs)),
        "roc_auc": float(roc_auc_score(y_valid, valid_probs)),
        "brier_score": float(brier_score_loss(y_valid, valid_probs)),
    }
    return pipeline, metrics


def save_model(model: Pipeline, path: str) -> None:
    dump(model, path)
