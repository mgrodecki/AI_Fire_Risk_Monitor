from __future__ import annotations

import pandas as pd

from wildfire_risk.data.schema import FEATURE_COLUMNS


def classify_risk(score: float, thresholds: dict) -> str:
    if score < thresholds["low"]:
        return "low"
    if score < thresholds["moderate"]:
        return "moderate"
    if score < thresholds["high"]:
        return "high"
    return "extreme"


def predict_scores(model, df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df[["date", "cell_id", "lat_center", "lon_center"]].copy()
    out["pred_fire_next_1d"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    out["risk_score"] = out["pred_fire_next_1d"]
    out["risk_class"] = out["risk_score"].apply(lambda x: classify_risk(float(x), thresholds))
    return out.sort_values(["date", "cell_id"]).reset_index(drop=True)
