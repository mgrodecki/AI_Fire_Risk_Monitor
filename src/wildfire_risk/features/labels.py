from __future__ import annotations

import pandas as pd


def make_next_day_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "cell_id", "synthetic_fire_today", "frp_today"]].copy()
    out["date"] = pd.to_datetime(out["date"])

    shifted = out.copy()
    shifted["date"] = shifted["date"] - pd.Timedelta(days=1)
    shifted = shifted.rename(
        columns={
            "synthetic_fire_today": "label_fire_count_next_1d",
            "frp_today": "label_frp_sum_next_1d",
        }
    )

    labels = out[["date", "cell_id"]].merge(
        shifted[["date", "cell_id", "label_fire_count_next_1d", "label_frp_sum_next_1d"]],
        on=["date", "cell_id"],
        how="left",
    )
    labels["label_fire_count_next_1d"] = labels["label_fire_count_next_1d"].fillna(0).astype(int)
    labels["label_frp_sum_next_1d"] = labels["label_frp_sum_next_1d"].fillna(0.0).round(2)
    labels["label_fire_next_1d"] = (labels["label_fire_count_next_1d"] > 0).astype(int)
    return labels
