from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    doy = dt.dt.dayofyear
    month = dt.dt.month

    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["cell_id", "date"]).copy()
    g = out.groupby("cell_id", group_keys=False)

    out["precip_3d_mm"] = g["precip_mm"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    out["precip_7d_mm"] = g["precip_mm"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    out["precip_30d_mm"] = g["precip_mm"].rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)
    out["vpd_7d_mean"] = g["vpd_kpa"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    out["fwi_3d_max"] = g["fwi"].rolling(3, min_periods=1).max().reset_index(level=0, drop=True)

    out["dry_days_30d"] = (
        g["precip_mm"]
        .rolling(30, min_periods=1)
        .apply(lambda x: float((x < 1.0).sum()), raw=False)
        .reset_index(level=0, drop=True)
    )

    out["fire_count_prev_7d"] = (
        g["synthetic_fire_today"]
        .shift(1)
        .rolling(7, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(int)
    )

    out["frp_sum_prev_7d"] = (
        g["frp_today"]
        .shift(1)
        .rolling(7, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .round(2)
    )

    out["ndvi_anom_30d"] = (
        out["ndvi"] - g["ndvi"].rolling(30, min_periods=5).mean().reset_index(level=0, drop=True)
    ).fillna(0.0).round(3)

    return out
