from __future__ import annotations

import numpy as np
import pandas as pd


def make_static_features(grid_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = len(grid_df)

    df = grid_df.copy()
    df["elevation_mean_m"] = rng.uniform(0, 3000, n).round(2)
    df["slope_mean_deg"] = rng.uniform(0, 35, n).round(2)

    land_mix = rng.dirichlet([2, 2, 2], size=n)
    df["forest_fraction"] = land_mix[:, 0].round(4)
    df["shrub_fraction"] = land_mix[:, 1].round(4)
    df["grass_fraction"] = land_mix[:, 2].round(4)

    df["road_density_km_per_km2"] = rng.uniform(0, 3, n).round(3)
    df["population_density"] = rng.uniform(1, 500, n).round(2)
    return df


def make_daily_base(grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    base = (
        pd.MultiIndex.from_product([dates, grid_df["cell_id"]], names=["date", "cell_id"])
        .to_frame(index=False)
    )
    return base.merge(grid_df, on="cell_id", how="left")


def make_dynamic_features(base_df: pd.DataFrame, static_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = base_df.merge(static_df.drop(columns=["lat_center", "lon_center", "area_km2"]), on="cell_id", how="left").copy()

    dayofyear = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    seasonal_heat = 15 + 12 * np.sin(2 * np.pi * dayofyear / 365.25)

    elevation_factor = (df["elevation_mean_m"].to_numpy() / 3000.0)
    forest_factor = df["forest_fraction"].to_numpy()
    shrub_factor = df["shrub_fraction"].to_numpy()

    df["t2m_max_c"] = (seasonal_heat + rng.normal(0, 4, len(df)) - elevation_factor * 6).round(2)
    df["rh_min_pct"] = np.clip(65 - seasonal_heat + rng.normal(0, 10, len(df)), 5, 95).round(2)
    df["wind10m_max_ms"] = np.clip(rng.gamma(2.2, 2.0, len(df)), 0.1, 25).round(2)
    df["precip_mm"] = np.clip(rng.exponential(1.5, len(df)) - 0.7, 0, None).round(2)
    df["vpd_kpa"] = np.clip((df["t2m_max_c"] / 20.0) + (1 - df["rh_min_pct"] / 100.0) * 2, 0.01, None).round(3)

    df["soil_moisture_surface_m3m3"] = np.clip(
        0.35 - (df["t2m_max_c"] / 100) - (df["wind10m_max_ms"] / 100) + rng.normal(0, 0.02, len(df)),
        0.02, 0.5
    ).round(3)
    df["soil_moisture_pct_of_normal"] = np.clip(df["soil_moisture_surface_m3m3"] / 0.25 * 100, 5, 180).round(2)

    raw_ndvi = 0.2 + 0.5 * forest_factor + 0.2 * shrub_factor + rng.normal(0, 0.05, len(df))
    df["ndvi"] = np.clip(raw_ndvi, -1, 1).round(3)
    df["lst_day_c"] = (df["t2m_max_c"] + rng.normal(3, 2, len(df))).round(2)

    dryness = (
        0.35 * (df["t2m_max_c"] / 40.0)
        + 0.25 * (df["wind10m_max_ms"] / 20.0)
        + 0.25 * (1 - df["soil_moisture_surface_m3m3"] / 0.5)
        + 0.15 * (1 - df["ndvi"])
    )
    df["veg_dryness_index"] = np.clip(dryness * 100, 0, 100).round(2)

    fwi = (
        0.35 * df["t2m_max_c"]
        + 0.35 * df["wind10m_max_ms"]
        + 0.25 * df["veg_dryness_index"] / 3
        - 0.25 * df["rh_min_pct"] / 3
        - 0.3 * df["precip_mm"]
    )
    df["fwi"] = np.clip(fwi, 0, None).round(2)

    return df


def make_synthetic_fire_activity(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()

    logits = (
        -5.5
        + 0.08 * out["t2m_max_c"]
        + 0.10 * out["wind10m_max_ms"]
        + 0.025 * out["veg_dryness_index"]
        - 0.03 * out["rh_min_pct"]
        - 1.2 * out["soil_moisture_surface_m3m3"]
        + 0.8 * out["forest_fraction"]
        + 0.4 * out["shrub_fraction"]
    )
    p = 1 / (1 + np.exp(-logits))
    out["synthetic_fire_today"] = rng.binomial(1, np.clip(p, 0, 0.85))

    out["frp_today"] = np.where(
        out["synthetic_fire_today"] == 1,
        rng.gamma(shape=2.5, scale=12.0, size=len(out)).round(2),
        0.0
    )

    return out
