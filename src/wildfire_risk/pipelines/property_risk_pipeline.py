from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from wildfire_risk.config import load_settings
from wildfire_risk.ingestion.copernicus_dem import sample_copdem_points
from wildfire_risk.ingestion.esa_worldcover import classify_worldcover_points
from wildfire_risk.utils.io import ensure_dir, read_csv, write_csv


VEGETATION_TYPE_RISK = {
    "water": 0.05,
    "urban": 0.20,
    "agriculture": 0.30,
    "grass": 0.35,
    "shrub": 0.55,
    "forest": 0.75,
}


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def _first_present_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_property_coordinates(properties_df: pd.DataFrame) -> pd.DataFrame:
    lat_col = _first_present_col(properties_df, ["lat", "latitude", "lat_center"])
    lon_col = _first_present_col(properties_df, ["lon", "longitude", "lon_center"])
    if lat_col is None or lon_col is None:
        raise ValueError("Property CSV must include one of lat/latitude and lon/longitude columns.")

    out = properties_df.copy()
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if "property_id" not in out.columns:
        out["property_id"] = [f"property_{i+1}" for i in range(len(out))]
    return out


def _add_worldcover_vegetation(properties_df: pd.DataFrame, api_connectors: dict | None) -> pd.DataFrame:
    out = properties_df.copy()
    wc = classify_worldcover_points(out, lat_col="lat", lon_col="lon", api_connectors=api_connectors)
    out = pd.concat([out, wc], axis=1)

    veg_col = _first_present_col(out, ["vegetation_type", "veg_type", "land_cover"])
    if veg_col is None:
        out["vegetation_type"] = out["vegetation_type_worldcover"]
    else:
        out[veg_col] = (
            out[veg_col]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan})
            .fillna(out["vegetation_type_worldcover"])
        )
    return out


def _add_copdem_slope(properties_df: pd.DataFrame, api_connectors: dict | None) -> pd.DataFrame:
    out = properties_df.copy()
    dem = sample_copdem_points(out, lat_col="lat", lon_col="lon", api_connectors=api_connectors)
    out = pd.concat([out, dem], axis=1)

    slope_col = _first_present_col(out, ["slope_deg", "slope_degrees", "slope"])
    if slope_col is None:
        out["slope_deg"] = pd.to_numeric(out["dem_slope_deg"], errors="coerce")
    else:
        out[slope_col] = (
            pd.to_numeric(out[slope_col], errors="coerce")
            .fillna(pd.to_numeric(out["dem_slope_deg"], errors="coerce"))
        )
    return out


def _nearest_cell_features(properties_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
    p_lat = properties_df["lat"].to_numpy(dtype=float)
    p_lon = properties_df["lon"].to_numpy(dtype=float)
    c_lat = prediction_df["lat_center"].to_numpy(dtype=float)
    c_lon = prediction_df["lon_center"].to_numpy(dtype=float)

    nearest_idx = []
    for lat, lon in zip(p_lat, p_lon):
        # Fast approximation for nearest cell in this limited regional grid.
        d2 = (c_lat - lat) ** 2 + (c_lon - lon) ** 2
        nearest_idx.append(int(np.argmin(d2)))

    nn = prediction_df.iloc[nearest_idx].reset_index(drop=True)
    out = properties_df.reset_index(drop=True).copy()
    out["nearest_cell_id"] = nn["cell_id"].astype(str)
    out["nearest_cell_lat"] = nn["lat_center"].astype(float)
    out["nearest_cell_lon"] = nn["lon_center"].astype(float)
    out["cell_risk_score"] = nn["risk_score"].astype(float)
    out["cell_risk_class"] = nn["risk_class"].astype(str)
    out["cell_ndvi"] = pd.to_numeric(nn.get("ndvi", np.nan), errors="coerce")
    out["cell_veg_dryness_index"] = pd.to_numeric(nn.get("veg_dryness_index", np.nan), errors="coerce")
    out["cell_fwi_proxy"] = pd.to_numeric(nn.get("fwi_proxy", np.nan), errors="coerce")
    return out


def _vegetation_risk(df: pd.DataFrame) -> pd.Series:
    # NDVI-based component (lower NDVI => higher fuel risk proxy).
    ndvi_col = _first_present_col(df, ["ndvi", "property_ndvi"])
    ndvi = pd.to_numeric(df[ndvi_col], errors="coerce") if ndvi_col else df["cell_ndvi"]
    ndvi = ndvi.fillna(df["cell_ndvi"]).fillna(0.5).clip(-1, 1)
    ndvi_risk = 1.0 - ((ndvi + 1.0) / 2.0)

    # Optional categorical vegetation override.
    veg_type_col = _first_present_col(df, ["vegetation_type", "veg_type", "land_cover"])
    if veg_type_col:
        mapped = (
            df[veg_type_col]
            .astype(str)
            .str.lower()
            .map(VEGETATION_TYPE_RISK)
            .astype(float)
        )
        mapped = mapped.fillna(ndvi_risk)
        return (0.6 * ndvi_risk + 0.4 * mapped).clip(0, 1)
    return ndvi_risk.clip(0, 1)


def _slope_risk(df: pd.DataFrame) -> pd.Series:
    slope_col = _first_present_col(df, ["slope_deg", "slope_degrees", "slope"])
    slope = pd.to_numeric(df[slope_col], errors="coerce") if slope_col else pd.Series(np.nan, index=df.index)
    slope = slope.fillna(5.0).clip(0, 60)
    return (slope / 45.0).clip(0, 1)


def _dryness_risk(df: pd.DataFrame) -> pd.Series:
    dryness_col = _first_present_col(df, ["dryness_index", "veg_dryness_index"])
    dryness = pd.to_numeric(df[dryness_col], errors="coerce") if dryness_col else df["cell_veg_dryness_index"]
    dryness = dryness.fillna(df["cell_veg_dryness_index"]).fillna(40.0).clip(0, 100) / 100.0

    fwi = pd.to_numeric(df["cell_fwi_proxy"], errors="coerce").fillna(10.0).clip(0, 50) / 50.0
    return (0.6 * dryness + 0.4 * fwi).clip(0, 1)


def _min_distance_to_fires_km(properties_df: pd.DataFrame, fires_df: pd.DataFrame) -> np.ndarray:
    if fires_df.empty:
        return np.full(len(properties_df), np.inf, dtype=float)

    f_lat = pd.to_numeric(fires_df["latitude"], errors="coerce").to_numpy(dtype=float)
    f_lon = pd.to_numeric(fires_df["longitude"], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(f_lat) | np.isnan(f_lon))
    f_lat = f_lat[valid]
    f_lon = f_lon[valid]
    if len(f_lat) == 0:
        return np.full(len(properties_df), np.inf, dtype=float)

    p_lat = properties_df["lat"].to_numpy(dtype=float)
    p_lon = properties_df["lon"].to_numpy(dtype=float)
    min_dist = np.full(len(properties_df), np.inf, dtype=float)

    batch = 200
    for i in range(0, len(properties_df), batch):
        j = min(i + batch, len(properties_df))
        lat_block = p_lat[i:j][:, None]
        lon_block = p_lon[i:j][:, None]
        d = _haversine_km(lat_block, lon_block, f_lat[None, :], f_lon[None, :])
        min_dist[i:j] = d.min(axis=1)
    return min_dist


def _proximity_risk(distance_km: np.ndarray) -> np.ndarray:
    # Fast drop-off after ~30km; very near fires are highest risk.
    out = np.exp(-distance_km / 30.0)
    out[distance_km <= 1.0] = 1.0
    out[np.isinf(distance_km)] = 0.0
    return np.clip(out, 0, 1)


def _classify(score: float, thresholds: dict) -> str:
    if score < thresholds["low"]:
        return "low"
    if score < thresholds["moderate"]:
        return "moderate"
    if score < thresholds["high"]:
        return "high"
    return "extreme"


def score_properties(
    properties_path: str | Path,
    prediction_date: str,
    lookback_days: int = 365,
    output_path: str | Path | None = None,
) -> str:
    """
    Score wildfire risk at property level.

    Inputs:
    - property CSV with lat/lon (+ optional slope_deg, ndvi, vegetation_type)
    - grid-level daily prediction file for the same date
    - historical FIRMS events to compute proximity-to-fire risk
    """
    settings = load_settings()
    prediction_path = Path(settings.paths.curated_dir) / "daily_predictions" / f"predictions_real_{prediction_date}.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found for date {prediction_date}: {prediction_path}")

    properties_df = _resolve_property_coordinates(read_csv(properties_path))
    properties_df = _add_worldcover_vegetation(properties_df, settings.api_connectors)
    properties_df = _add_copdem_slope(properties_df, settings.api_connectors)
    pred_df = read_csv(prediction_path)
    required_pred = {"cell_id", "lat_center", "lon_center", "risk_score", "risk_class"}
    missing = required_pred - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")

    out = _nearest_cell_features(properties_df, pred_df)
    out["vegetation_risk_component"] = _vegetation_risk(out)
    out["slope_risk_component"] = _slope_risk(out)
    out["dryness_risk_component"] = _dryness_risk(out)

    firms_path = Path(settings.real_data["firms_events_file"])
    fires_df = read_csv(firms_path) if firms_path.exists() else pd.DataFrame()
    if not fires_df.empty and "event_date" in fires_df.columns:
        cutoff = pd.to_datetime(prediction_date) - pd.Timedelta(days=lookback_days)
        fire_dt = pd.to_datetime(fires_df["event_date"], errors="coerce")
        fires_df = fires_df[fire_dt >= cutoff].copy()

    out["distance_to_recent_fire_km"] = _min_distance_to_fires_km(out, fires_df)
    out["proximity_risk_component"] = _proximity_risk(out["distance_to_recent_fire_km"].to_numpy(dtype=float))

    # Weighted property-level risk model.
    out["property_risk_score"] = (
        0.25 * out["vegetation_risk_component"]
        + 0.20 * out["slope_risk_component"]
        + 0.35 * out["dryness_risk_component"]
        + 0.20 * out["proximity_risk_component"]
    ).clip(0, 1)

    thresholds = settings.training.prediction_thresholds
    out["property_risk_class"] = out["property_risk_score"].apply(lambda x: _classify(float(x), thresholds))
    out["prediction_date"] = prediction_date
    out["fire_lookback_days"] = int(lookback_days)

    if output_path is None:
        out_dir = ensure_dir(Path(settings.paths.curated_dir) / "property_predictions")
        output_path = Path(out_dir) / f"property_risk_{prediction_date}.csv"
    write_csv(out, output_path)
    return str(output_path)
