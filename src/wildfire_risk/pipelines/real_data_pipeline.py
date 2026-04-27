from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import lightgbm as lgb

from wildfire_risk.config import load_settings
from wildfire_risk.utils.io import ensure_dir, write_csv, read_csv, write_json
from wildfire_risk.data.grid import build_grid
from wildfire_risk.data.schema_real import REAL_FEATURE_COLUMNS
from wildfire_risk.ingestion.openmeteo import OpenMeteoClient
from wildfire_risk.ingestion.firms import FIRMSClient


def _bbox_from_settings(settings) -> str:
    """Build FIRMS/Open APIs bounding-box string as: lon_min,lat_min,lon_max,lat_max."""
    return ",".join(
        [
            str(settings.grid.lon_min),
            str(settings.grid.lat_min),
            str(settings.grid.lon_max),
            str(settings.grid.lat_max),
        ]
    )


def build_grid_from_settings(settings) -> pd.DataFrame:
    """Create the spatial modeling grid from config bounds + resolution."""
    return build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )


def fetch_real_sources(start_date: str, end_date: str, forecast_days: int | None = None, firms_source: str | None = None) -> dict:
    """
    Fetch and persist all external raw sources used by the real-data pipeline.

    Output files:
    - Open-Meteo history/forecast weather
    - NASA FIRMS detections mapped to grid cells
    """
    settings = load_settings()
    grid_df = build_grid_from_settings(settings)
    forecast_days = forecast_days or settings.real_data["forecast_days"]
    firms_source = firms_source or settings.real_data["firms_source"]

    # Ensure raw directory exists before writing source CSVs.
    raw_dir = ensure_dir(settings.paths.raw_dir)
    openmeteo = OpenMeteoClient()

    # Weather pulls are done at each grid-cell center.
    hist_df = openmeteo.fetch_grid_history(grid_df, start_date=start_date, end_date=end_date)
    fcst_df = openmeteo.fetch_grid_forecast(grid_df, forecast_days=forecast_days)

    hist_path = Path(settings.real_data["openmeteo_history_file"])
    fcst_path = Path(settings.real_data["openmeteo_forecast_file"])
    write_csv(hist_df, hist_path)
    write_csv(fcst_df, fcst_path)

    # FIRMS requires an API key in env; fail early with a clear message.
    api_key = os.getenv("FIRMS_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("FIRMS_API_KEY is not set")

    firms = FIRMSClient(map_key=api_key)
    bbox = _bbox_from_settings(settings)
    # FIRMS API limits day ranges, so this method internally chunks requests.
    firms_df = firms.fetch_area_range_chunked(
        source=firms_source,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
    )
    firms_df = firms.attach_cells(firms_df, grid_df=grid_df, resolution_deg=settings.grid.resolution_deg)
    firms_path = Path(settings.real_data["firms_events_file"])
    write_csv(firms_df, firms_path)

    return {
        "history_path": str(hist_path),
        "forecast_path": str(fcst_path),
        "firms_path": str(firms_path),
        "grid_rows": int(len(grid_df)),
        "history_rows": int(len(hist_df)),
        "forecast_rows": int(len(fcst_df)),
        "firms_rows": int(len(firms_df)),
    }


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical calendar features used by the model.

    Using sin/cos encodings avoids discontinuities (e.g., Dec -> Jan).
    """
    out = df.copy()
    dt = pd.to_datetime(out["date"])
    doy = dt.dt.dayofyear
    month = dt.dt.month
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def _compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive weather-driven fire-risk features from raw daily weather columns.

    Adds:
    - vpd_kpa: approximate vapor pressure deficit
    - ffmc, dmc, dc, isi, bui, fwi: Canadian Fire Weather Index System
    - fwi_proxy: compatibility alias pointing to real FWI
    """
    out = df.copy()
    out["t2m_max_c"] = pd.to_numeric(out["t2m_max_c"], errors="coerce")
    out["rh_min_pct"] = pd.to_numeric(out["rh_min_pct"], errors="coerce")
    out["wind10m_max_ms"] = pd.to_numeric(out["wind10m_max_ms"], errors="coerce")
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"])

    rh = np.clip(out["rh_min_pct"].astype(float), 1.0, 100.0)
    t = out["t2m_max_c"].astype(float)

    # Approximate VPD from temp and RH
    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    ea = es * (rh / 100.0)
    out["vpd_kpa"] = (es - ea).round(3)

    # Northern Hemisphere monthly factors for DMC/DC.
    dmc_day_length = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0], dtype=float)
    dc_drying_factor = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6], dtype=float)

    def _ffmc_step(prev_ffmc: float, temp_c: float, rh_pct: float, wind_kmh: float, rain_mm: float) -> float:
        ffmc_prev = float(np.clip(prev_ffmc, 0.0, 101.0))
        temp_c = float(np.nan_to_num(temp_c, nan=20.0))
        rh_pct = float(np.clip(np.nan_to_num(rh_pct, nan=45.0), 0.0, 100.0))
        wind_kmh = float(np.clip(np.nan_to_num(wind_kmh, nan=10.0), 0.0, None))
        rain_mm = float(np.clip(np.nan_to_num(rain_mm, nan=0.0), 0.0, None))

        mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
        if rain_mm > 0.5:
            rf = rain_mm - 0.5
            if mo > 150.0:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
                mo = mo + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)
            else:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
            mo = min(mo, 250.0)

        ed = (
            0.942 * (rh_pct ** 0.679)
            + 11.0 * np.exp((rh_pct - 100.0) / 10.0)
            + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh_pct))
        )
        if mo < ed:
            ew = (
                0.618 * (rh_pct ** 0.753)
                + 10.0 * np.exp((rh_pct - 100.0) / 10.0)
                + 0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh_pct))
            )
            kl = 0.424 * (1.0 - ((100.0 - rh_pct) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (
                1.0 - ((100.0 - rh_pct) / 100.0) ** 8
            )
            kw = kl * 0.581 * np.exp(0.0365 * temp_c)
            mo = ew - (ew - mo) * (10.0 ** (-kw))
        else:
            kl = 0.424 * (1.0 - (rh_pct / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind_kmh) * (1.0 - (rh_pct / 100.0) ** 8)
            kw = kl * 0.581 * np.exp(0.0365 * temp_c)
            mo = ed + (mo - ed) * (10.0 ** (-kw))

        ffmc = 59.5 * (250.0 - mo) / (147.2 + mo)
        return float(np.clip(ffmc, 0.0, 101.0))

    def _dmc_step(prev_dmc: float, month: int, temp_c: float, rh_pct: float, rain_mm: float) -> float:
        dmc_prev = float(max(prev_dmc, 0.0))
        temp_c = float(np.nan_to_num(temp_c, nan=20.0))
        rh_pct = float(np.clip(np.nan_to_num(rh_pct, nan=45.0), 0.0, 100.0))
        rain_mm = float(np.clip(np.nan_to_num(rain_mm, nan=0.0), 0.0, None))
        month_idx = int(np.clip(month - 1, 0, 11))

        rk = 1.894 * (temp_c + 1.1) * (100.0 - rh_pct) * dmc_day_length[month_idx] * 1e-6
        rk = max(rk, 0.0)

        if rain_mm > 1.5:
            re = 0.92 * rain_mm - 1.27
            mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
            if dmc_prev <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc_prev)
            elif dmc_prev <= 65.0:
                b = 14.0 - 1.3 * np.log(dmc_prev)
            else:
                b = 6.2 * np.log(dmc_prev) - 17.2
            mr = mo + (1000.0 * re) / (48.77 + b * re)
            dmc_r = 43.43 * (5.6348 - np.log(max(mr - 20.0, 1e-6)))
            dmc_new = max(dmc_r, 0.0) + 100.0 * rk
        else:
            dmc_new = dmc_prev + 100.0 * rk

        return float(max(dmc_new, 0.0))

    def _dc_step(prev_dc: float, month: int, temp_c: float, rain_mm: float) -> float:
        dc_prev = float(max(prev_dc, 0.0))
        temp_c = float(np.nan_to_num(temp_c, nan=20.0))
        rain_mm = float(np.clip(np.nan_to_num(rain_mm, nan=0.0), 0.0, None))
        month_idx = int(np.clip(month - 1, 0, 11))

        if rain_mm > 2.8:
            rd = 0.83 * rain_mm - 1.27
            qo = 800.0 * np.exp(-dc_prev / 400.0)
            qr = qo + 3.937 * rd
            dc_r = 400.0 * np.log(800.0 / max(qr, 1e-6))
            dc_r = max(dc_r, 0.0)
        else:
            dc_r = dc_prev

        v = 0.36 * (temp_c + 2.8) + dc_drying_factor[month_idx]
        v = max(v, 0.0)
        dc_new = dc_r + 0.5 * v
        return float(max(dc_new, 0.0))

    def _isi_from_ffmc(ffmc: float, wind_kmh: float) -> float:
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 19.115 * np.exp(-0.1386 * mo) * (1.0 + (mo**5.31) / 4.93e7)
        return float(ff * np.exp(0.05039 * wind_kmh))

    def _bui_from_dmc_dc(dmc: float, dc: float) -> float:
        if dmc <= 0.4 * dc:
            denom = dmc + 0.4 * dc
            bui = (0.8 * dmc * dc / denom) if denom > 0 else 0.0
        else:
            denom = dmc + 0.4 * dc
            bui = dmc - (1.0 - (0.8 * dc / denom)) * (0.92 + (0.0114 * dmc) ** 1.7)
        return float(max(bui, 0.0))

    def _fwi_from_isi_bui(isi: float, bui: float) -> float:
        if bui <= 80.0:
            fd = 0.626 * (bui**0.809) + 2.0
        else:
            fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
        b = 0.1 * isi * fd
        if b <= 1.0:
            return float(max(b, 0.0))
        return float(np.exp(2.72 * ((0.434 * np.log(b)) ** 0.647)))

    ffmc_vals = np.full(len(out), np.nan, dtype=float)
    dmc_vals = np.full(len(out), np.nan, dtype=float)
    dc_vals = np.full(len(out), np.nan, dtype=float)
    isi_vals = np.full(len(out), np.nan, dtype=float)
    bui_vals = np.full(len(out), np.nan, dtype=float)
    fwi_vals = np.full(len(out), np.nan, dtype=float)

    if "cell_id" in out.columns:
        grouped = out.sort_values(["cell_id", "date"]).groupby("cell_id", sort=False)
        for _, g in grouped:
            ffmc_prev = 85.0
            dmc_prev = 6.0
            dc_prev = 15.0
            for idx, row in g.iterrows():
                month = int(pd.Timestamp(row["date"]).month)
                wind_kmh = float(np.clip(np.nan_to_num(row["wind10m_max_ms"], nan=3.0) * 3.6, 0.0, None))
                ffmc = _ffmc_step(ffmc_prev, row["t2m_max_c"], row["rh_min_pct"], wind_kmh, row["precip_mm"])
                dmc = _dmc_step(dmc_prev, month, row["t2m_max_c"], row["rh_min_pct"], row["precip_mm"])
                dc = _dc_step(dc_prev, month, row["t2m_max_c"], row["precip_mm"])
                isi = _isi_from_ffmc(ffmc, wind_kmh)
                bui = _bui_from_dmc_dc(dmc, dc)
                fwi = _fwi_from_isi_bui(isi, bui)

                ffmc_vals[out.index.get_loc(idx)] = ffmc
                dmc_vals[out.index.get_loc(idx)] = dmc
                dc_vals[out.index.get_loc(idx)] = dc
                isi_vals[out.index.get_loc(idx)] = isi
                bui_vals[out.index.get_loc(idx)] = bui
                fwi_vals[out.index.get_loc(idx)] = fwi

                ffmc_prev, dmc_prev, dc_prev = ffmc, dmc, dc
    else:
        # Fallback for one-off frames without cell_id: process sequentially.
        ffmc_prev = 85.0
        dmc_prev = 6.0
        dc_prev = 15.0
        for i, row in out.sort_values("date").iterrows():
            month = int(pd.Timestamp(row["date"]).month)
            wind_kmh = float(np.clip(np.nan_to_num(row["wind10m_max_ms"], nan=3.0) * 3.6, 0.0, None))
            ffmc = _ffmc_step(ffmc_prev, row["t2m_max_c"], row["rh_min_pct"], wind_kmh, row["precip_mm"])
            dmc = _dmc_step(dmc_prev, month, row["t2m_max_c"], row["rh_min_pct"], row["precip_mm"])
            dc = _dc_step(dc_prev, month, row["t2m_max_c"], row["precip_mm"])
            isi = _isi_from_ffmc(ffmc, wind_kmh)
            bui = _bui_from_dmc_dc(dmc, dc)
            fwi = _fwi_from_isi_bui(isi, bui)

            pos = out.index.get_loc(i)
            ffmc_vals[pos] = ffmc
            dmc_vals[pos] = dmc
            dc_vals[pos] = dc
            isi_vals[pos] = isi
            bui_vals[pos] = bui
            fwi_vals[pos] = fwi
            ffmc_prev, dmc_prev, dc_prev = ffmc, dmc, dc

    out["ffmc"] = np.round(ffmc_vals, 2)
    out["dmc"] = np.round(dmc_vals, 2)
    out["dc"] = np.round(dc_vals, 2)
    out["isi"] = np.round(isi_vals, 2)
    out["bui"] = np.round(bui_vals, 2)
    out["fwi"] = np.round(fwi_vals, 2)
    # Keep historical column name for compatibility with existing models/scripts.
    out["fwi_proxy"] = out["fwi"]
    return out


def _aggregate_fire_history(firms_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate FIRMS event-level detections into daily per-cell fire totals."""
    if firms_df.empty:
        return pd.DataFrame(columns=["date", "cell_id", "fire_count_today", "frp_sum_today"])

    daily = (
        firms_df.groupby(["event_date", "cell_id"], as_index=False)
        .agg(
            fire_count_today=("detection_id", "count"),
            frp_sum_today=("frp_mw", "sum"),
        )
        .rename(columns={"event_date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["frp_sum_today"] = daily["frp_sum_today"].round(2)
    return daily


def _complete_base(weather_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete date x cell panel and left-join weather onto it.

    This guarantees every grid cell exists for every weather date even if
    some weather observations are missing.
    """
    dates = pd.to_datetime(weather_df["date"]).sort_values().unique()
    base = (
        pd.MultiIndex.from_product([dates, grid_df["cell_id"]], names=["date", "cell_id"])
        .to_frame(index=False)
        .merge(grid_df[["cell_id", "lat_center", "lon_center"]], on="cell_id", how="left")
    )
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    out = base.merge(
        weather_df[["date", "cell_id", "t2m_max_c", "rh_min_pct", "wind10m_max_ms", "precip_mm", "lat_center", "lon_center"]],
        on=["date", "cell_id", "lat_center", "lon_center"],
        how="left",
    )
    return out.sort_values(["cell_id", "date"]).reset_index(drop=True)


def _merge_ndvi(base_df: pd.DataFrame, ndvi_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge NDVI observations and derive vegetation stress indicators.

    Adds:
    - ndvi
    - ndvi_anom_30d
    - veg_dryness_index
    """
    out = base_df.copy()
    if ndvi_df is None or ndvi_df.empty:
        out["ndvi"] = np.nan
    else:
        ndvi = ndvi_df.copy()
        ndvi["date"] = pd.to_datetime(ndvi["date"])
        ndvi["cell_id"] = ndvi["cell_id"].astype(str)
        ndvi["ndvi"] = pd.to_numeric(ndvi["ndvi"], errors="coerce")
        ndvi = ndvi.dropna(subset=["date", "cell_id", "ndvi"])
        out = out.merge(ndvi[["date", "cell_id", "ndvi"]], on=["date", "cell_id"], how="left")

    # Fill missing NDVI per cell so all rows have usable vegetation state.
    out = out.sort_values(["cell_id", "date"]).copy()
    out["ndvi"] = out.groupby("cell_id")["ndvi"].ffill().bfill()
    out["ndvi"] = out["ndvi"].fillna(0.5).clip(-1, 1).round(3)

    # NDVI anomaly is current value relative to each cell's recent baseline.
    rolling_mean = (
        out.groupby("cell_id")["ndvi"]
        .rolling(30, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["ndvi_anom_30d"] = (out["ndvi"] - rolling_mean).fillna(0.0).round(3)

    # Composite dryness signal blending vegetation and atmospheric stress.
    dryness = (
        0.45 * (1 - ((out["ndvi"] + 1.0) / 2.0))
        + 0.35 * np.clip(out["vpd_kpa"] / 4.0, 0, 1)
        + 0.20 * np.clip(out["t2m_max_c"] / 40.0, 0, 1)
    )
    out["veg_dryness_index"] = np.clip(dryness * 100, 0, 100).round(2)
    return out


def _merge_smap(base_df: pd.DataFrame, smap_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge SMAP soil moisture and produce normalized soil moisture context.

    Adds:
    - soil_moisture_surface_m3m3
    - soil_moisture_pct_of_normal
    """
    out = base_df.copy()
    if smap_df is None or smap_df.empty:
        out["soil_moisture_surface_m3m3"] = np.nan
        out["soil_moisture_pct_of_normal"] = np.nan
    else:
        sm = smap_df.copy()
        sm["date"] = pd.to_datetime(sm["date"])
        sm["cell_id"] = sm["cell_id"].astype(str)
        sm["soil_moisture_surface_m3m3"] = pd.to_numeric(sm["soil_moisture_surface_m3m3"], errors="coerce")
        if "soil_moisture_pct_of_normal" in sm.columns:
            sm["soil_moisture_pct_of_normal"] = pd.to_numeric(sm["soil_moisture_pct_of_normal"], errors="coerce")
        sm = sm.dropna(subset=["date", "cell_id", "soil_moisture_surface_m3m3"])
        keep = ["date", "cell_id", "soil_moisture_surface_m3m3"]
        if "soil_moisture_pct_of_normal" in sm.columns:
            keep.append("soil_moisture_pct_of_normal")
        out = out.merge(sm[keep], on=["date", "cell_id"], how="left")

    # Fill soil moisture gaps with nearest values in each cell's timeline.
    out = out.sort_values(["cell_id", "date"]).copy()
    out["soil_moisture_surface_m3m3"] = (
        out.groupby("cell_id")["soil_moisture_surface_m3m3"].ffill().bfill().fillna(0.25).clip(0, 1).round(3)
    )

    if "soil_moisture_pct_of_normal" not in out.columns:
        out["soil_moisture_pct_of_normal"] = np.nan

    # Use rolling baseline to estimate "% of normal" when not provided.
    rolling_sm = (
        out.groupby("cell_id")["soil_moisture_surface_m3m3"]
        .rolling(30, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    derived_pct = np.where(
        rolling_sm.notna() & (rolling_sm > 0),
        (out["soil_moisture_surface_m3m3"] / rolling_sm) * 100,
        np.nan
    )
    out["soil_moisture_pct_of_normal"] = (
        out["soil_moisture_pct_of_normal"]
        .fillna(pd.Series(derived_pct, index=out.index))
        .fillna(100.0)
        .clip(0, 300)
        .round(2)
    )
    return out


def _merge_lightning(base_df: pd.DataFrame, lightning_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge lightning activity features, defaulting to zeros when unavailable.

    Adds:
    - lightning_count
    - dry_lightning_count
    - lightning_probability
    """
    out = base_df.copy()
    if lightning_df is None or lightning_df.empty:
        out["lightning_count"] = 0.0
        out["dry_lightning_count"] = 0.0
        out["lightning_probability"] = 0.0
    else:
        li = lightning_df.copy()
        li["date"] = pd.to_datetime(li["date"])
        li["cell_id"] = li["cell_id"].astype(str)
        li["lightning_count"] = pd.to_numeric(li["lightning_count"], errors="coerce").fillna(0).clip(0, None)
        if "dry_lightning_count" in li.columns:
            li["dry_lightning_count"] = pd.to_numeric(li["dry_lightning_count"], errors="coerce").fillna(0).clip(0, None)
        else:
            li["dry_lightning_count"] = 0.0
        if "lightning_probability" in li.columns:
            li["lightning_probability"] = pd.to_numeric(li["lightning_probability"], errors="coerce").fillna(0).clip(0, 1)
        else:
            li["lightning_probability"] = np.where(li["lightning_count"] > 0, 1.0, 0.0)

        keep = ["date", "cell_id", "lightning_count", "dry_lightning_count", "lightning_probability"]
        out = out.merge(li[keep], on=["date", "cell_id"], how="left")

    out["lightning_count"] = out["lightning_count"].fillna(0.0)
    out["dry_lightning_count"] = out["dry_lightning_count"].fillna(0.0)
    out["lightning_probability"] = out["lightning_probability"].fillna(0.0).clip(0, 1)
    return out


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add short-term rolling and lagged features that capture recent conditions.

    Adds precipitation windows, dry-day count, and prior lightning windows.
    """
    out = df.sort_values(["cell_id", "date"]).copy()
    g = out.groupby("cell_id", group_keys=False)

    out["precip_3d_mm"] = g["precip_mm"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    out["precip_7d_mm"] = g["precip_mm"].rolling(7, min_periods=1).sum().reset_index(level=0, drop=True)
    out["precip_30d_mm"] = g["precip_mm"].rolling(30, min_periods=1).sum().reset_index(level=0, drop=True)
    out["dry_days_30d"] = (
        g["precip_mm"].rolling(30, min_periods=1).apply(lambda x: float((x.fillna(0) < 1.0).sum()), raw=False)
        .reset_index(level=0, drop=True)
    )
    # Shift(1) ensures these are strictly prior-day history features.
    out["lightning_count_prev_3d"] = (
        g["lightning_count"].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)
    )
    out["dry_lightning_count_prev_3d"] = (
        g["dry_lightning_count"].shift(1).rolling(3, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0)
    )
    return out


def _add_fire_history_and_labels(df: pd.DataFrame, fire_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fire-history predictors and next-day supervision labels.

    Labels are aligned so each row predicts fire outcomes for date + 1 day.
    """
    out = df.copy()
    fire_daily = fire_daily_df.copy()
    if fire_daily.empty:
        out["fire_count_today"] = 0
        out["frp_sum_today"] = 0.0
    else:
        out = out.merge(fire_daily, on=["date", "cell_id"], how="left")
        out["fire_count_today"] = out["fire_count_today"].fillna(0).astype(int)
        out["frp_sum_today"] = out["frp_sum_today"].fillna(0.0).round(2)

    g = out.groupby("cell_id", group_keys=False)
    out["fire_count_prev_7d"] = (
        g["fire_count_today"].shift(1).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0).astype(int)
    )
    out["frp_sum_prev_7d"] = (
        g["frp_sum_today"].shift(1).rolling(7, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0).round(2)
    )

    # Shift today's fire metrics back one day to create next-day labels.
    shifted = out[["date", "cell_id", "fire_count_today", "frp_sum_today"]].copy()
    shifted["date"] = shifted["date"] - pd.Timedelta(days=1)
    shifted = shifted.rename(
        columns={
            "fire_count_today": "label_fire_count_next_1d",
            "frp_sum_today": "label_frp_sum_next_1d",
        }
    )
    out = out.merge(shifted, on=["date", "cell_id"], how="left")
    out["label_fire_count_next_1d"] = out["label_fire_count_next_1d"].fillna(0).astype(int)
    out["label_frp_sum_next_1d"] = out["label_frp_sum_next_1d"].fillna(0.0).round(2)
    out["label_fire_next_1d"] = (out["label_fire_count_next_1d"] > 0).astype(int)
    return out


def build_real_training_table() -> str:
    """
    Build the full curated training table from weather + FIRMS + NDVI + SMAP + lightning.

    This is the central feature-engineering entrypoint for real-data model training.
    """
    settings = load_settings()
    grid_df = build_grid_from_settings(settings)

    history_path = Path(settings.real_data["openmeteo_history_file"])
    firms_path = Path(settings.real_data["firms_events_file"])
    ndvi_path = Path(settings.real_data["ndvi_observations_file"])
    smap_path = Path(settings.real_data["smap_observations_file"])
    lightning_path = Path(settings.real_data["lightning_observations_file"])
    output_path = Path(settings.real_data["training_table_real_file"])

    # Weather is required; the other sources are optional and safely default empty.
    weather_df = read_csv(history_path)
    firms_df = read_csv(firms_path) if firms_path.exists() else pd.DataFrame()
    ndvi_df = read_csv(ndvi_path) if ndvi_path.exists() else pd.DataFrame()
    smap_df = read_csv(smap_path) if smap_path.exists() else pd.DataFrame()
    lightning_df = read_csv(lightning_path) if lightning_path.exists() else pd.DataFrame()

    # Feature engineering stack order matters: weather first, then merges, then rolling/lags.
    base = _complete_base(weather_df, grid_df)
    base = _compute_weather_features(base)
    base = _merge_ndvi(base, ndvi_df)
    base = _merge_smap(base, smap_df)
    base = _merge_lightning(base, lightning_df)
    base = _add_time_features(base)
    base = _add_rolling_features(base)

    # FIRMS events are aggregated to daily cell counts/FRP, then used for labels + lag features.
    if not firms_df.empty:
        firms_df["acq_datetime_utc"] = pd.to_datetime(firms_df["acq_datetime_utc"], utc=True)
    fire_daily = _aggregate_fire_history(firms_df)
    training_df = _add_fire_history_and_labels(base, fire_daily)
    write_csv(training_df, output_path)
    return str(output_path)


def _build_real_model(random_state: int = 42):
    """
    Construct the train-time model pipeline.

    Pipeline:
    - median imputation for missing numeric features
    - LightGBM binary classifier
    """
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


def train_real_model() -> dict:
    """
    Train the real-data model, evaluate on holdout dates, and save artifacts.

    Returns model path, manifest path, and evaluation metrics.
    """
    settings = load_settings()
    input_path = Path(settings.real_data["training_table_real_file"])
    model_path = Path(settings.paths.models_dir) / settings.training.model_filename_real
    ensure_dir(model_path.parent)
    ensure_dir(settings.paths.manifests_dir)

    df = read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    # Time-based split to avoid leakage from future to past.
    unique_dates = sorted(df["date"].dt.date.unique())
    split_idx = max(1, int(len(unique_dates) * 0.8))
    split_date = pd.Timestamp(unique_dates[split_idx - 1])

    train_df = df[df["date"] <= split_date].copy()
    valid_df = df[df["date"] > split_date].copy()
    if valid_df.empty:
        valid_df = train_df.copy()

    X_train = train_df[REAL_FEATURE_COLUMNS]
    y_train = train_df["label_fire_next_1d"].astype(int)
    X_valid = valid_df[REAL_FEATURE_COLUMNS]
    y_valid = valid_df["label_fire_next_1d"].astype(int)

    model = _build_real_model(settings.training.random_state)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_valid)[:, 1]
    metrics = {
        "rows_train": int(len(train_df)),
        "rows_valid": int(len(valid_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_valid": float(y_valid.mean()),
        "average_precision": float(average_precision_score(y_valid, probs)) if len(set(y_valid)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_valid, probs)) if len(set(y_valid)) > 1 else None,
        "brier_score": float(brier_score_loss(y_valid, probs)),
    }

    # Persist model binary + manifest metadata for inference reproducibility.
    dump(model, model_path)
    manifest_path = Path(settings.paths.manifests_dir) / "model_manifest_real.json"
    write_json(
        {
            "model_filename": settings.training.model_filename_real,
            "metrics": metrics,
            "feature_columns": REAL_FEATURE_COLUMNS,
        },
        manifest_path,
    )
    return {"model_path": str(model_path), "manifest_path": str(manifest_path), "metrics": metrics}


def predict_real_daily(prediction_date: str) -> str:
    """
    Generate per-cell wildfire risk for one forecast date.

    Uses forecast weather + recent historical context to reconstruct all model
    features needed at inference time.
    """
    settings = load_settings()
    model_path = Path(settings.paths.models_dir) / settings.training.model_filename_real
    forecast_path = Path(settings.real_data["openmeteo_forecast_file"])
    history_path = Path(settings.real_data["training_table_real_file"])
    out_dir = ensure_dir(Path(settings.paths.curated_dir) / "daily_predictions")

    model = load(model_path)
    forecast_df = read_csv(forecast_path)
    history_df = read_csv(history_path)

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    history_df["date"] = pd.to_datetime(history_df["date"])

    forecast_df = _compute_weather_features(forecast_df)
    forecast_df = _add_time_features(forecast_df)

    # Use history strictly before prediction date to compute lag/rolling inputs.
    pred_date = pd.to_datetime(prediction_date)
    latest_history = history_df[history_df["date"] < pred_date].copy()

    rows = []
    for row in forecast_df[forecast_df["date"] == pred_date].itertuples(index=False):
        # Build one inference feature row per grid cell.
        cell_hist = latest_history[latest_history["cell_id"] == row.cell_id].sort_values("date")
        precip_tail = cell_hist["precip_mm"].tail(29).tolist() + [float(row.precip_mm)]
        fire_tail = cell_hist["fire_count_today"].tail(7).tolist() if "fire_count_today" in cell_hist.columns else []
        frp_tail = cell_hist["frp_sum_today"].tail(7).tolist() if "frp_sum_today" in cell_hist.columns else []
        ndvi_value = float(cell_hist["ndvi"].dropna().iloc[-1]) if "ndvi" in cell_hist.columns and cell_hist["ndvi"].dropna().shape[0] else 0.5
        ndvi_tail = cell_hist["ndvi"].dropna().tail(30).tolist() if "ndvi" in cell_hist.columns else []
        ndvi_mean_30 = float(np.mean(ndvi_tail)) if ndvi_tail else ndvi_value
        ndvi_anom_30d = float(ndvi_value - ndvi_mean_30)

        veg_dryness_index = float(np.clip(
            (
                0.45 * (1 - ((ndvi_value + 1.0) / 2.0))
                + 0.35 * np.clip(float(row.vpd_kpa) / 4.0, 0, 1)
                + 0.20 * np.clip(float(row.t2m_max_c) / 40.0, 0, 1)
            ) * 100,
            0,
            100
        ))

        # Fallback defaults keep inference robust when optional source history is sparse.
        if "soil_moisture_surface_m3m3" in cell_hist.columns and cell_hist["soil_moisture_surface_m3m3"].dropna().shape[0]:
            sm_value = float(cell_hist["soil_moisture_surface_m3m3"].dropna().iloc[-1])
        else:
            sm_value = 0.25

        if "soil_moisture_pct_of_normal" in cell_hist.columns and cell_hist["soil_moisture_pct_of_normal"].dropna().shape[0]:
            sm_pct = float(cell_hist["soil_moisture_pct_of_normal"].dropna().iloc[-1])
        else:
            sm_tail = cell_hist["soil_moisture_surface_m3m3"].dropna().tail(30).tolist() if "soil_moisture_surface_m3m3" in cell_hist.columns else []
            sm_mean = float(np.mean(sm_tail)) if sm_tail else sm_value
            sm_pct = float((sm_value / sm_mean) * 100) if sm_mean > 0 else 100.0

        if "lightning_count" in cell_hist.columns and cell_hist["lightning_count"].dropna().shape[0]:
            lightning_value = float(cell_hist["lightning_count"].dropna().iloc[-1])
        else:
            lightning_value = 0.0

        if "dry_lightning_count" in cell_hist.columns and cell_hist["dry_lightning_count"].dropna().shape[0]:
            dry_lightning_value = float(cell_hist["dry_lightning_count"].dropna().iloc[-1])
        else:
            dry_lightning_value = 0.0

        if "lightning_probability" in cell_hist.columns and cell_hist["lightning_probability"].dropna().shape[0]:
            lightning_prob = float(cell_hist["lightning_probability"].dropna().iloc[-1])
        else:
            lightning_prob = 1.0 if lightning_value > 0 else 0.0

        lightning_tail = cell_hist["lightning_count"].fillna(0).tail(3).tolist() if "lightning_count" in cell_hist.columns else []
        dry_lightning_tail = cell_hist["dry_lightning_count"].fillna(0).tail(3).tolist() if "dry_lightning_count" in cell_hist.columns else []

        rows.append(
            {
                "date": row.date,
                "cell_id": row.cell_id,
                "lat_center": row.lat_center,
                "lon_center": row.lon_center,
                "t2m_max_c": row.t2m_max_c,
                "rh_min_pct": row.rh_min_pct,
                "wind10m_max_ms": row.wind10m_max_ms,
                "precip_mm": row.precip_mm,
                "precip_3d_mm": float(sum(precip_tail[-3:])),
                "precip_7d_mm": float(sum(precip_tail[-7:])),
                "precip_30d_mm": float(sum(precip_tail[-30:])),
                "dry_days_30d": int(sum(1 for x in precip_tail[-30:] if x < 1.0)),
                "vpd_kpa": row.vpd_kpa,
                "ffmc": row.ffmc,
                "dmc": row.dmc,
                "dc": row.dc,
                "isi": row.isi,
                "bui": row.bui,
                "fwi": row.fwi,
                "fwi_proxy": row.fwi_proxy,
                "ndvi": ndvi_value,
                "ndvi_anom_30d": ndvi_anom_30d,
                "veg_dryness_index": veg_dryness_index,
                "soil_moisture_surface_m3m3": sm_value,
                "soil_moisture_pct_of_normal": sm_pct,
                "lightning_count": lightning_value,
                "dry_lightning_count": dry_lightning_value,
                "lightning_probability": lightning_prob,
                "lightning_count_prev_3d": float(sum(lightning_tail[-3:])) if lightning_tail else 0.0,
                "dry_lightning_count_prev_3d": float(sum(dry_lightning_tail[-3:])) if dry_lightning_tail else 0.0,
                "fire_count_prev_7d": int(sum(fire_tail[-7:])) if fire_tail else 0,
                "frp_sum_prev_7d": float(sum(frp_tail[-7:])) if frp_tail else 0.0,
                "month_sin": row.month_sin,
                "month_cos": row.month_cos,
                "doy_sin": row.doy_sin,
                "doy_cos": row.doy_cos,
            }
        )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        raise ValueError(f"No forecast rows found for prediction date {prediction_date}")

    # Model output is probability of next-day fire event.
    pred_df["pred_fire_next_1d"] = model.predict_proba(pred_df[REAL_FEATURE_COLUMNS])[:, 1]

    thresholds = settings.training.prediction_thresholds
    def classify(x: float) -> str:
        if x < thresholds["low"]:
            return "low"
        if x < thresholds["moderate"]:
            return "moderate"
        if x < thresholds["high"]:
            return "high"
        return "extreme"

    pred_df["risk_score"] = pred_df["pred_fire_next_1d"]
    pred_df["risk_class"] = pred_df["risk_score"].apply(classify)

    # Persist date-specific prediction artifact for downstream consumption.
    output_path = Path(out_dir) / f"predictions_real_{pred_date.date()}.csv"
    write_csv(pred_df, output_path)
    return str(output_path)
