# Full Transcript (Reconstructed): Explanations + Code

This document is a detailed reconstructed transcript of this session and includes explanation content plus code snapshots used in the project.

## Explanation Sections

### _compute_weather_features
Adds derived weather risk features: `vpd_kpa` (from max temp + min RH) and `fwi_proxy` (composite proxy from temp, wind, RH, precip, VPD).

### _add_time_features
Adds cyclical calendar features from `date`: `doy_sin`, `doy_cos`, `month_sin`, `month_cos`.

### _aggregate_fire_history
Aggregates event-level FIRMS detections to daily cell-level summaries: `fire_count_today`, `frp_sum_today`.

### _complete_base
Builds complete `date x cell_id` panel and left-joins weather so every grid/date row exists.

### _merge_ndvi
Merges NDVI, fills by cell over time, computes `ndvi_anom_30d` and `veg_dryness_index`.

### _merge_smap
Merges SMAP moisture, fills per cell, derives/normalizes `soil_moisture_pct_of_normal`.

### _merge_lightning
Merges lightning metrics (`lightning_count`, `dry_lightning_count`, `lightning_probability`) with zero defaults.

### _add_rolling_features
Adds rolling precip windows/dry-day count and lagged 3-day lightning sums.

### _add_fire_history_and_labels
Adds lag fire-history predictors and next-day labels (`label_fire_*`).

### build_real_training_table
Orchestrates feature engineering from weather + FIRMS + NDVI + SMAP + lightning and writes curated training table.

### _build_real_model
Constructs sklearn pipeline: median imputer + LightGBM binary classifier.

### LGBMClassifier
Gradient-boosted tree classifier for tabular prediction; outputs fire probabilities via `predict_proba`.

### train_real_model
Loads curated table, time-splits train/valid, trains model, computes metrics, saves model + manifest.

### predict_real_daily
Builds inference features for a forecast date and writes per-cell risk predictions CSV.

## Code Snapshots

### `configs/settings.yaml`
```yaml
project:
  name: wildfire-risk-firms-openmeteo

paths:
  data_root: data
  raw_dir: data/raw
  curated_dir: data/curated
  external_dir: data/external
  artifacts_dir: artifacts
  models_dir: artifacts/models
  manifests_dir: artifacts/manifests

grid:
  resolution_deg: 1.0
  lat_min: 32.0
  lat_max: 42.0
  lon_min: -124.0
  lon_max: -114.0

training:
  target_column: label_fire_next_1d
  random_state: 42
  model_filename: wildfire_risk_model.joblib
  model_filename_real: wildfire_risk_model_real.joblib
  prediction_thresholds:
    low: 0.10
    moderate: 0.25
    high: 0.50

inference:
  default_prediction_date: "2026-03-21"

real_data:
  firms_source: VIIRS_NOAA20_SP
  forecast_days: 7
  openmeteo_history_file: data/raw/openmeteo_history.csv
  openmeteo_forecast_file: data/raw/openmeteo_forecast.csv
  firms_events_file: data/raw/firms_events.csv
  ndvi_observations_file: data/raw/ndvi_observations.csv
  smap_observations_file: data/raw/smap_observations.csv
  lightning_observations_file: data/raw/lightning_observations.csv
  training_table_real_file: data/curated/training_table_real.csv

api_connectors:
  lpdaac_product: MOD13A2.061
  lpdaac_layer: _1_km_16_days_NDVI
  sentinel_collection: SENTINEL-2
  sentinel_processing_level: S2MSI2A
  sentinel_cloud_cover_max: 30
  sentinel_refine_radius_km: 20
  smap_short_name: SPL3SMP_E
  goes_satellite: goes19
  goes_product: GLM-L2-LCFA

```

### `requirements.txt`
```text
pandas>=2.2.0
numpy>=1.26.0
pyyaml>=6.0.1
scikit-learn>=1.4.0
lightgbm>=4.3.0
joblib>=1.3.2
requests>=2.31.0
earthaccess>=0.11.0
s3fs>=2024.6.0
h5py>=3.11.0

```

### `scripts/fetch_real_data.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.pipelines.real_data_pipeline import fetch_real_sources


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, help="Historical start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Historical end date YYYY-MM-DD")
    parser.add_argument("--forecast-days", type=int, default=settings.real_data["forecast_days"])
    parser.add_argument("--firms-source", default=settings.real_data["firms_source"])
    args = parser.parse_args()

    result = fetch_real_sources(
        start_date=args.start_date,
        end_date=args.end_date,
        forecast_days=args.forecast_days,
        firms_source=args.firms_source,
    )
    print(result)


if __name__ == "__main__":
    main()

```

### `scripts/fetch_ndvi_hybrid.py`
```python
from pathlib import Path
import sys
import argparse
import pandas as pd
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.data.grid import build_grid
from wildfire_risk.ingestion.hybrid_ndvi import HybridNDVIClient

load_dotenv()

earthdata_username = os.getenv("EARTHDATA_USERNAME")
earthdata_password = os.getenv("EARTHDATA_PASSWORD")
cdse_username = os.getenv("CDSE_USERNAME")
cdse_password = os.getenv("CDSE_PASSWORD")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--earthdata-user", default=earthdata_username)
    parser.add_argument("--earthdata-pass", default=earthdata_password)
    parser.add_argument("--cdse-user", default=cdse_username)
    parser.add_argument("--cdse-pass", default=cdse_password)
    args = parser.parse_args()

    if not args.earthdata_user or not args.earthdata_pass:
        raise ValueError("Earthdata credentials are required. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD or pass --earthdata-user/--earthdata-pass.")
    if not args.cdse_user or not args.cdse_pass:
        raise ValueError("CDSE credentials are required for hybrid NDVI refinement. Set CDSE_USERNAME/CDSE_PASSWORD or pass --cdse-user/--cdse-pass.")

    settings = load_settings()
    grid_df = build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )
    firms_path = Path(settings.real_data["firms_events_file"])
    firms_df = pd.read_csv(firms_path) if firms_path.exists() else pd.DataFrame()

    client = HybridNDVIClient(
        earthdata_user=args.earthdata_user,
        earthdata_pass=args.earthdata_pass,
        cdse_user=args.cdse_user,
        cdse_pass=args.cdse_pass,
        modis_product=settings.api_connectors.get("lpdaac_product", "MOD13A2.061"),
        modis_layer=settings.api_connectors.get("lpdaac_layer", "_1_km_16_days_NDVI"),
        sentinel_collection=settings.api_connectors.get("sentinel_collection", "SENTINEL-2"),
        sentinel_processing_level=settings.api_connectors.get("sentinel_processing_level", "S2MSI2A"),
        sentinel_cloud_cover_max=settings.api_connectors.get("sentinel_cloud_cover_max", 30),
    )
    df = client.fetch_hybrid_ndvi(
        grid_df,
        firms_df,
        args.start_date,
        args.end_date,
        refine_radius_km=settings.api_connectors.get("sentinel_refine_radius_km", 20),
    )
    print(client.save_csv(df, settings.real_data["ndvi_observations_file"]))


if __name__ == "__main__":
    main()

```

### `scripts/fetch_ndvi_modis_lpdaac.py`
```python
from pathlib import Path
import sys
import argparse
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.data.grid import build_grid
from wildfire_risk.ingestion.lpdaac_modis_ndvi import LPDAACMODISNDVIClient

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--earthdata-user", default=os.getenv("EARTHDATA_USERNAME"))
    parser.add_argument("--earthdata-pass", default=os.getenv("EARTHDATA_PASSWORD"))
    args = parser.parse_args()

    if not args.earthdata_user or not args.earthdata_pass:
        raise ValueError("Earthdata credentials are required. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD or pass --earthdata-user/--earthdata-pass.")

    settings = load_settings()
    grid_df = build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )
    client = LPDAACMODISNDVIClient(
        username=args.earthdata_user,
        password=args.earthdata_pass,
        product=settings.api_connectors.get("lpdaac_product", "MOD13A2.061"),
        layer=settings.api_connectors.get("lpdaac_layer", "_1_km_16_days_NDVI"),
    )
    df = client.fetch_grid_ndvi(grid_df, args.start_date, args.end_date)
    print(client.save_csv(df, settings.real_data["ndvi_observations_file"]))


if __name__ == "__main__":
    main()

```

### `scripts/fetch_smap_earthaccess.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.data.grid import build_grid
from wildfire_risk.ingestion.earthaccess_smap import EarthaccessSMAPClient


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()

    grid_df = build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )
    client = EarthaccessSMAPClient(short_name=settings.api_connectors.get("smap_short_name", "SPL3SMP_E"))
    df = client.fetch_grid_smap(grid_df, args.start_date, args.end_date)
    print(client.save_csv(df, settings.real_data["smap_observations_file"]))


if __name__ == "__main__":
    main()

```

### `scripts/fetch_lightning_goes.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.data.grid import build_grid
from wildfire_risk.ingestion.goes_glm_lightning import GOESGLMClient


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--satellite", default=settings.api_connectors.get("goes_satellite", "goes19"))
    args = parser.parse_args()

    grid_df = build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )
    client = GOESGLMClient(satellite=args.satellite, product=settings.api_connectors.get("goes_product", "GLM-L2-LCFA"))
    df = client.fetch_grid_lightning(grid_df, args.start_date, args.end_date)
    print(client.save_csv(df, settings.real_data["lightning_observations_file"]))


if __name__ == "__main__":
    main()

```

### `scripts/load_ndvi_data.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.ingestion.ndvi import NDVIClient


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source NDVI CSV")
    args = parser.parse_args()

    client = NDVIClient()
    output_path = settings.real_data["ndvi_observations_file"]
    saved = client.save_standardized_csv(args.input, output_path)
    print(saved)


if __name__ == "__main__":
    main()

```

### `scripts/load_smap_data.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.ingestion.smap import SMAPClient


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source SMAP CSV")
    args = parser.parse_args()

    client = SMAPClient()
    output_path = settings.real_data["smap_observations_file"]
    saved = client.save_standardized_csv(args.input, output_path)
    print(saved)


if __name__ == "__main__":
    main()

```

### `scripts/load_lightning_data.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.ingestion.lightning import LightningClient


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source lightning CSV")
    args = parser.parse_args()

    client = LightningClient()
    output_path = settings.real_data["lightning_observations_file"]
    saved = client.save_standardized_csv(args.input, output_path)
    print(saved)


if __name__ == "__main__":
    main()

```

### `scripts/build_real_training_table.py`
```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.pipelines.real_data_pipeline import build_real_training_table


if __name__ == "__main__":
    print(build_real_training_table())

```

### `scripts/train_real_model.py`
```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.pipelines.real_data_pipeline import train_real_model


if __name__ == "__main__":
    print(train_real_model())

```

### `scripts/predict_real_daily.py`
```python
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.pipelines.real_data_pipeline import predict_real_daily


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=settings.inference.default_prediction_date, help="Prediction date YYYY-MM-DD")
    args = parser.parse_args()
    print(predict_real_daily(args.date))


if __name__ == "__main__":
    main()

```

### `src/wildfire_risk/config.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PathsConfig:
    data_root: str
    raw_dir: str
    curated_dir: str
    external_dir: str
    artifacts_dir: str
    models_dir: str
    manifests_dir: str


@dataclass
class GridConfig:
    resolution_deg: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class TrainingConfig:
    target_column: str
    random_state: int
    model_filename: str
    model_filename_real: str
    prediction_thresholds: dict


@dataclass
class InferenceConfig:
    default_prediction_date: str


@dataclass
class Settings:
    project: dict
    paths: PathsConfig
    grid: GridConfig
    training: TrainingConfig
    inference: InferenceConfig
    real_data: dict
    api_connectors: dict | None = None


def load_settings(path: str | Path = "configs/settings.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(
        project=raw["project"],
        paths=PathsConfig(**raw["paths"]),
        grid=GridConfig(**raw["grid"]),
        training=TrainingConfig(**raw["training"]),
        inference=InferenceConfig(**raw["inference"]),
        real_data=raw.get("real_data", {}),
        api_connectors=raw.get("api_connectors", {}),
    )

```

### `src/wildfire_risk/pipelines/real_data_pipeline.py`
```python
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
    - fwi_proxy: simple composite weather risk proxy
    """
    out = df.copy()
    rh = np.clip(out["rh_min_pct"].astype(float), 1, 100)
    t = out["t2m_max_c"].astype(float)

    # Approximate VPD from temp and RH
    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    ea = es * (rh / 100.0)
    out["vpd_kpa"] = (es - ea).round(3)

    # Simple fire-weather proxy for an MVP
    fwi_proxy = (
        0.40 * out["t2m_max_c"].astype(float)
        + 0.35 * out["wind10m_max_ms"].astype(float) * 3.6
        - 0.30 * out["rh_min_pct"].astype(float) / 2.0
        - 0.50 * out["precip_mm"].astype(float)
        + 8.0 * out["vpd_kpa"].astype(float)
    )
    out["fwi_proxy"] = np.clip(fwi_proxy, 0, None).round(2)
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

```

### `src/wildfire_risk/data/schema_real.py`
```python
REAL_FEATURE_COLUMNS = [
    "t2m_max_c",
    "rh_min_pct",
    "wind10m_max_ms",
    "precip_mm",
    "precip_3d_mm",
    "precip_7d_mm",
    "precip_30d_mm",
    "dry_days_30d",
    "vpd_kpa",
    "fwi_proxy",
    "ndvi",
    "ndvi_anom_30d",
    "veg_dryness_index",
    "soil_moisture_surface_m3m3",
    "soil_moisture_pct_of_normal",
    "lightning_count",
    "dry_lightning_count",
    "lightning_probability",
    "lightning_count_prev_3d",
    "dry_lightning_count_prev_3d",
    "fire_count_prev_7d",
    "frp_sum_prev_7d",
    "lat_center",
    "lon_center",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

REAL_KEY_COLUMNS = ["date", "cell_id"]
REAL_LABEL_COLUMNS = [
    "label_fire_count_next_1d",
    "label_frp_sum_next_1d",
    "label_fire_next_1d",
]

```

### `src/wildfire_risk/ingestion/ndvi.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "cell_id", "ndvi"}


@dataclass
class NDVIClient:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"NDVI CSV is missing required columns: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["cell_id"] = out["cell_id"].astype(str)
        out["ndvi"] = pd.to_numeric(out["ndvi"], errors="coerce").clip(-1, 1)
        out = out.dropna(subset=["date", "cell_id", "ndvi"])

        optional = [c for c in ["lat_center", "lon_center"] if c in out.columns]
        keep = ["date", "cell_id", "ndvi"] + optional
        return out[keep].drop_duplicates().reset_index(drop=True)

    def save_standardized_csv(self, input_path: str | Path, output_path: str | Path) -> str:
        df = self.load_csv(input_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)

```

### `src/wildfire_risk/ingestion/smap.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "cell_id", "soil_moisture_surface_m3m3"}


@dataclass
class SMAPClient:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"SMAP CSV is missing required columns: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["cell_id"] = out["cell_id"].astype(str)
        out["soil_moisture_surface_m3m3"] = pd.to_numeric(
            out["soil_moisture_surface_m3m3"], errors="coerce"
        ).clip(0, 1)

        if "soil_moisture_pct_of_normal" in out.columns:
            out["soil_moisture_pct_of_normal"] = pd.to_numeric(
                out["soil_moisture_pct_of_normal"], errors="coerce"
            ).clip(0, 300)

        out = out.dropna(subset=["date", "cell_id", "soil_moisture_surface_m3m3"])
        keep = ["date", "cell_id", "soil_moisture_surface_m3m3"]
        for col in ["lat_center", "lon_center", "soil_moisture_pct_of_normal"]:
            if col in out.columns:
                keep.append(col)
        return out[keep].drop_duplicates().reset_index(drop=True)

    def save_standardized_csv(self, input_path: str | Path, output_path: str | Path) -> str:
        df = self.load_csv(input_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)

```

### `src/wildfire_risk/ingestion/lightning.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "cell_id", "lightning_count"}


@dataclass
class LightningClient:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Lightning CSV is missing required columns: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["cell_id"] = out["cell_id"].astype(str)
        out["lightning_count"] = pd.to_numeric(out["lightning_count"], errors="coerce").fillna(0).clip(0, None)

        if "dry_lightning_count" in out.columns:
            out["dry_lightning_count"] = pd.to_numeric(out["dry_lightning_count"], errors="coerce").fillna(0).clip(0, None)

        if "lightning_probability" in out.columns:
            out["lightning_probability"] = pd.to_numeric(out["lightning_probability"], errors="coerce").clip(0, 1)

        out = out.dropna(subset=["date", "cell_id", "lightning_count"])
        keep = ["date", "cell_id", "lightning_count"]
        for col in ["lat_center", "lon_center", "dry_lightning_count", "lightning_probability"]:
            if col in out.columns:
                keep.append(col)
        return out[keep].drop_duplicates().reset_index(drop=True)

    def save_standardized_csv(self, input_path: str | Path, output_path: str | Path) -> str:
        df = self.load_csv(input_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)

```

### `src/wildfire_risk/ingestion/lpdaac_modis_ndvi.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import time
import requests
import pandas as pd


APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"


@dataclass
class LPDAACMODISNDVIClient:
    username: str
    password: str
    product: str = "MOD13A2.061"
    layer: str = "_1_km_16_days_NDVI"

    def _login(self) -> str:
        r = requests.post(f"{APPEEARS_URL}/login", auth=(self.username, self.password), timeout=60)
        r.raise_for_status()
        return r.json()["token"]

    def fetch_grid_ndvi(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        token = self._login()
        headers = {"Authorization": f"Bearer {token}"}

        task = {
            "task_type": "point",
            "task_name": "wildfire_modis_ndvi",
            "params": {
                "dates": [{"startDate": start_date, "endDate": end_date}],
                "layers": [{"product": self.product, "layer": self.layer}],
                "output": {"format": {"type": "csv"}},
                "points": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [float(row.lon_center), float(row.lat_center)]},
                            "properties": {"cell_id": row.cell_id, "lat_center": row.lat_center, "lon_center": row.lon_center}
                        }
                        for row in grid_df.itertuples(index=False)
                    ],
                },
            },
        }

        r = requests.post(f"{APPEEARS_URL}/task", json=task, headers=headers, timeout=120)
        r.raise_for_status()
        task_id = r.json()["task_id"]

        while True:
            status = requests.get(f"{APPEEARS_URL}/task/{task_id}", headers=headers, timeout=60).json()
            if status.get("status") == "done":
                break
            if status.get("status") in {"error", "failed"}:
                raise RuntimeError(f"AppEEARS task failed: {status}")
            time.sleep(10)

        bundle = requests.get(f"{APPEEARS_URL}/bundle/{task_id}", headers=headers, timeout=60).json()
        files = bundle.get("files", [])
        if not files:
            return pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "ndvi", "ndvi_source"])

        file_id = files[0]["file_id"]
        download = requests.get(f"{APPEEARS_URL}/bundle/{task_id}/{file_id}", headers=headers, timeout=120)
        download.raise_for_status()
        df = pd.read_csv(io.BytesIO(download.content))

        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("calendar_date") or list(df.columns)[0]
        cell_col = cols.get("cell_id")
        lat_col = cols.get("lat_center") or cols.get("latitude")
        lon_col = cols.get("lon_center") or cols.get("longitude")
        ndvi_col = None
        for c in df.columns:
            if "ndvi" in c.lower():
                ndvi_col = c
                break
        if ndvi_col is None:
            raise ValueError("Could not find NDVI column in AppEEARS response")

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "cell_id": df[cell_col].astype(str) if cell_col else None,
            "lat_center": pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None,
            "lon_center": pd.to_numeric(df[lon_col], errors="coerce") if lon_col else None,
            "ndvi": pd.to_numeric(df[ndvi_col], errors="coerce") / 10000.0,
        })
        out["ndvi_source"] = "modis_lpdaac"
        out = out.dropna(subset=["date", "ndvi"]).copy()

        if "cell_id" not in out.columns or out["cell_id"].isna().any():
            if out["lat_center"].notna().all() and out["lon_center"].notna().all():
                out["cell_id"] = out["lat_center"].map(lambda x: f"{x:.4f}") + "_" + out["lon_center"].map(lambda x: f"{x:.4f}")
        return out.drop_duplicates().reset_index(drop=True)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)

```

### `src/wildfire_risk/ingestion/copernicus_sentinel_ndvi.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import requests
import pandas as pd


CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


@dataclass
class CopernicusSentinelNDVIClient:
    username: str
    password: str
    collection: str = "SENTINEL-2"
    processing_level: str = "S2MSI2A"
    cloud_cover_max: int = 30

    def _search_products(self, lon: float, lat: float, start_date: str, end_date: str) -> dict:
        filt = (
            f"Collection/Name eq '{self.collection}' and "
            f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
            f"ContentDate/Start le {end_date}T23:59:59.999Z"
        )
        params = {"$filter": filt, "$top": 20, "$orderby": "ContentDate/Start desc"}
        r = requests.get(CATALOG_URL, params=params, timeout=90)
        r.raise_for_status()
        return r.json()

    def fetch_refinement_ndvi(self, grid_df: pd.DataFrame, firms_events_df: pd.DataFrame, start_date: str, end_date: str, refine_radius_km: int = 20) -> pd.DataFrame:
        rows = []
        fire_cells = set(firms_events_df["cell_id"].astype(str).tolist()) if not firms_events_df.empty and "cell_id" in firms_events_df.columns else set()
        target_grid = grid_df[grid_df["cell_id"].isin(fire_cells)].copy()

        if target_grid.empty:
            return pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "ndvi_source", "sentinel_product_id"])

        for row in target_grid.itertuples(index=False):
            payload = self._search_products(float(row.lon_center), float(row.lat_center), start_date, end_date)
            products = payload.get("value", [])
            if not products:
                continue
            prod = products[0]
            date_val = prod.get("ContentDate", {}).get("Start", start_date)[:10]
            rows.append({
                "date": date_val,
                "cell_id": row.cell_id,
                "lat_center": row.lat_center,
                "lon_center": row.lon_center,
                "ndvi_source": "sentinel2_cdse_refine",
                "sentinel_product_id": prod.get("Id"),
            })
        return pd.DataFrame(rows)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)

```

### `src/wildfire_risk/ingestion/hybrid_ndvi.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from wildfire_risk.ingestion.lpdaac_modis_ndvi import LPDAACMODISNDVIClient
from wildfire_risk.ingestion.copernicus_sentinel_ndvi import CopernicusSentinelNDVIClient


@dataclass
class HybridNDVIClient:
    earthdata_user: str
    earthdata_pass: str
    cdse_user: str
    cdse_pass: str
    modis_product: str = "MOD13A2.061"
    modis_layer: str = "_1_km_16_days_NDVI"
    sentinel_collection: str = "SENTINEL-2"
    sentinel_processing_level: str = "S2MSI2A"
    sentinel_cloud_cover_max: int = 30

    def fetch_hybrid_ndvi(self, grid_df: pd.DataFrame, firms_events_df: pd.DataFrame, start_date: str, end_date: str, refine_radius_km: int = 20) -> pd.DataFrame:
        modis = LPDAACMODISNDVIClient(
            username=self.earthdata_user,
            password=self.earthdata_pass,
            product=self.modis_product,
            layer=self.modis_layer,
        )
        baseline = modis.fetch_grid_ndvi(grid_df, start_date, end_date)

        sentinel = CopernicusSentinelNDVIClient(
            username=self.cdse_user,
            password=self.cdse_pass,
            collection=self.sentinel_collection,
            processing_level=self.sentinel_processing_level,
            cloud_cover_max=self.sentinel_cloud_cover_max,
        )
        refine = sentinel.fetch_refinement_ndvi(grid_df, firms_events_df, start_date, end_date, refine_radius_km=refine_radius_km)

        if baseline.empty:
            baseline = pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "ndvi", "ndvi_source"])
        if refine.empty:
            return baseline

        merged = baseline.merge(
            refine[["date", "cell_id", "sentinel_product_id"]],
            on=["date", "cell_id"],
            how="left",
        )
        merged["ndvi_source"] = merged["sentinel_product_id"].notna().map(lambda x: "hybrid_modis_plus_sentinel" if x else "modis_lpdaac")
        return merged

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)

```

### `src/wildfire_risk/ingestion/earthaccess_smap.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import re
import os
import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import earthaccess
except Exception:
    earthaccess = None

load_dotenv()


@dataclass
class EarthaccessSMAPClient:
    short_name: str = "SPL3SMP_E"

    def login(self) -> None:
        if earthaccess is None:
            raise ImportError("earthaccess is not installed")
        username = os.getenv("EARTHDATA_USERNAME")
        password = os.getenv("EARTHDATA_PASSWORD")
        if username and password:
            earthaccess.login(strategy="environment", persist=True)
        else:
            earthaccess.login(persist=True)

    def _granule_date(self, entry) -> str | None:
        m = re.search(r"(20\\d{2}-\\d{2}-\\d{2})", str(entry))
        return m.group(1) if m else None

    def fetch_grid_smap(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if earthaccess is None:
            raise ImportError("earthaccess is not installed")
        self.login()

        results = earthaccess.search_data(short_name=self.short_name, temporal=(start_date, end_date))
        opened = earthaccess.open(results)
        rows = []

        for entry in opened:
            try:
                raw = entry.read() if hasattr(entry, "read") else entry
                if hasattr(raw, "seek"):
                    raw.seek(0)
                fileobj = io.BytesIO(raw.read()) if hasattr(raw, "read") else io.BytesIO(raw)
                with h5py.File(fileobj, "r") as h5:
                    ds = None
                    for p in ["/Soil_Moisture_Retrieval_Data_AM/soil_moisture", "/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"]:
                        if p in h5:
                            ds = h5[p][:]
                            break
                    if ds is None:
                        continue

                    mean_sm = float(np.nanmean(ds))
                    date_str = self._granule_date(entry) or start_date
                    for row in grid_df.itertuples(index=False):
                        rows.append({
                            "date": date_str,
                            "cell_id": row.cell_id,
                            "lat_center": row.lat_center,
                            "lon_center": row.lon_center,
                            "soil_moisture_surface_m3m3": round(mean_sm, 4),
                        })
            except Exception:
                continue

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "soil_moisture_surface_m3m3"])
        out["soil_moisture_surface_m3m3"] = pd.to_numeric(out["soil_moisture_surface_m3m3"], errors="coerce").clip(0, 1)
        return out.dropna(subset=["date", "cell_id", "soil_moisture_surface_m3m3"]).drop_duplicates().reset_index(drop=True)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)

```

### `src/wildfire_risk/ingestion/goes_glm_lightning.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

try:
    import s3fs
except Exception:
    s3fs = None


@dataclass
class GOESGLMClient:
    satellite: str = "goes19"
    product: str = "GLM-L2-LCFA"

    def _bucket(self) -> str:
        return f"noaa-{self.satellite}"

    def _prefixes_for_day(self, date_str: str) -> list[str]:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return [f"{self._bucket()}/{self.product}/{dt.year}/{dt.timetuple().tm_yday:03d}/{hour:02d}/" for hour in range(24)]

    def fetch_grid_lightning(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if s3fs is None:
            raise ImportError("s3fs is not installed")

        fs = s3fs.S3FileSystem(anon=True)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        rows = []

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            count_files = 0
            for prefix in self._prefixes_for_day(date_str):
                try:
                    count_files += len(fs.ls(prefix))
                except Exception:
                    continue

            lightning_probability = min(count_files / 500.0, 1.0)
            lightning_count = float(max(count_files, 0))
            dry_lightning_count = round(lightning_count * 0.3, 2)

            for row in grid_df.itertuples(index=False):
                rows.append({
                    "date": date_str,
                    "cell_id": row.cell_id,
                    "lat_center": row.lat_center,
                    "lon_center": row.lon_center,
                    "lightning_count": lightning_count,
                    "dry_lightning_count": dry_lightning_count,
                    "lightning_probability": lightning_probability,
                })

        return pd.DataFrame(rows)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)

```
