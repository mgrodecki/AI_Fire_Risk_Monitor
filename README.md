# Wildfire Risk - Real NASA FIRMS + Open-Meteo + Hybrid NDVI + SMAP + Lightning

This version has a real-data path wired to:

- NASA FIRMS for active fire detections
- Open-Meteo for historical and forecast weather
- Hybrid NDVI connectors (MODIS LP DAAC baseline + Sentinel CDSE refinement)
- SMAP via EarthAccess
- Lightning via GOES GLM

It provides a runnable pipeline that can:

1. fetch historical daily weather for each grid cell from Open-Meteo
2. fetch FIRMS fire detections for a bounding box
3. fetch NDVI via hybrid connectors
4. fetch SMAP observations via EarthAccess
5. fetch lightning observations via GOES
6. build a real training table
7. train a next-day wildfire probability model
8. generate daily risk predictions

## What Is Included

New modules:

- `src/wildfire_risk/ingestion/openmeteo.py`
- `src/wildfire_risk/ingestion/firms.py`
- `src/wildfire_risk/ingestion/ndvi.py`
- `src/wildfire_risk/ingestion/smap.py`
- `src/wildfire_risk/ingestion/lightning.py`
- `src/wildfire_risk/ingestion/lpdaac_modis_ndvi.py`
- `src/wildfire_risk/ingestion/copernicus_sentinel_ndvi.py`
- `src/wildfire_risk/ingestion/hybrid_ndvi.py`
- `src/wildfire_risk/ingestion/earthaccess_smap.py`
- `src/wildfire_risk/ingestion/goes_glm_lightning.py`
- `src/wildfire_risk/data/schema_real.py`
- `src/wildfire_risk/pipelines/real_data_pipeline.py`

New runnable scripts:

- `scripts/fetch_real_data.py`
- `scripts/fetch_ndvi_modis_lpdaac.py`
- `scripts/fetch_ndvi_hybrid.py`
- `scripts/fetch_smap_earthaccess.py`
- `scripts/fetch_lightning_goes.py`
- `scripts/build_real_training_table.py`
- `scripts/train_real_model.py`
- `scripts/predict_real_daily.py`

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Set your FIRMS key:

```bash
# Linux/macOS
export FIRMS_API_KEY=your_key_here

# Windows PowerShell
$env:FIRMS_API_KEY="your_key_here"
```

For connector scripts, configure `api_connectors` in `configs/settings.yaml`:

- `lpdaac_product`, `lpdaac_layer` for MODIS/AppEEARS NDVI
- `sentinel_collection`, `sentinel_processing_level`, `sentinel_cloud_cover_max`, `sentinel_refine_radius_km` for Sentinel refinement
- `smap_short_name` for EarthAccess SMAP search
- `goes_satellite`, `goes_product` for GOES GLM lightning

Credential environment variables for connector runs:

- `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`
- `CDSE_USERNAME`, `CDSE_PASSWORD`

## Example Run

```bash
python scripts/fetch_real_data.py --start-date 2025-06-01 --end-date 2025-09-30 --forecast-days 7 --firms-source VIIRS_NOAA20_SP
python scripts/fetch_ndvi_hybrid.py --start-date 2025-06-01 --end-date 2025-09-30
python scripts/fetch_smap_earthaccess.py --start-date 2025-06-01 --end-date 2025-09-30
python scripts/fetch_lightning_goes.py --start-date 2025-06-01 --end-date 2025-09-30
python scripts/build_real_training_table.py
python scripts/train_real_model.py
python scripts/predict_real_daily.py --date 2026-03-22
```

Outputs:

- `data/raw/openmeteo_history.csv`
- `data/raw/openmeteo_forecast.csv`
- `data/raw/firms_events.csv`
- `data/raw/ndvi_observations.csv`
- `data/raw/smap_observations.csv`
- `data/raw/lightning_observations.csv`
- `data/curated/training_table_real.csv`
- `artifacts/models/wildfire_risk_model_real.joblib`
- `data/curated/daily_predictions/predictions_real_YYYY-MM-DD.csv`

## Important Constraints

- FIRMS Area API requires a `MAP_KEY` / `FIRMS_API_KEY`.
- FIRMS Area API allows day ranges from 1 to 5; requests are chunked automatically.
- Open-Meteo history and forecast are queried per grid-cell center.
- AppEEARS, Copernicus Data Space, EarthAccess, and GOES connectors each require credentials/network access.
