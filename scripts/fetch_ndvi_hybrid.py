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
