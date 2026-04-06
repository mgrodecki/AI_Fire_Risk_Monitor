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
