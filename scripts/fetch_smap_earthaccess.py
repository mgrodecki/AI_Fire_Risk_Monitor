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
