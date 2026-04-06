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
