from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.utils.io import ensure_dir, write_csv
from wildfire_risk.utils.logging_utils import get_logger
from wildfire_risk.data.grid import build_grid
from wildfire_risk.data.demo_data import (
    make_static_features,
    make_daily_base,
    make_dynamic_features,
    make_synthetic_fire_activity,
)
from wildfire_risk.data.assemble import assemble_training_table

logger = get_logger(__name__)


def main():
    settings = load_settings()
    raw_dir = ensure_dir(settings.paths.raw_dir)
    curated_dir = ensure_dir(settings.paths.curated_dir)

    grid_df = build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )
    static_df = make_static_features(grid_df, random_state=settings.training.random_state)
    base_df = make_daily_base(grid_df, start_date="2025-01-01", end_date="2026-03-21")
    dynamic_df = make_dynamic_features(base_df, static_df, random_state=settings.training.random_state)
    dynamic_df = make_synthetic_fire_activity(dynamic_df, random_state=settings.training.random_state)
    training_table = assemble_training_table(dynamic_df)

    write_csv(grid_df, Path(raw_dir) / "grid_cells.csv")
    write_csv(static_df, Path(raw_dir) / "cell_static_features.csv")
    write_csv(training_table, Path(curated_dir) / "training_table.csv")

    logger.info("Wrote grid, static features, and training table")
    logger.info("Training table rows: %d", len(training_table))
    logger.info("Positive rate: %.4f", training_table["label_fire_next_1d"].mean())


if __name__ == "__main__":
    main()
