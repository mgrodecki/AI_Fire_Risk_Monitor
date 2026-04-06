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
