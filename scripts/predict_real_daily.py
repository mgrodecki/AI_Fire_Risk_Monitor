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
