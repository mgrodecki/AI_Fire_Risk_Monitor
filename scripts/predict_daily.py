from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.pipelines.inference_pipeline import run_inference


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=settings.inference.default_prediction_date,
        help="Prediction date in YYYY-MM-DD format",
    )
    args = parser.parse_args()
    output = run_inference(args.date)
    print(output)


if __name__ == "__main__":
    main()
