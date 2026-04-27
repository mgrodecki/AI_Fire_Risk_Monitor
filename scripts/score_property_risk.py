from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.pipelines.property_risk_pipeline import score_properties


def main():
    parser = argparse.ArgumentParser(description="Score property-level wildfire risk.")
    parser.add_argument("--input", required=True, help="Property CSV path (must include lat/lon).")
    parser.add_argument("--date", required=True, help="Prediction date YYYY-MM-DD (must match an existing daily prediction file).")
    parser.add_argument("--lookback-days", type=int, default=365, help="Days of FIRMS history for proximity calculation.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    out = score_properties(
        properties_path=args.input,
        prediction_date=args.date,
        lookback_days=args.lookback_days,
        output_path=args.output,
    )
    print(out)


if __name__ == "__main__":
    main()
