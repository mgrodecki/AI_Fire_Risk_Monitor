from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.ingestion.lightning import LightningClient


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source lightning CSV")
    args = parser.parse_args()

    client = LightningClient()
    output_path = settings.real_data["lightning_observations_file"]
    saved = client.save_standardized_csv(args.input, output_path)
    print(saved)


if __name__ == "__main__":
    main()
