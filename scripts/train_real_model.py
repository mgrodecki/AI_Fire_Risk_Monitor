from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.pipelines.real_data_pipeline import train_real_model


if __name__ == "__main__":
    print(train_real_model())
