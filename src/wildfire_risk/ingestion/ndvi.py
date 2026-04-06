from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "cell_id", "ndvi"}


@dataclass
class NDVIClient:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"NDVI CSV is missing required columns: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["cell_id"] = out["cell_id"].astype(str)
        out["ndvi"] = pd.to_numeric(out["ndvi"], errors="coerce").clip(-1, 1)
        out = out.dropna(subset=["date", "cell_id", "ndvi"])

        optional = [c for c in ["lat_center", "lon_center"] if c in out.columns]
        keep = ["date", "cell_id", "ndvi"] + optional
        return out[keep].drop_duplicates().reset_index(drop=True)

    def save_standardized_csv(self, input_path: str | Path, output_path: str | Path) -> str:
        df = self.load_csv(input_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)
