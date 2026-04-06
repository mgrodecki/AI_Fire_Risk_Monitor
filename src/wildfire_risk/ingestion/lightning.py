from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "cell_id", "lightning_count"}


@dataclass
class LightningClient:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Lightning CSV is missing required columns: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["cell_id"] = out["cell_id"].astype(str)
        out["lightning_count"] = pd.to_numeric(out["lightning_count"], errors="coerce").fillna(0).clip(0, None)

        if "dry_lightning_count" in out.columns:
            out["dry_lightning_count"] = pd.to_numeric(out["dry_lightning_count"], errors="coerce").fillna(0).clip(0, None)

        if "lightning_probability" in out.columns:
            out["lightning_probability"] = pd.to_numeric(out["lightning_probability"], errors="coerce").clip(0, 1)

        out = out.dropna(subset=["date", "cell_id", "lightning_count"])
        keep = ["date", "cell_id", "lightning_count"]
        for col in ["lat_center", "lon_center", "dry_lightning_count", "lightning_probability"]:
            if col in out.columns:
                keep.append(col)
        return out[keep].drop_duplicates().reset_index(drop=True)

    def save_standardized_csv(self, input_path: str | Path, output_path: str | Path) -> str:
        df = self.load_csv(input_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)
