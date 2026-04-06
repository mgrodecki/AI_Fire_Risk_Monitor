from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {"date", "cell_id", "soil_moisture_surface_m3m3"}


@dataclass
class SMAPClient:
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"SMAP CSV is missing required columns: {sorted(missing)}")

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out["cell_id"] = out["cell_id"].astype(str)
        out["soil_moisture_surface_m3m3"] = pd.to_numeric(
            out["soil_moisture_surface_m3m3"], errors="coerce"
        ).clip(0, 1)

        if "soil_moisture_pct_of_normal" in out.columns:
            out["soil_moisture_pct_of_normal"] = pd.to_numeric(
                out["soil_moisture_pct_of_normal"], errors="coerce"
            ).clip(0, 300)

        out = out.dropna(subset=["date", "cell_id", "soil_moisture_surface_m3m3"])
        keep = ["date", "cell_id", "soil_moisture_surface_m3m3"]
        for col in ["lat_center", "lon_center", "soil_moisture_pct_of_normal"]:
            if col in out.columns:
                keep.append(col)
        return out[keep].drop_duplicates().reset_index(drop=True)

    def save_standardized_csv(self, input_path: str | Path, output_path: str | Path) -> str:
        df = self.load_csv(input_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(output_path)
