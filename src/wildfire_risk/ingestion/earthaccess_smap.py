from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import re
import os
import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import earthaccess
except Exception:
    earthaccess = None

load_dotenv()


@dataclass
class EarthaccessSMAPClient:
    short_name: str = "SPL3SMP_E"

    def login(self) -> None:
        if earthaccess is None:
            raise ImportError("earthaccess is not installed")
        username = os.getenv("EARTHDATA_USERNAME")
        password = os.getenv("EARTHDATA_PASSWORD")
        if username and password:
            earthaccess.login(strategy="environment", persist=True)
        else:
            earthaccess.login(persist=True)

    def _granule_date(self, entry) -> str | None:
        m = re.search(r"(20\\d{2}-\\d{2}-\\d{2})", str(entry))
        return m.group(1) if m else None

    def fetch_grid_smap(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if earthaccess is None:
            raise ImportError("earthaccess is not installed")
        self.login()

        results = earthaccess.search_data(short_name=self.short_name, temporal=(start_date, end_date))
        opened = earthaccess.open(results)
        rows = []

        for entry in opened:
            try:
                raw = entry.read() if hasattr(entry, "read") else entry
                if hasattr(raw, "seek"):
                    raw.seek(0)
                fileobj = io.BytesIO(raw.read()) if hasattr(raw, "read") else io.BytesIO(raw)
                with h5py.File(fileobj, "r") as h5:
                    ds = None
                    for p in ["/Soil_Moisture_Retrieval_Data_AM/soil_moisture", "/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm"]:
                        if p in h5:
                            ds = h5[p][:]
                            break
                    if ds is None:
                        continue

                    mean_sm = float(np.nanmean(ds))
                    date_str = self._granule_date(entry) or start_date
                    for row in grid_df.itertuples(index=False):
                        rows.append({
                            "date": date_str,
                            "cell_id": row.cell_id,
                            "lat_center": row.lat_center,
                            "lon_center": row.lon_center,
                            "soil_moisture_surface_m3m3": round(mean_sm, 4),
                        })
            except Exception:
                continue

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "soil_moisture_surface_m3m3"])
        out["soil_moisture_surface_m3m3"] = pd.to_numeric(out["soil_moisture_surface_m3m3"], errors="coerce").clip(0, 1)
        return out.dropna(subset=["date", "cell_id", "soil_moisture_surface_m3m3"]).drop_duplicates().reset_index(drop=True)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)
