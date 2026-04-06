from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import time
import requests
import pandas as pd


APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"


@dataclass
class LPDAACMODISNDVIClient:
    username: str
    password: str
    product: str = "MOD13A2.061"
    layer: str = "_1_km_16_days_NDVI"

    def _login(self) -> str:
        r = requests.post(f"{APPEEARS_URL}/login", auth=(self.username, self.password), timeout=60)
        r.raise_for_status()
        return r.json()["token"]

    def fetch_grid_ndvi(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        token = self._login()
        headers = {"Authorization": f"Bearer {token}"}

        task = {
            "task_type": "point",
            "task_name": "wildfire_modis_ndvi",
            "params": {
                "dates": [{"startDate": start_date, "endDate": end_date}],
                "layers": [{"product": self.product, "layer": self.layer}],
                "output": {"format": {"type": "csv"}},
                "points": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [float(row.lon_center), float(row.lat_center)]},
                            "properties": {"cell_id": row.cell_id, "lat_center": row.lat_center, "lon_center": row.lon_center}
                        }
                        for row in grid_df.itertuples(index=False)
                    ],
                },
            },
        }

        r = requests.post(f"{APPEEARS_URL}/task", json=task, headers=headers, timeout=120)
        r.raise_for_status()
        task_id = r.json()["task_id"]

        while True:
            status = requests.get(f"{APPEEARS_URL}/task/{task_id}", headers=headers, timeout=60).json()
            if status.get("status") == "done":
                break
            if status.get("status") in {"error", "failed"}:
                raise RuntimeError(f"AppEEARS task failed: {status}")
            time.sleep(10)

        bundle = requests.get(f"{APPEEARS_URL}/bundle/{task_id}", headers=headers, timeout=60).json()
        files = bundle.get("files", [])
        if not files:
            return pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "ndvi", "ndvi_source"])

        file_id = files[0]["file_id"]
        download = requests.get(f"{APPEEARS_URL}/bundle/{task_id}/{file_id}", headers=headers, timeout=120)
        download.raise_for_status()
        df = pd.read_csv(io.BytesIO(download.content))

        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("calendar_date") or list(df.columns)[0]
        cell_col = cols.get("cell_id")
        lat_col = cols.get("lat_center") or cols.get("latitude")
        lon_col = cols.get("lon_center") or cols.get("longitude")
        ndvi_col = None
        for c in df.columns:
            if "ndvi" in c.lower():
                ndvi_col = c
                break
        if ndvi_col is None:
            raise ValueError("Could not find NDVI column in AppEEARS response")

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "cell_id": df[cell_col].astype(str) if cell_col else None,
            "lat_center": pd.to_numeric(df[lat_col], errors="coerce") if lat_col else None,
            "lon_center": pd.to_numeric(df[lon_col], errors="coerce") if lon_col else None,
            "ndvi": pd.to_numeric(df[ndvi_col], errors="coerce") / 10000.0,
        })
        out["ndvi_source"] = "modis_lpdaac"
        out = out.dropna(subset=["date", "ndvi"]).copy()

        if "cell_id" not in out.columns or out["cell_id"].isna().any():
            if out["lat_center"].notna().all() and out["lon_center"].notna().all():
                out["cell_id"] = out["lat_center"].map(lambda x: f"{x:.4f}") + "_" + out["lon_center"].map(lambda x: f"{x:.4f}")
        return out.drop_duplicates().reset_index(drop=True)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)
