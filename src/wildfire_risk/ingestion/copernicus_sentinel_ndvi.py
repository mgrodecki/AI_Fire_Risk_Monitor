from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import requests
import pandas as pd


CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


@dataclass
class CopernicusSentinelNDVIClient:
    username: str
    password: str
    collection: str = "SENTINEL-2"
    processing_level: str = "S2MSI2A"
    cloud_cover_max: int = 30

    def _search_products(self, lon: float, lat: float, start_date: str, end_date: str) -> dict:
        filt = (
            f"Collection/Name eq '{self.collection}' and "
            f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
            f"ContentDate/Start le {end_date}T23:59:59.999Z"
        )
        params = {"$filter": filt, "$top": 20, "$orderby": "ContentDate/Start desc"}
        r = requests.get(CATALOG_URL, params=params, timeout=90)
        r.raise_for_status()
        return r.json()

    def fetch_refinement_ndvi(self, grid_df: pd.DataFrame, firms_events_df: pd.DataFrame, start_date: str, end_date: str, refine_radius_km: int = 20) -> pd.DataFrame:
        rows = []
        fire_cells = set(firms_events_df["cell_id"].astype(str).tolist()) if not firms_events_df.empty and "cell_id" in firms_events_df.columns else set()
        target_grid = grid_df[grid_df["cell_id"].isin(fire_cells)].copy()

        if target_grid.empty:
            return pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "ndvi_source", "sentinel_product_id"])

        for row in target_grid.itertuples(index=False):
            payload = self._search_products(float(row.lon_center), float(row.lat_center), start_date, end_date)
            products = payload.get("value", [])
            if not products:
                continue
            prod = products[0]
            date_val = prod.get("ContentDate", {}).get("Start", start_date)[:10]
            rows.append({
                "date": date_val,
                "cell_id": row.cell_id,
                "lat_center": row.lat_center,
                "lon_center": row.lon_center,
                "ndvi_source": "sentinel2_cdse_refine",
                "sentinel_product_id": prod.get("Id"),
            })
        return pd.DataFrame(rows)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)
