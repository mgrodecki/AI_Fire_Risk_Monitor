from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from wildfire_risk.ingestion.lpdaac_modis_ndvi import LPDAACMODISNDVIClient
from wildfire_risk.ingestion.copernicus_sentinel_ndvi import CopernicusSentinelNDVIClient


@dataclass
class HybridNDVIClient:
    earthdata_user: str
    earthdata_pass: str
    cdse_user: str
    cdse_pass: str
    modis_product: str = "MOD13A2.061"
    modis_layer: str = "_1_km_16_days_NDVI"
    sentinel_collection: str = "SENTINEL-2"
    sentinel_processing_level: str = "S2MSI2A"
    sentinel_cloud_cover_max: int = 30

    def fetch_hybrid_ndvi(self, grid_df: pd.DataFrame, firms_events_df: pd.DataFrame, start_date: str, end_date: str, refine_radius_km: int = 20) -> pd.DataFrame:
        modis = LPDAACMODISNDVIClient(
            username=self.earthdata_user,
            password=self.earthdata_pass,
            product=self.modis_product,
            layer=self.modis_layer,
        )
        baseline = modis.fetch_grid_ndvi(grid_df, start_date, end_date)

        sentinel = CopernicusSentinelNDVIClient(
            username=self.cdse_user,
            password=self.cdse_pass,
            collection=self.sentinel_collection,
            processing_level=self.sentinel_processing_level,
            cloud_cover_max=self.sentinel_cloud_cover_max,
        )
        refine = sentinel.fetch_refinement_ndvi(grid_df, firms_events_df, start_date, end_date, refine_radius_km=refine_radius_km)

        if baseline.empty:
            baseline = pd.DataFrame(columns=["date", "cell_id", "lat_center", "lon_center", "ndvi", "ndvi_source"])
        if refine.empty:
            return baseline

        merged = baseline.merge(
            refine[["date", "cell_id", "sentinel_product_id"]],
            on=["date", "cell_id"],
            how="left",
        )
        merged["ndvi_source"] = merged["sentinel_product_id"].notna().map(lambda x: "hybrid_modis_plus_sentinel" if x else "modis_lpdaac")
        return merged

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return str(p)
