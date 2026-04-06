from __future__ import annotations

import numpy as np
import pandas as pd


def build_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_deg: float,
) -> pd.DataFrame:
    lats = np.arange(lat_min, lat_max, resolution_deg)
    lons = np.arange(lon_min, lon_max, resolution_deg)

    rows = []
    for lat in lats:
        for lon in lons:
            lat_center = round(float(lat + resolution_deg / 2.0), 4)
            lon_center = round(float(lon + resolution_deg / 2.0), 4)
            rows.append(
                {
                    "cell_id": f"{lat_center:.4f}_{lon_center:.4f}",
                    "lat_center": lat_center,
                    "lon_center": lon_center,
                    "area_km2": 111.0 * 111.0 * (resolution_deg ** 2),
                }
            )
    return pd.DataFrame(rows)
