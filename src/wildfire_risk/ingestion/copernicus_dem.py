from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import pandas as pd

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency at runtime
    rasterio = None


@dataclass
class CopernicusDEMConfig:
    s3_root: str = "s3://copernicus-dem-30m"
    dataset: str = "GLO-30"
    resolution_m: int = 30


def _to_dem_config(api_connectors: dict | None) -> CopernicusDEMConfig:
    cfg = api_connectors or {}
    return CopernicusDEMConfig(
        s3_root=str(cfg.get("copdem_s3_root", "s3://copernicus-dem-30m")),
        dataset=str(cfg.get("copdem_dataset", "GLO-30")),
        resolution_m=int(cfg.get("copdem_resolution_m", 30)),
    )


def _tile_origin(lat: float, lon: float) -> tuple[int, int]:
    # Copernicus DEM tiles are 1x1 degree; use southwest corner.
    lat0 = math.floor(lat)
    lon0 = math.floor(lon)
    return lat0, lon0


def _tile_token(lat0: int, lon0: int) -> str:
    lat_hemi = "N" if lat0 >= 0 else "S"
    lon_hemi = "E" if lon0 >= 0 else "W"
    return f"{lat_hemi}{abs(lat0):02d}_00_{lon_hemi}{abs(lon0):03d}_00"


def _cog_code(resolution_m: int) -> str:
    # Copernicus naming uses COG_10 for 30m and COG_30 for 90m.
    return "COG_10" if int(resolution_m) <= 30 else "COG_30"


def _copdem_href(lat: float, lon: float, cfg: CopernicusDEMConfig) -> str:
    lat0, lon0 = _tile_origin(lat, lon)
    token = _tile_token(lat0, lon0)
    cog_code = _cog_code(cfg.resolution_m)
    stem = f"Copernicus_DSM_{cog_code}_{token}_DEM"
    return f"{cfg.s3_root.rstrip('/')}/{stem}/{stem}.tif"


@lru_cache(maxsize=1024)
def _open_dataset(href: str):
    if rasterio is None:
        raise RuntimeError("rasterio is required for Copernicus DEM sampling. Install with: pip install rasterio")

    # Prefer native path first (works with GDAL VSI S3 if available), then HTTPS fallback.
    try:
        return rasterio.open(href)
    except Exception:
        if href.startswith("s3://"):
            parts = href[5:].split("/", 1)
            if len(parts) == 2:
                bucket, key = parts
                https_url = f"https://{bucket}.s3.amazonaws.com/{key}"
                return rasterio.open(https_url)
        raise


def _pixel_size_m(ds, lat: float) -> tuple[float, float]:
    xres_deg = abs(float(ds.transform.a))
    yres_deg = abs(float(ds.transform.e))
    lat_rad = math.radians(lat)
    xres_m = xres_deg * 111320.0 * max(math.cos(lat_rad), 1e-6)
    yres_m = yres_deg * 110540.0
    return xres_m, yres_m


def _sample_elevation_and_slope(lat: float, lon: float, cfg: CopernicusDEMConfig) -> tuple[float | None, float | None]:
    href = _copdem_href(lat, lon, cfg)
    ds = _open_dataset(href)

    row, col = ds.index(lon, lat)
    if row < 1 or col < 1 or row >= ds.height - 1 or col >= ds.width - 1:
        return None, None

    window = rasterio.windows.Window(col_off=col - 1, row_off=row - 1, width=3, height=3)
    arr = ds.read(1, window=window, masked=True)
    if np.ma.isMaskedArray(arr):
        mask = np.ma.getmaskarray(arr)
        if mask.shape == arr.shape and mask[1, 1]:
            return None, None
        z = np.array(arr.filled(np.nan), dtype=float)
    else:
        z = np.array(arr, dtype=float)

    zc = float(z[1, 1])
    if not np.isfinite(zc):
        return None, None

    xres_m, yres_m = _pixel_size_m(ds, lat)
    dzdx = (z[1, 2] - z[1, 0]) / (2.0 * xres_m)
    dzdy = (z[2, 1] - z[0, 1]) / (2.0 * yres_m)
    if not np.isfinite(dzdx) or not np.isfinite(dzdy):
        return zc, None

    slope_rad = math.atan(math.sqrt(float(dzdx) ** 2 + float(dzdy) ** 2))
    slope_deg = math.degrees(slope_rad)
    return zc, float(np.clip(slope_deg, 0.0, 89.9))


def sample_copdem_points(
    points_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    api_connectors: dict | None = None,
) -> pd.DataFrame:
    """
    Sample Copernicus DEM elevation and derive slope for each input point.
    Returns:
    - dem_elevation_m
    - dem_slope_deg
    - dem_source
    """
    cfg = _to_dem_config(api_connectors)
    elevations: list[float | None] = []
    slopes: list[float | None] = []
    source = f"Copernicus DEM {cfg.dataset}"
    sources: list[str] = []

    for _, row in points_df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        elev = None
        slope = None
        try:
            elev, slope = _sample_elevation_and_slope(lat=lat, lon=lon, cfg=cfg)
        except Exception:
            elev, slope = None, None
        elevations.append(elev)
        slopes.append(slope)
        sources.append(source if slope is not None or elev is not None else "")

    return pd.DataFrame(
        {
            "dem_elevation_m": elevations,
            "dem_slope_deg": slopes,
            "dem_source": sources,
        },
        index=points_df.index,
    )
