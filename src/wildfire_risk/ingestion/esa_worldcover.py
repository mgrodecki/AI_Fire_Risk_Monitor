from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:
    import rasterio
except Exception:  # pragma: no cover - optional dependency at runtime
    rasterio = None


WORLDCOVER_CODE_TO_LABEL = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

WORLDCOVER_LABEL_TO_VEGETATION_TYPE = {
    "tree_cover": "forest",
    "shrubland": "shrub",
    "grassland": "grass",
    "cropland": "agriculture",
    "built_up": "urban",
    "bare_sparse_vegetation": "shrub",
    "snow_ice": "water",
    "permanent_water": "water",
    "herbaceous_wetland": "grass",
    "mangroves": "forest",
    "moss_lichen": "grass",
}

DEFAULT_STAC_URLS = [
    "https://services.terrascope.be/stac/v1",
    "https://planetarycomputer.microsoft.com/api/stac/v1",
]


@dataclass
class WorldCoverConfig:
    stac_urls: list[str]
    collection_terrascope: str = "urn:eop:VITO:ESA_WorldCover_10m_2021_V2"
    collection_planetary: str = "esa-worldcover"
    asset_key: str = "map"
    user_agent: str = "wildfire-risk/1.0"
    timeout_seconds: int = 30


def _to_worldcover_config(api_connectors: dict | None) -> WorldCoverConfig:
    cfg = api_connectors or {}
    raw_urls = cfg.get("worldcover_stac_urls", DEFAULT_STAC_URLS)
    if isinstance(raw_urls, str):
        stac_urls = [raw_urls]
    else:
        stac_urls = [str(x) for x in raw_urls]
    stac_urls = [u.strip() for u in stac_urls if u and u.strip()]
    if not stac_urls:
        stac_urls = list(DEFAULT_STAC_URLS)

    return WorldCoverConfig(
        stac_urls=stac_urls,
        collection_terrascope=str(
            cfg.get("worldcover_collection_terrascope", "urn:eop:VITO:ESA_WorldCover_10m_2021_V2")
        ),
        collection_planetary=str(cfg.get("worldcover_collection_planetary", "esa-worldcover")),
        asset_key=str(cfg.get("worldcover_asset_key", "map")),
        user_agent=str(cfg.get("worldcover_user_agent", "wildfire-risk/1.0")),
        timeout_seconds=int(cfg.get("worldcover_timeout_seconds", 30)),
    )


def _collection_for_url(stac_url: str, cfg: WorldCoverConfig) -> str:
    if "planetarycomputer.microsoft.com" in stac_url.lower():
        return cfg.collection_planetary
    return cfg.collection_terrascope


def _search_worldcover_item(lat: float, lon: float, cfg: WorldCoverConfig) -> tuple[dict[str, Any] | None, str | None]:
    last_err: Exception | None = None
    for stac_url in cfg.stac_urls:
        payload = {
            "collections": [_collection_for_url(stac_url, cfg)],
            "intersects": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "limit": 1,
        }
        try:
            r = requests.post(
                f"{stac_url.rstrip('/')}/search",
                json=payload,
                timeout=cfg.timeout_seconds,
                headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
            )
            r.raise_for_status()
            features = r.json().get("features", [])
            if features:
                return features[0], stac_url
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err
    return None, None


def _asset_href_from_item(item: dict[str, Any], preferred_key: str) -> str | None:
    assets = item.get("assets", {})
    if preferred_key in assets and assets[preferred_key].get("href"):
        return str(assets[preferred_key]["href"])

    for key in ("map", "visual", "data"):
        if key in assets and assets[key].get("href"):
            return str(assets[key]["href"])

    for value in assets.values():
        href = value.get("href")
        if href:
            return str(href)
    return None


def _sign_planetary_href(href: str, cfg: WorldCoverConfig) -> str:
    try:
        q_href = quote(href, safe="")
        r = requests.get(
            f"https://planetarycomputer.microsoft.com/api/sas/v1/sign?href={q_href}",
            timeout=cfg.timeout_seconds,
            headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
        )
        r.raise_for_status()
        signed = r.json().get("href")
        if signed:
            return str(signed)
    except Exception:
        pass
    return href


def _normalize_asset_href(href: str) -> str:
    if href.startswith("s3://"):
        parts = href[5:].split("/", 1)
        if len(parts) == 2:
            bucket, key = parts
            return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def _sample_worldcover_code(href: str, lat: float, lon: float) -> int | None:
    if rasterio is None:
        raise RuntimeError("rasterio is required for ESA WorldCover sampling. Install with: pip install rasterio")
    with rasterio.open(_normalize_asset_href(href)) as ds:
        value = next(ds.sample([(float(lon), float(lat))]))[0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(value)
    except Exception:
        return None


def classify_worldcover_points(
    points_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    api_connectors: dict | None = None,
) -> pd.DataFrame:
    cfg = _to_worldcover_config(api_connectors)

    codes: list[int | None] = []
    labels: list[str | None] = []
    mapped_types: list[str | None] = []
    tile_cache: dict[tuple[int, int], tuple[int | None, str | None, str | None]] = {}

    for _, row in points_df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        cache_key = (int(round(lat * 100)), int(round(lon * 100)))
        if cache_key in tile_cache:
            c, l, m = tile_cache[cache_key]
            codes.append(c)
            labels.append(l)
            mapped_types.append(m)
            continue

        code: int | None = None
        label: str | None = None
        veg_type: str | None = None

        try:
            item, stac_url = _search_worldcover_item(lat=lat, lon=lon, cfg=cfg)
            if item:
                href = _asset_href_from_item(item, preferred_key=cfg.asset_key)
                if href:
                    if stac_url and "planetarycomputer.microsoft.com" in stac_url.lower():
                        href = _sign_planetary_href(href, cfg)
                    code = _sample_worldcover_code(href=href, lat=lat, lon=lon)
                    if code in WORLDCOVER_CODE_TO_LABEL:
                        label = WORLDCOVER_CODE_TO_LABEL[code]
                        veg_type = WORLDCOVER_LABEL_TO_VEGETATION_TYPE.get(label)
        except Exception:
            code = None
            label = None
            veg_type = None

        tile_cache[cache_key] = (code, label, veg_type)
        codes.append(code)
        labels.append(label)
        mapped_types.append(veg_type)

    return pd.DataFrame(
        {
            "worldcover_class_code": codes,
            "worldcover_class_label": labels,
            "vegetation_type_worldcover": mapped_types,
        },
        index=points_df.index,
    )

