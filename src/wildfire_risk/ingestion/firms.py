from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterable, Optional
import requests
import pandas as pd
import numpy as np


BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


@dataclass
class FIRMSClient:
    map_key: str
    timeout: int = 90

    def fetch_area_csv(
        self,
        source: str,
        bbox: str,
        day_range: int = 1,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        if not 1 <= day_range <= 5:
            raise ValueError("FIRMS day_range must be between 1 and 5")

        if date:
            url = f"{BASE_URL}/{self.map_key}/{source}/{bbox}/{day_range}/{date}"
        else:
            url = f"{BASE_URL}/{self.map_key}/{source}/{bbox}/{day_range}"

        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()

        text = r.text.strip()
        if not text:
            return pd.DataFrame()

        # FIRMS returns CSV text
        return pd.read_csv(StringIO(text))

    def fetch_area_range_chunked(
        self,
        source: str,
        bbox: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        if len(dates) == 0:
            return pd.DataFrame()

        frames = []
        i = 0
        while i < len(dates):
            chunk = dates[i : i + 5]
            chunk_start = chunk[0].strftime("%Y-%m-%d")
            day_range = len(chunk)
            frames.append(self.fetch_area_csv(source=source, bbox=bbox, day_range=day_range, date=chunk_start))
            i += 5

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True).drop_duplicates()
        return self.standardize(out)

    @staticmethod
    def standardize(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "acq_date" in out.columns and "acq_time" in out.columns:
            out["acq_datetime_utc"] = pd.to_datetime(
                out["acq_date"].astype(str) + " " + out["acq_time"].astype(str).str.zfill(4),
                format="%Y-%m-%d %H%M",
                utc=True,
                errors="coerce",
            )
        else:
            out["acq_datetime_utc"] = pd.NaT

        if "frp" in out.columns:
            out["frp_mw"] = pd.to_numeric(out["frp"], errors="coerce").fillna(0.0)
        else:
            out["frp_mw"] = 0.0

        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
        out["confidence"] = pd.to_numeric(out.get("confidence", 0), errors="coerce")
        out["instrument"] = out.get("instrument", "")
        out["satellite"] = out.get("satellite", "")
        out["daynight"] = out.get("daynight", "")
        out["detection_id"] = (
            out["latitude"].round(4).astype(str)
            + "_"
            + out["longitude"].round(4).astype(str)
            + "_"
            + out["acq_datetime_utc"].astype(str)
        )
        keep = [
            "detection_id",
            "latitude",
            "longitude",
            "acq_datetime_utc",
            "confidence",
            "frp_mw",
            "instrument",
            "satellite",
            "daynight",
        ]
        return out[keep].dropna(subset=["latitude", "longitude", "acq_datetime_utc"]).reset_index(drop=True)

    @staticmethod
    def attach_cells(events_df: pd.DataFrame, grid_df: pd.DataFrame, resolution_deg: float) -> pd.DataFrame:
        if events_df.empty:
            return events_df.copy()

        lat_min = float(grid_df["lat_center"].min() - resolution_deg / 2.0)
        lon_min = float(grid_df["lon_center"].min() - resolution_deg / 2.0)

        out = events_df.copy()
        lat_center = np.floor((out["latitude"] - lat_min) / resolution_deg) * resolution_deg + lat_min + resolution_deg / 2.0
        lon_center = np.floor((out["longitude"] - lon_min) / resolution_deg) * resolution_deg + lon_min + resolution_deg / 2.0

        out["lat_center"] = lat_center.round(4)
        out["lon_center"] = lon_center.round(4)
        out["cell_id"] = out["lat_center"].map(lambda x: f"{x:.4f}") + "_" + out["lon_center"].map(lambda x: f"{x:.4f}")

        valid_cells = set(grid_df["cell_id"].tolist())
        out = out[out["cell_id"].isin(valid_cells)].copy()
        out["event_date"] = out["acq_datetime_utc"].dt.strftime("%Y-%m-%d")
        return out.reset_index(drop=True)
