from __future__ import annotations

from dataclasses import dataclass
import logging
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
LOGGER = logging.getLogger(__name__)


@dataclass
class OpenMeteoClient:
    timeout: int = 60
    max_retries: int = 5
    backoff_factor: float = 1.0

    def __post_init__(self) -> None:
        retry_cfg = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            status=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get_json(self, url: str, params: dict) -> dict:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def fetch_daily_history(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join([
                "temperature_2m_max",
                "relative_humidity_2m_min",
                "wind_speed_10m_max",
                "precipitation_sum",
            ]),
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        payload = self._get_json(ARCHIVE_URL, params)

        daily = payload.get("daily", {})
        if not daily or "time" not in daily:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "date": daily["time"],
                "t2m_max_c": daily.get("temperature_2m_max", []),
                "rh_min_pct": daily.get("relative_humidity_2m_min", []),
                "wind10m_max_ms": daily.get("wind_speed_10m_max", []),
                "precip_mm": daily.get("precipitation_sum", []),
                "lat_center": lat,
                "lon_center": lon,
            }
        )

    def fetch_daily_forecast(self, lat: float, lon: float, forecast_days: int = 7) -> pd.DataFrame:
        params = {
            "latitude": lat,
            "longitude": lon,
            "forecast_days": forecast_days,
            "daily": ",".join([
                "temperature_2m_max",
                "relative_humidity_2m_min",
                "wind_speed_10m_max",
                "precipitation_sum",
            ]),
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        payload = self._get_json(FORECAST_URL, params)

        daily = payload.get("daily", {})
        if not daily or "time" not in daily:
            return pd.DataFrame()

        return pd.DataFrame(
            {
                "date": daily["time"],
                "t2m_max_c": daily.get("temperature_2m_max", []),
                "rh_min_pct": daily.get("relative_humidity_2m_min", []),
                "wind10m_max_ms": daily.get("wind_speed_10m_max", []),
                "precip_mm": daily.get("precipitation_sum", []),
                "lat_center": lat,
                "lon_center": lon,
            }
        )

    def fetch_grid_history(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        frames = []
        for row in grid_df.itertuples(index=False):
            try:
                frames.append(self.fetch_daily_history(row.lat_center, row.lon_center, start_date, end_date))
            except requests.RequestException as exc:
                LOGGER.warning(
                    "Open-Meteo history request failed for lat=%s lon=%s: %s",
                    row.lat_center,
                    row.lon_center,
                    exc,
                )
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["cell_id"] = out["lat_center"].map(lambda x: f"{x:.4f}") + "_" + out["lon_center"].map(lambda x: f"{x:.4f}")
        return out

    def fetch_grid_forecast(self, grid_df: pd.DataFrame, forecast_days: int = 7) -> pd.DataFrame:
        frames = []
        for row in grid_df.itertuples(index=False):
            try:
                frames.append(self.fetch_daily_forecast(row.lat_center, row.lon_center, forecast_days))
            except requests.RequestException as exc:
                LOGGER.warning(
                    "Open-Meteo forecast request failed for lat=%s lon=%s: %s",
                    row.lat_center,
                    row.lon_center,
                    exc,
                )
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["cell_id"] = out["lat_center"].map(lambda x: f"{x:.4f}") + "_" + out["lon_center"].map(lambda x: f"{x:.4f}")
        return out
