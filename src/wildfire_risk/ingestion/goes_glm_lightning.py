from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

try:
    import s3fs
except Exception:
    s3fs = None


@dataclass
class GOESGLMClient:
    satellite: str = "goes19"
    product: str = "GLM-L2-LCFA"

    def _bucket(self) -> str:
        return f"noaa-{self.satellite}"

    def _prefixes_for_day(self, date_str: str) -> list[str]:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return [f"{self._bucket()}/{self.product}/{dt.year}/{dt.timetuple().tm_yday:03d}/{hour:02d}/" for hour in range(24)]

    def fetch_grid_lightning(self, grid_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if s3fs is None:
            raise ImportError("s3fs is not installed")

        fs = s3fs.S3FileSystem(anon=True)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        rows = []

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            count_files = 0
            for prefix in self._prefixes_for_day(date_str):
                try:
                    count_files += len(fs.ls(prefix))
                except Exception:
                    continue

            lightning_probability = min(count_files / 500.0, 1.0)
            lightning_count = float(max(count_files, 0))
            dry_lightning_count = round(lightning_count * 0.3, 2)

            for row in grid_df.itertuples(index=False):
                rows.append({
                    "date": date_str,
                    "cell_id": row.cell_id,
                    "lat_center": row.lat_center,
                    "lon_center": row.lon_center,
                    "lightning_count": lightning_count,
                    "dry_lightning_count": dry_lightning_count,
                    "lightning_probability": lightning_probability,
                })

        return pd.DataFrame(rows)

    def save_csv(self, df: pd.DataFrame, output_path: str | Path) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return str(path)
