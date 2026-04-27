from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wildfire_risk.config import load_settings
from wildfire_risk.data.grid import build_grid
from wildfire_risk.ingestion.copernicus_dem import sample_copdem_points
from wildfire_risk.utils.io import ensure_dir, write_csv


def _slope_class(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "unknown"
    x = float(v)
    if x < 5:
        return "flat"
    if x < 15:
        return "gentle"
    if x < 30:
        return "moderate"
    return "steep"


def _build_html(df: pd.DataFrame, title: str) -> str:
    points = []
    for row in df.itertuples(index=False):
        points.append(
            {
                "cell_id": str(row.cell_id),
                "lat": float(row.lat_center),
                "lon": float(row.lon_center),
                "slope_deg": None if pd.isna(row.dem_slope_deg) else float(row.dem_slope_deg),
                "elevation_m": None if pd.isna(row.dem_elevation_m) else float(row.dem_elevation_m),
                "slope_class": str(row.slope_class),
            }
        )

    payload = json.dumps(points)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; }}
    #map {{ height: 100vh; width: 100vw; }}
    .legend {{
      position: absolute; z-index: 1000; right: 12px; bottom: 12px;
      background: #fff; border: 1px solid #bbb; border-radius: 6px; padding: 10px 12px;
      font-size: 12px; line-height: 1.4;
    }}
    .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  </style>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <div><b>Slope Classes</b></div>
    <div><span class="sw" style="background:#2E8B57"></span>flat (&lt; 5°)</div>
    <div><span class="sw" style="background:#F0C419"></span>gentle (5-15°)</div>
    <div><span class="sw" style="background:#F28C28"></span>moderate (15-30°)</div>
    <div><span class="sw" style="background:#C0392B"></span>steep (&ge; 30°)</div>
    <div><span class="sw" style="background:#999"></span>unknown</div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const pts = {payload};
    const lats = pts.map(p => p.lat), lons = pts.map(p => p.lon);
    const cLat = (Math.min(...lats) + Math.max(...lats)) / 2;
    const cLon = (Math.min(...lons) + Math.max(...lons)) / 2;
    const map = L.map('map').setView([cLat, cLon], 6);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18, attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    function color(cls) {{
      if (cls === 'flat') return '#2E8B57';
      if (cls === 'gentle') return '#F0C419';
      if (cls === 'moderate') return '#F28C28';
      if (cls === 'steep') return '#C0392B';
      return '#999';
    }}
    pts.forEach(p => {{
      L.circleMarker([p.lat, p.lon], {{
        radius: 6, color: color(p.slope_class), fillColor: color(p.slope_class), fillOpacity: 0.85, weight: 1
      }}).addTo(map).bindPopup(
        '<b>Cell:</b> ' + p.cell_id + '<br/>' +
        '<b>Slope (deg):</b> ' + (p.slope_deg == null ? 'NA' : p.slope_deg.toFixed(2)) + '<br/>' +
        '<b>Elevation (m):</b> ' + (p.elevation_m == null ? 'NA' : p.elevation_m.toFixed(1)) + '<br/>' +
        '<b>Class:</b> ' + p.slope_class
      );
    }});
  </script>
</body>
</html>"""


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Derive a Copernicus DEM slope map over the configured grid.")
    parser.add_argument("--output", default=str(Path(settings.paths.curated_dir) / "slope_map_copdem.csv"))
    parser.add_argument("--html-output", default=None, help="Optional HTML map output path.")
    parser.add_argument("--resolution-m", type=int, default=None, choices=[30, 90], help="DEM resolution (30=GLO-30, 90=GLO-90).")
    args = parser.parse_args()

    grid = build_grid(
        lat_min=settings.grid.lat_min,
        lat_max=settings.grid.lat_max,
        lon_min=settings.grid.lon_min,
        lon_max=settings.grid.lon_max,
        resolution_deg=settings.grid.resolution_deg,
    )

    connectors = dict(settings.api_connectors or {})
    if args.resolution_m is not None:
        connectors["copdem_resolution_m"] = int(args.resolution_m)
        connectors["copdem_dataset"] = "GLO-30" if int(args.resolution_m) <= 30 else "GLO-90"

    sampled = sample_copdem_points(grid, lat_col="lat_center", lon_col="lon_center", api_connectors=connectors)
    out = pd.concat([grid.copy(), sampled], axis=1)
    out["slope_class"] = out["dem_slope_deg"].apply(_slope_class)

    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    write_csv(out, out_path)
    print(str(out_path))

    if args.html_output:
        html = _build_html(out, title=f"Copernicus DEM Slope Map ({connectors.get('copdem_dataset', 'GLO-30')})")
        html_path = Path(args.html_output)
        ensure_dir(html_path.parent)
        html_path.write_text(html, encoding="utf-8")
        print(str(html_path))


if __name__ == "__main__":
    main()

