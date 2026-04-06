from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd


RISK_COLORS = {
    "low": "#2a9d8f",
    "moderate": "#e9c46a",
    "high": "#f4a261",
    "extreme": "#e76f51",
}


def _radius_from_score(score: float) -> float:
    # Keep marker sizes readable even when probabilities are very small.
    if score < 0.02:
        return 5
    if score < 0.05:
        return 7
    if score < 0.15:
        return 9
    return 11


def build_map_html(df: pd.DataFrame, title: str) -> str:
    center_lat = float(df["lat_center"].mean())
    center_lon = float(df["lon_center"].mean())

    thresholds = {"low": 0.10, "moderate": 0.25, "high": 0.50}
    counts = df["risk_class"].value_counts().to_dict()
    score_min = float(df["risk_score"].min())
    score_max = float(df["risk_score"].max())
    score_mean = float(df["risk_score"].mean())

    features = []
    for row in df.itertuples(index=False):
        risk_class = str(row.risk_class).lower()
        color = RISK_COLORS.get(risk_class, "#6c757d")
        features.append(
            {
                "lat": float(row.lat_center),
                "lon": float(row.lon_center),
                "cell_id": str(row.cell_id),
                "risk_class": risk_class,
                "risk_score": float(row.risk_score),
                "pred_fire_next_1d": float(row.pred_fire_next_1d),
                "vpd_kpa": float(row.vpd_kpa),
                "fwi_proxy": float(row.fwi_proxy),
                "ndvi": float(row.ndvi),
                "soil_moisture_surface_m3m3": float(row.soil_moisture_surface_m3m3),
                "lightning_count": float(row.lightning_count),
                "color": color,
                "radius": _radius_from_score(float(row.risk_score)),
            }
        )

    payload = {
        "center": {"lat": center_lat, "lon": center_lon},
        "title": title,
        "features": features,
        "thresholds": thresholds,
        "counts": counts,
        "stats": {"min": score_min, "max": score_max, "mean": score_mean, "n": int(len(df))},
        "colors": RISK_COLORS,
    }
    data_json = json.dumps(payload)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__TITLE__</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      font-family: Arial, sans-serif;
    }}
    .legend {{
      background: #ffffff;
      border: 1px solid #d0d0d0;
      border-radius: 8px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.2);
      color: #222;
      line-height: 1.3;
      padding: 10px 12px;
      width: 295px;
    }}
    .legend h4 {{
      margin: 0 0 8px 0;
      font-size: 14px;
    }}
    .legend .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .legend .row {{
      margin: 3px 0;
      font-size: 12px;
    }}
    .legend .meta {{
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #eee;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const data = __DATA_JSON__;
    const map = L.map('map').setView([data.center.lat, data.center.lon], 6);

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const groups = {{
      low: L.layerGroup(),
      moderate: L.layerGroup(),
      high: L.layerGroup(),
      extreme: L.layerGroup()
    }};

    for (const f of data.features) {{
      const marker = L.circleMarker([f.lat, f.lon], {{
        radius: f.radius,
        color: f.color,
        fillColor: f.color,
        fillOpacity: 0.8,
        weight: 1
      }});

      const popup = `
        <b>Cell:</b> ${f.cell_id}<br/>
        <b>Risk Class:</b> ${f.risk_class}<br/>
        <b>Risk Score:</b> ${f.risk_score.toFixed(6)}<br/>
        <b>Pred Fire Next 1d:</b> ${f.pred_fire_next_1d.toFixed(6)}<br/>
        <hr/>
        <b>VPD (kPa):</b> ${f.vpd_kpa.toFixed(3)}<br/>
        <b>FWI Proxy:</b> ${f.fwi_proxy.toFixed(2)}<br/>
        <b>NDVI:</b> ${f.ndvi.toFixed(3)}<br/>
        <b>Soil Moisture:</b> ${f.soil_moisture_surface_m3m3.toFixed(3)}<br/>
        <b>Lightning Count:</b> ${f.lightning_count.toFixed(2)}
      `;
      marker.bindPopup(popup);

      const key = groups[f.risk_class] ? f.risk_class : "low";
      marker.addTo(groups[key]);
    }}

    groups.low.addTo(map);
    groups.moderate.addTo(map);
    groups.high.addTo(map);
    groups.extreme.addTo(map);

    L.control.layers(null, {{
      "Low": groups.low,
      "Moderate": groups.moderate,
      "High": groups.high,
      "Extreme": groups.extreme
    }}, {{ collapsed: false }}).addTo(map);

    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = `
        <h4>${data.title}</h4>
        <div class="row"><span class="swatch" style="background:${{data.colors.low}}"></span><b>Low</b> (&lt; ${{data.thresholds.low.toFixed(2)}}): ${{data.counts.low || 0}}</div>
        <div class="row"><span class="swatch" style="background:${{data.colors.moderate}}"></span><b>Moderate</b> (${{data.thresholds.low.toFixed(2)}}-${{data.thresholds.moderate.toFixed(2)}}): ${{data.counts.moderate || 0}}</div>
        <div class="row"><span class="swatch" style="background:${{data.colors.high}}"></span><b>High</b> (${{data.thresholds.moderate.toFixed(2)}}-${{data.thresholds.high.toFixed(2)}}): ${{data.counts.high || 0}}</div>
        <div class="row"><span class="swatch" style="background:${{data.colors.extreme}}"></span><b>Extreme</b> (≥ ${{data.thresholds.high.toFixed(2)}}): ${{data.counts.extreme || 0}}</div>
        <div class="meta">
          <b>Score Stats</b><br/>
          Min: ${{data.stats.min.toFixed(6)}}<br/>
          Mean: ${{data.stats.mean.toFixed(6)}}<br/>
          Max: ${{data.stats.max.toFixed(6)}}<br/>
          Cells: ${{data.stats.n}}
        </div>
      `;
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>"""
    return html_template.replace("__DATA_JSON__", data_json).replace("__TITLE__", title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an interactive HTML map from wildfire prediction CSV.")
    parser.add_argument("--input", required=True, help="Path to predictions_real_YYYY-MM-DD.csv")
    parser.add_argument("--output", default=None, help="Output HTML path (default: same name with .html)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".html")
    df = pd.read_csv(input_path)
    required = {"lat_center", "lon_center", "cell_id", "risk_score", "risk_class", "pred_fire_next_1d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for map: {sorted(missing)}")

    title = f"Wildfire Risk Map - {input_path.stem.replace('predictions_real_', '')}"
    html = build_map_html(df, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
