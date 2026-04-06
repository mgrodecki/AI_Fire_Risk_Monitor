from __future__ import annotations

from pathlib import Path
import argparse
import json
import urllib.request
import pandas as pd


RISK_COLORS = {
    "low": "#2a9d8f",
    "moderate": "#e9c46a",
    "high": "#f4a261",
    "extreme": "#e76f51",
}

STATE_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

DEFAULT_PLACES = [
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "Sacramento", "lat": 38.5816, "lon": -121.4944},
    {"name": "Fresno", "lat": 36.7378, "lon": -119.7871},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "San Diego", "lat": 32.7157, "lon": -117.1611},
    {"name": "Redding", "lat": 40.5865, "lon": -122.3917},
    {"name": "Santa Rosa", "lat": 38.4404, "lon": -122.7141},
    {"name": "Reno", "lat": 39.5296, "lon": -119.8138},
    {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398},
    {"name": "Eugene", "lat": 44.0521, "lon": -123.0868},
]


def _radius_from_score(score: float) -> float:
    if score < 0.02:
        return 4.5
    if score < 0.05:
        return 6.0
    if score < 0.15:
        return 7.5
    return 9.0


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _download_json(url: str, cache_path: Path) -> dict:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = resp.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return json.loads(text)


def _iter_coords(geometry: dict):
    if not geometry:
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            for lon, lat in ring:
                yield float(lon), float(lat)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for lon, lat in ring:
                    yield float(lon), float(lat)


def _feature_bbox(geometry: dict):
    pts = list(_iter_coords(geometry))
    if not pts:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return (min(lons), min(lats), max(lons), max(lats))


def _bbox_intersects(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def _geometry_to_rings(geometry: dict) -> list[list[list[float]]]:
    rings = []
    if not geometry:
        return rings
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            rings.append([[float(lon), float(lat)] for lon, lat in ring])
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                rings.append([[float(lon), float(lat)] for lon, lat in ring])
    return rings


def _filter_overlay_features(geojson: dict, bbox: tuple[float, float, float, float], label_key: str | None = None) -> list[dict]:
    out = []
    for feat in geojson.get("features", []):
        geom = feat.get("geometry")
        fb = _feature_bbox(geom)
        if fb is None:
            continue
        if not _bbox_intersects(fb, bbox):
            continue
        props = feat.get("properties", {}) or {}
        label = props.get(label_key) if label_key else None
        out.append({"label": label, "rings": _geometry_to_rings(geom)})
    return out


def _load_overlays(lon_min: float, lon_max: float, lat_min: float, lat_max: float, cache_dir: Path) -> dict:
    # Slightly expand the envelope so near-edge boundaries/labels are retained.
    bbox = (lon_min - 2.0, lat_min - 2.0, lon_max + 2.0, lat_max + 2.0)
    overlays = {"states": [], "counties": [], "places": []}

    try:
        states = _download_json(STATE_GEOJSON_URL, cache_dir / "us_states.geojson")
        overlays["states"] = _filter_overlay_features(states, bbox, label_key="name")
    except Exception:
        overlays["states"] = []

    try:
        counties = _download_json(COUNTY_GEOJSON_URL, cache_dir / "us_counties.geojson")
        overlays["counties"] = _filter_overlay_features(counties, bbox, label_key="NAME")
    except Exception:
        overlays["counties"] = []

    places = []
    for p in DEFAULT_PLACES:
        if bbox[1] <= p["lat"] <= bbox[3] and bbox[0] <= p["lon"] <= bbox[2]:
            places.append(p)
    overlays["places"] = places
    return overlays


def build_offline_html(df: pd.DataFrame, title: str, overlays: dict) -> str:
    thresholds = {"low": 0.10, "moderate": 0.25, "high": 0.50}
    counts = df["risk_class"].value_counts().to_dict()

    lon_min = _safe_float(df["lon_center"].min())
    lon_max = _safe_float(df["lon_center"].max())
    lat_min = _safe_float(df["lat_center"].min())
    lat_max = _safe_float(df["lat_center"].max())

    features = []
    for row in df.itertuples(index=False):
        risk_class = str(row.risk_class).lower()
        features.append(
            {
                "cell_id": str(row.cell_id),
                "lat": _safe_float(row.lat_center),
                "lon": _safe_float(row.lon_center),
                "risk_class": risk_class,
                "risk_score": _safe_float(row.risk_score),
                "pred_fire_next_1d": _safe_float(row.pred_fire_next_1d),
                "vpd_kpa": _safe_float(getattr(row, "vpd_kpa", 0.0)),
                "fwi_proxy": _safe_float(getattr(row, "fwi_proxy", 0.0)),
                "ndvi": _safe_float(getattr(row, "ndvi", 0.0)),
                "soil_moisture_surface_m3m3": _safe_float(getattr(row, "soil_moisture_surface_m3m3", 0.0)),
                "lightning_count": _safe_float(getattr(row, "lightning_count", 0.0)),
                "color": RISK_COLORS.get(risk_class, "#6c757d"),
                "radius": _radius_from_score(_safe_float(row.risk_score)),
            }
        )

    payload = {
        "title": title,
        "thresholds": thresholds,
        "counts": counts,
        "colors": RISK_COLORS,
        "stats": {
            "n": int(len(df)),
            "min": _safe_float(df["risk_score"].min()),
            "mean": _safe_float(df["risk_score"].mean()),
            "max": _safe_float(df["risk_score"].max()),
        },
        "bounds": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
        "features": features,
        "overlays": overlays,
    }
    data_json = json.dumps(payload)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f7f7f7;
      --panel: #ffffff;
      --text: #202124;
      --muted: #5f6368;
      --border: #d9d9d9;
    }
    html, body { margin: 0; background: var(--bg); color: var(--text); font-family: Arial, sans-serif; }
    .app { display: grid; grid-template-columns: 340px 1fr; gap: 12px; min-height: 100vh; padding: 12px; box-sizing: border-box; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
    h1 { font-size: 16px; margin: 0 0 10px 0; }
    .sub { color: var(--muted); font-size: 12px; margin-bottom: 10px; }
    .legend-row { display: flex; align-items: center; justify-content: space-between; font-size: 12px; margin: 6px 0; }
    .left { display: flex; align-items: center; gap: 8px; }
    .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; border: 1px solid rgba(0,0,0,0.2); }
    .stats, .thresholds, .score-explain { margin-top: 10px; padding-top: 8px; border-top: 1px solid #ededed; font-size: 12px; line-height: 1.45; }
    .map-wrap { position: relative; background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    svg { width: 100%; height: calc(100vh - 24px); display: block; background: #fff; }
    .tooltip { position: absolute; pointer-events: none; background: rgba(32,33,36,0.95); color: #fff; padding: 8px 10px; border-radius: 6px; font-size: 12px; line-height: 1.35; min-width: 230px; transform: translate(8px, 8px); display: none; z-index: 4; }
    .controls { margin-top: 10px; font-size: 12px; display: grid; gap: 6px; }
    .axis-label { fill: #666; font-size: 11px; }
    .tick { stroke: #ececec; stroke-width: 1; }
    .county { fill: none; stroke: #b7b7b7; stroke-width: 0.8; }
    .state { fill: none; stroke: #666; stroke-width: 1.5; }
    .place { fill: #3b3b3b; font-size: 11px; font-weight: 600; }
    .place-dot { fill: #222; }
    .point { cursor: pointer; stroke: rgba(0,0,0,0.2); stroke-width: 1; fill-opacity: 0.85; }
  </style>
</head>
<body>
  <div class="app">
    <div class="panel">
      <h1 id="title"></h1>
      <div class="sub">Offline interactive risk map with boundaries, graticule, and place labels</div>
      <div class="controls">
        <label><input type="checkbox" id="toggle-low" checked /> Show Low</label>
        <label><input type="checkbox" id="toggle-moderate" checked /> Show Moderate</label>
        <label><input type="checkbox" id="toggle-high" checked /> Show High</label>
        <label><input type="checkbox" id="toggle-extreme" checked /> Show Extreme</label>
        <label><input type="checkbox" id="toggle-counties" checked /> Show County Boundaries</label>
        <label><input type="checkbox" id="toggle-states" checked /> Show State Boundaries</label>
        <label><input type="checkbox" id="toggle-places" checked /> Show Place Names</label>
        <label><input type="checkbox" id="toggle-grid" checked /> Show Lat/Lon Grid</label>
      </div>
      <div id="legend"></div>
      <div id="score-explain" class="score-explain"></div>
      <div id="thresholds" class="thresholds"></div>
      <div id="stats" class="stats"></div>
    </div>
    <div class="map-wrap">
      <svg id="svg" viewBox="0 0 1200 760" preserveAspectRatio="xMidYMid meet"></svg>
      <div id="tooltip" class="tooltip"></div>
    </div>
  </div>

  <script>
    const data = __DATA_JSON__;
    const svg = document.getElementById('svg');
    const tooltip = document.getElementById('tooltip');
    const W = 1200, H = 760, PAD = 58;

    document.getElementById('title').textContent = data.title;
    const lonMin = data.bounds.lon_min, lonMax = data.bounds.lon_max;
    const latMin = data.bounds.lat_min, latMax = data.bounds.lat_max;
    const lonRange = Math.max(1e-9, lonMax - lonMin);
    const latRange = Math.max(1e-9, latMax - latMin);
    const xScale = lon => PAD + ((lon - lonMin) / lonRange) * (W - 2 * PAD);
    const yScale = lat => H - PAD - ((lat - latMin) / latRange) * (H - 2 * PAD);

    const groups = {
      grid: document.createElementNS('http://www.w3.org/2000/svg', 'g'),
      counties: document.createElementNS('http://www.w3.org/2000/svg', 'g'),
      states: document.createElementNS('http://www.w3.org/2000/svg', 'g'),
      places: document.createElementNS('http://www.w3.org/2000/svg', 'g'),
      points: document.createElementNS('http://www.w3.org/2000/svg', 'g')
    };
    Object.values(groups).forEach(g => svg.appendChild(g));

    function makeLine(x1, y1, x2, y2, klass='tick') {
      const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      l.setAttribute('x1', x1); l.setAttribute('y1', y1);
      l.setAttribute('x2', x2); l.setAttribute('y2', y2);
      l.setAttribute('class', klass);
      return l;
    }
    function makeText(x, y, txt, anchor='middle', cls='axis-label') {
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', x); t.setAttribute('y', y);
      t.setAttribute('class', cls);
      t.setAttribute('text-anchor', anchor);
      t.textContent = txt;
      return t;
    }
    function ringsToPath(rings) {
      let d = '';
      for (const ring of rings) {
        if (!ring.length) continue;
        const [lon0, lat0] = ring[0];
        d += `M ${xScale(lon0)} ${yScale(lat0)} `;
        for (let i = 1; i < ring.length; i++) {
          const [lon, lat] = ring[i];
          d += `L ${xScale(lon)} ${yScale(lat)} `;
        }
        d += 'Z ';
      }
      return d.trim();
    }

    // Graticule + axis labels
    for (let i = 0; i <= 6; i++) {
      const x = PAD + (i / 6) * (W - 2 * PAD);
      const y = PAD + (i / 6) * (H - 2 * PAD);
      groups.grid.appendChild(makeLine(x, PAD, x, H - PAD, 'tick'));
      groups.grid.appendChild(makeLine(PAD, y, W - PAD, y, 'tick'));
      const lon = lonMin + (i / 6) * lonRange;
      const lat = latMax - (i / 6) * latRange;
      groups.grid.appendChild(makeText(x, H - PAD + 18, lon.toFixed(2)));
      groups.grid.appendChild(makeText(PAD - 8, y + 4, lat.toFixed(2), 'end'));
    }
    groups.grid.appendChild(makeText(W / 2, H - 8, 'Longitude'));
    groups.grid.appendChild(makeText(16, H / 2, 'Latitude', 'middle'));

    // Counties (thin)
    for (const f of data.overlays.counties || []) {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', ringsToPath(f.rings));
      p.setAttribute('class', 'county');
      groups.counties.appendChild(p);
    }

    // States (thicker)
    for (const f of data.overlays.states || []) {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', ringsToPath(f.rings));
      p.setAttribute('class', 'state');
      groups.states.appendChild(p);
    }

    // Places
    for (const place of data.overlays.places || []) {
      const x = xScale(place.lon), y = yScale(place.lat);
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', x); dot.setAttribute('cy', y);
      dot.setAttribute('r', 2.4); dot.setAttribute('class', 'place-dot');
      groups.places.appendChild(dot);
      const lbl = makeText(x + 5, y - 5, place.name, 'start', 'place');
      groups.places.appendChild(lbl);
    }

    // Risk points
    const pointEls = [];
    for (const f of data.features) {
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', xScale(f.lon));
      c.setAttribute('cy', yScale(f.lat));
      c.setAttribute('r', f.radius);
      c.setAttribute('fill', f.color);
      c.setAttribute('class', `point risk-${f.risk_class}`);
      c.dataset.risk = f.risk_class;
      c.dataset.payload = JSON.stringify(f);

      c.addEventListener('mousemove', (e) => {
        const d = JSON.parse(c.dataset.payload);
        tooltip.style.display = 'block';
        tooltip.style.left = `${e.clientX - 350}px`;
        tooltip.style.top = `${e.clientY - 42}px`;
        tooltip.innerHTML = `
          <b>Cell:</b> ${d.cell_id}<br/>
          <b>Class:</b> ${d.risk_class}<br/>
          <b>Risk score:</b> ${d.risk_score.toFixed(6)}<br/>
          <b>Pred fire 1d:</b> ${d.pred_fire_next_1d.toFixed(6)}<br/>
          <hr/>
          <b>VPD:</b> ${d.vpd_kpa.toFixed(3)} kPa<br/>
          <b>FWI proxy:</b> ${d.fwi_proxy.toFixed(2)}<br/>
          <b>NDVI:</b> ${d.ndvi.toFixed(3)}<br/>
          <b>Soil moisture:</b> ${d.soil_moisture_surface_m3m3.toFixed(3)}<br/>
          <b>Lightning:</b> ${d.lightning_count.toFixed(2)}
        `;
      });
      c.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
      groups.points.appendChild(c);
      pointEls.push(c);
    }

    // Legend + stats
    const legend = document.getElementById('legend');
    const classes = ['low', 'moderate', 'high', 'extreme'];
    legend.innerHTML = classes.map(c => {
      const label = c.charAt(0).toUpperCase() + c.slice(1);
      const range = c === 'low'
        ? `(< ${data.thresholds.low.toFixed(2)})`
        : c === 'moderate'
          ? `(${data.thresholds.low.toFixed(2)}-${data.thresholds.moderate.toFixed(2)})`
          : c === 'high'
            ? `(${data.thresholds.moderate.toFixed(2)}-${data.thresholds.high.toFixed(2)})`
            : `(>= ${data.thresholds.high.toFixed(2)})`;
      return `<div class="legend-row"><div class="left"><span class="dot" style="background:${data.colors[c]}"></span><b>${label}</b> ${range}</div><div>${data.counts[c] || 0}</div></div>`;
    }).join('');
    document.getElementById('thresholds').innerHTML =
      `<b>Thresholds</b><br/>low=${data.thresholds.low.toFixed(2)}, moderate=${data.thresholds.moderate.toFixed(2)}, high=${data.thresholds.high.toFixed(2)}`;
    document.getElementById('score-explain').innerHTML =
      `<b>Score Labels</b><br/>` +
      `<b>pred_fire_next_1d</b>: model probability (0-1) that a fire occurs in the cell on the next day.<br/>` +
      `<b>risk_score</b>: same probability used for ranking and color mapping.<br/>` +
      `<b>risk_class</b>: threshold-based bucket derived from risk_score (low/moderate/high/extreme).`;
    document.getElementById('stats').innerHTML =
      `<b>Score Stats</b><br/>Min: ${data.stats.min.toFixed(6)}<br/>Mean: ${data.stats.mean.toFixed(6)}<br/>Max: ${data.stats.max.toFixed(6)}<br/>Cells: ${data.stats.n}<br/>State overlays: ${(data.overlays.states||[]).length}<br/>County overlays: ${(data.overlays.counties||[]).length}<br/>Place labels: ${(data.overlays.places||[]).length}`;

    function refreshVisibility() {
      const show = {
        low: document.getElementById('toggle-low').checked,
        moderate: document.getElementById('toggle-moderate').checked,
        high: document.getElementById('toggle-high').checked,
        extreme: document.getElementById('toggle-extreme').checked
      };
      pointEls.forEach(el => { el.style.display = show[el.dataset.risk] ? '' : 'none'; });
      groups.counties.style.display = document.getElementById('toggle-counties').checked ? '' : 'none';
      groups.states.style.display = document.getElementById('toggle-states').checked ? '' : 'none';
      groups.places.style.display = document.getElementById('toggle-places').checked ? '' : 'none';
      groups.grid.style.display = document.getElementById('toggle-grid').checked ? '' : 'none';
    }
    ['toggle-low','toggle-moderate','toggle-high','toggle-extreme','toggle-counties','toggle-states','toggle-places','toggle-grid']
      .forEach(id => document.getElementById(id).addEventListener('change', refreshVisibility));
    refreshVisibility();
  </script>
</body>
</html>
"""
    return html.replace("__DATA_JSON__", data_json).replace("__TITLE__", title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an offline interactive SVG map from wildfire prediction CSV.")
    parser.add_argument("--input", required=True, help="Path to predictions_real_YYYY-MM-DD.csv")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {input_path}")
    output_path = Path(args.output) if args.output else input_path.with_suffix(".offline_map.html")

    df = pd.read_csv(input_path)
    required = {"lat_center", "lon_center", "cell_id", "risk_score", "risk_class", "pred_fire_next_1d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for map: {sorted(missing)}")

    lon_min = _safe_float(df["lon_center"].min())
    lon_max = _safe_float(df["lon_center"].max())
    lat_min = _safe_float(df["lat_center"].min())
    lat_max = _safe_float(df["lat_center"].max())
    overlays = _load_overlays(lon_min, lon_max, lat_min, lat_max, cache_dir=Path("data/external/boundaries"))

    date_label = input_path.stem.replace("predictions_real_", "")
    html = build_offline_html(df, f"Wildfire Risk Map (Offline) - {date_label}", overlays=overlays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
