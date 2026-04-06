from __future__ import annotations

from pathlib import Path
from datetime import datetime

from wildfire_risk.utils.io import write_json, ensure_dir


def write_model_manifest(metrics: dict, output_dir: str, model_filename: str) -> str:
    ensure_dir(output_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_dir) / f"model_manifest_{ts}.json"
    payload = {
        "created_at_utc": ts,
        "model_filename": model_filename,
        "metrics": metrics,
    }
    write_json(payload, path)
    return str(path)
