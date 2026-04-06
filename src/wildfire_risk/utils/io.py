from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_json(data: dict, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
