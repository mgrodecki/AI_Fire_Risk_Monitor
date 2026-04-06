from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PathsConfig:
    data_root: str
    raw_dir: str
    curated_dir: str
    external_dir: str
    artifacts_dir: str
    models_dir: str
    manifests_dir: str


@dataclass
class GridConfig:
    resolution_deg: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass
class TrainingConfig:
    target_column: str
    random_state: int
    model_filename: str
    model_filename_real: str
    prediction_thresholds: dict


@dataclass
class InferenceConfig:
    default_prediction_date: str


@dataclass
class Settings:
    project: dict
    paths: PathsConfig
    grid: GridConfig
    training: TrainingConfig
    inference: InferenceConfig
    real_data: dict
    api_connectors: dict | None = None


def load_settings(path: str | Path = "configs/settings.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(
        project=raw["project"],
        paths=PathsConfig(**raw["paths"]),
        grid=GridConfig(**raw["grid"]),
        training=TrainingConfig(**raw["training"]),
        inference=InferenceConfig(**raw["inference"]),
        real_data=raw.get("real_data", {}),
        api_connectors=raw.get("api_connectors", {}),
    )
