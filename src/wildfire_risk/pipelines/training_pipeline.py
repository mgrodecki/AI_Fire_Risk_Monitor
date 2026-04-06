from __future__ import annotations

from pathlib import Path

from wildfire_risk.config import load_settings
from wildfire_risk.utils.io import read_csv, ensure_dir
from wildfire_risk.utils.logging_utils import get_logger
from wildfire_risk.modeling.train import train_model, save_model
from wildfire_risk.modeling.registry import write_model_manifest

logger = get_logger(__name__)


def run_training() -> None:
    settings = load_settings()
    curated_dir = Path(settings.paths.curated_dir)
    models_dir = ensure_dir(settings.paths.models_dir)
    manifests_dir = ensure_dir(settings.paths.manifests_dir)

    training_path = curated_dir / "training_table.csv"
    if not training_path.exists():
        raise FileNotFoundError(f"Training table not found: {training_path}")

    logger.info("Loading training data from %s", training_path)
    df = read_csv(training_path)

    logger.info("Training model")
    model, metrics = train_model(
        df=df,
        target_col=settings.training.target_column,
        random_state=settings.training.random_state,
    )

    model_path = Path(models_dir) / settings.training.model_filename
    save_model(model, str(model_path))
    manifest_path = write_model_manifest(metrics, str(manifests_dir), settings.training.model_filename)

    logger.info("Saved model to %s", model_path)
    logger.info("Wrote manifest to %s", manifest_path)
    logger.info("Metrics: %s", metrics)
