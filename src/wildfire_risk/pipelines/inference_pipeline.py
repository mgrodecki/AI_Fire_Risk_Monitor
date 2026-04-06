from __future__ import annotations

from pathlib import Path
import pandas as pd
from joblib import load

from wildfire_risk.config import load_settings
from wildfire_risk.utils.io import read_csv, write_csv, ensure_dir
from wildfire_risk.utils.logging_utils import get_logger
from wildfire_risk.modeling.predict import predict_scores

logger = get_logger(__name__)


def run_inference(prediction_date: str) -> str:
    settings = load_settings()
    curated_dir = Path(settings.paths.curated_dir)
    models_dir = Path(settings.paths.models_dir)
    predictions_dir = ensure_dir(curated_dir / "daily_predictions")

    model_path = models_dir / settings.training.model_filename
    features_path = curated_dir / "training_table.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features table not found: {features_path}")

    logger.info("Loading model from %s", model_path)
    model = load(model_path)

    logger.info("Loading features from %s", features_path)
    df = read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])

    pred_date = pd.to_datetime(prediction_date)
    inference_df = df[df["date"] == pred_date].copy()
    if inference_df.empty:
        raise ValueError(f"No rows found for prediction date {prediction_date}")

    logger.info("Running inference for %s with %d rows", prediction_date, len(inference_df))
    preds = predict_scores(model, inference_df, settings.training.prediction_thresholds)

    output_path = predictions_dir / f"predictions_{pred_date.date()}.csv"
    write_csv(preds, output_path)
    logger.info("Saved predictions to %s", output_path)
    return str(output_path)
