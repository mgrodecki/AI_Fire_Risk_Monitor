from __future__ import annotations

import pandas as pd

from wildfire_risk.features.engineering import add_time_features, add_rolling_features
from wildfire_risk.features.labels import make_next_day_labels


def assemble_training_table(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    df = dynamic_df.copy()
    df = add_time_features(df)
    df = add_rolling_features(df)
    labels = make_next_day_labels(df)
    df = df.merge(labels, on=["date", "cell_id"], how="left")
    return df.sort_values(["date", "cell_id"]).reset_index(drop=True)
