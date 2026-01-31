from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def make_window_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    required = {"unit_id", "cycle", "rul"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    sensor_cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    expected_sensors = [f"s{i}" for i in range(1, 22)]
    if sensor_cols != expected_sensors:
        raise ValueError("Expected sensor columns s1..s21 in order")

    ordered = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    grouped = ordered.groupby("unit_id", sort=False)

    features = pd.DataFrame(
        {
            "unit_id": ordered["unit_id"].astype("int64"),
            "cycle": ordered["cycle"].astype("int64"),
            "rul": ordered["rul"].astype("int64"),
        }
    )

    max_cycle = grouped["cycle"].transform("max").astype("float64")
    features["cycle_norm"] = ordered["cycle"].astype("float64") / max_cycle
    features["cycle_current"] = ordered["cycle"].astype("float64")

    for col in sensor_cols:
        roll = grouped[col].rolling(window=window, min_periods=1)
        features[f"{col}_mean_w{window}"] = roll.mean().reset_index(level=0, drop=True)
        features[f"{col}_std_w{window}"] = roll.std(ddof=0).reset_index(level=0, drop=True)
        features[f"{col}_min_w{window}"] = roll.min().reset_index(level=0, drop=True)
        features[f"{col}_max_w{window}"] = roll.max().reset_index(level=0, drop=True)
        features[f"{col}_slope_w{window}"] = roll.apply(_rolling_slope, raw=True).reset_index(level=0, drop=True)

    feature_cols = [c for c in features.columns if c not in {"unit_id", "cycle", "rul"}]
    if features[feature_cols].isna().any().any():
        raise ValueError("NaNs detected in computed feature columns")

    return features


def _rolling_slope(values: Iterable[float]) -> float:
    arr = np.asarray(values, dtype="float64")
    n = arr.size
    if n <= 1:
        return 0.0
    x = np.arange(n, dtype="float64")
    x_mean = x.mean()
    y_mean = arr.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0.0:
        return 0.0
    num = ((x - x_mean) * (arr - y_mean)).sum()
    return float(num / denom)
