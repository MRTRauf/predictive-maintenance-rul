from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_by_unit(df: pd.DataFrame, train_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_ids = np.array(sorted(df["unit_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unit_ids)
    train_size = int(len(unit_ids) * train_frac)
    train_units = set(unit_ids[:train_size])
    train_df = df[df["unit_id"].isin(train_units)].copy()
    valid_df = df[~df["unit_id"].isin(train_units)].copy()
    return train_df, valid_df


def train_baseline_regressor(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rul",
) -> dict:
    model = HistGradientBoostingRegressor(random_state=42)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    mae = float(mean_absolute_error(y_valid, preds))
    mse = mean_squared_error(y_valid, preds)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_valid, preds))

    per_unit = _per_unit_mae(valid_df, y_valid, preds)
    worst = per_unit.sort_values("mae", ascending=False).head(5).to_dict(orient="records")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "worst_units": worst,
        "train_rows": int(train_df.shape[0]),
        "valid_rows": int(valid_df.shape[0]),
        "train_units": int(train_df["unit_id"].nunique()),
        "valid_units": int(valid_df["unit_id"].nunique()),
    }

    return {"model": model, "metrics": metrics, "per_unit_mae": per_unit}


def _per_unit_mae(valid_df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    temp = valid_df[["unit_id"]].copy()
    temp["abs_error"] = np.abs(y_true.values - y_pred)
    per_unit = temp.groupby("unit_id")["abs_error"].mean().reset_index()
    per_unit = per_unit.rename(columns={"abs_error": "mae"})
    return per_unit
