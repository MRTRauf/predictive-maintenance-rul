from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def global_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype="float64")
    else:
        if not hasattr(model, "pi_X_") or not hasattr(model, "pi_y_"):
            raise ValueError("Permutation importance requires model.pi_X_ and model.pi_y_ samples")
        result = permutation_importance(
            model,
            model.pi_X_,
            model.pi_y_,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importances = np.asarray(result.importances_mean, dtype="float64")

    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def unit_driver_scores(unit_pred_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    values = unit_pred_df[feature_cols]
    means = values.mean()
    stds = values.std(ddof=0).replace(0.0, np.nan)
    last_row = values.iloc[-1]
    z_scores = (last_row - means) / stds
    z_scores = z_scores.fillna(0.0)

    drivers = pd.DataFrame(
        {"feature": z_scores.index.tolist(), "z_score": z_scores.values.astype("float64")}
    )
    drivers["abs_score"] = drivers["z_score"].abs()
    drivers = drivers.sort_values("abs_score", ascending=False).head(10)
    drivers = drivers.drop(columns=["abs_score"]).reset_index(drop=True)
    return drivers
