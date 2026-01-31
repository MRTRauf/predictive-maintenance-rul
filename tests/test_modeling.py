import numpy as np

from pm_rul.features import make_window_features
from pm_rul.modeling import split_by_unit, train_baseline_regressor

from tests.helpers import make_toy_cmapss_df


def test_train_baseline_regressor_metrics():
    df = make_toy_cmapss_df(units=12, cycles=30, seed=3)
    feats = make_window_features(df, window=15)
    feature_cols = [c for c in feats.columns if c not in {"unit_id", "cycle", "rul"}]
    train_df, valid_df = split_by_unit(feats, train_frac=0.8, seed=42)
    results = train_baseline_regressor(train_df, valid_df, feature_cols)
    metrics = results["metrics"]
    for key in ["mae", "rmse", "r2"]:
        assert key in metrics
        assert np.isfinite(metrics[key])
    per_unit = results["per_unit_mae"]
    assert len(per_unit) == valid_df["unit_id"].nunique()
