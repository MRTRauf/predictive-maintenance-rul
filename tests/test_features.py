from pm_rul.features import make_window_features

from tests.helpers import make_toy_cmapss_df


def test_make_window_features_basic():
    df = make_toy_cmapss_df(units=3, cycles=20, seed=7)
    feats = make_window_features(df, window=10)
    assert {"unit_id", "cycle", "rul"}.issubset(feats.columns)
    assert "s1_mean_w10" in feats.columns
    assert "s1_std_w10" in feats.columns
    assert "s1_slope_w10" in feats.columns
    feature_cols = [c for c in feats.columns if c not in {"unit_id", "cycle", "rul"}]
    assert not feats[feature_cols].isna().any().any()
    assert len(feats) == len(df)
