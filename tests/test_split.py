from pm_rul.modeling import split_by_unit

from tests.helpers import make_toy_cmapss_df


def test_split_by_unit_disjoint():
    df = make_toy_cmapss_df(units=10, cycles=5, seed=11)
    train_df, valid_df = split_by_unit(df, train_frac=0.8, seed=42)
    train_units = set(train_df["unit_id"].unique())
    valid_units = set(valid_df["unit_id"].unique())
    assert train_units.isdisjoint(valid_units)
    assert train_units.union(valid_units) == set(df["unit_id"].unique())
