from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from pm_rul.explain import global_feature_importance, unit_driver_scores
from pm_rul.features import make_window_features


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_path(path: Path) -> None:
    if not path.exists():
        st.error(f"Missing required file: {path}")
        st.stop()


def _load_artifacts(artifact_dir: Path) -> dict:
    model_path = artifact_dir / "model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    feature_cols_path = artifact_dir / "feature_cols.json"
    metadata_path = artifact_dir / "run_metadata.json"

    _require_path(model_path)
    _require_path(metrics_path)
    _require_path(feature_cols_path)
    _require_path(metadata_path)

    model = joblib.load(model_path)
    metrics = _load_json(metrics_path)
    feature_cols = _load_json(feature_cols_path)
    metadata = _load_json(metadata_path)

    return {
        "model": model,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "metadata": metadata,
    }


def _load_dataset(project_root: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    demo_base = project_root / "demo_assets" / "cmapss"
    local_base = project_root / "data" / "processed" / "cmapss"

    demo_train = demo_base / f"train_{subset}.parquet"
    demo_test = demo_base / f"test_{subset}.parquet"
    local_train = local_base / f"train_{subset}.parquet"
    local_test = local_base / f"test_{subset}.parquet"

    if demo_train.exists() and demo_test.exists():
        train_df = pd.read_parquet(demo_train)
        test_df = pd.read_parquet(demo_test)
        return train_df, test_df

    if local_train.exists() and local_test.exists():
        train_df = pd.read_parquet(local_train)
        test_df = pd.read_parquet(local_test)
        return train_df, test_df

    st.error(
        "Missing datasets. Expected demo paths: "
        f"{demo_train} and {demo_test}. "
        "Expected local-generated paths: "
        f"{local_train} and {local_test}. "
        "Generate locally with: python -m pm_rul ingest-cmapss --subset FD001 "
        "then python -m pm_rul train-baseline --subset FD001 --window 30."
    )
    st.stop()


def _dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "units": int(df["unit_id"].nunique()),
        "rows": int(df.shape[0]),
        "cycle_min": int(df["cycle"].min()),
        "cycle_max": int(df["cycle"].max()),
    }


def _fleet_overview(pred_df: pd.DataFrame, subset: str) -> None:
    last_rows = pred_df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)
    last_rows = last_rows.copy()
    last_rows["risk_bucket"] = pd.cut(
        last_rows["predicted_rul"],
        bins=[-float("inf"), 30, 80, float("inf")],
        labels=["High", "Medium", "Low"],
    )
    last_rows = last_rows.sort_values("predicted_rul")

    selected_buckets = st.multiselect(
        "Risk buckets",
        ["High", "Medium", "Low"],
        default=["High", "Medium"],
    )
    if selected_buckets:
        last_rows = last_rows[last_rows["risk_bucket"].isin(selected_buckets)]

    export_path = Path("artifacts") / "baseline_fd001_w30" / "exports" / f"alerts_{subset}.csv"
    if st.button("Export alert list"):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        alert_df = last_rows[["unit_id", "cycle", "predicted_rul", "risk_bucket"]].rename(
            columns={"cycle": "last_cycle"}
        )
        alert_df.to_csv(export_path, index=False)
        st.success(f"Exported alerts to {export_path}")

    st.dataframe(
        last_rows[["unit_id", "cycle", "predicted_rul", "risk_bucket"]],
        use_container_width=True,
    )

    fig = px.bar(
        last_rows,
        x="unit_id",
        y="predicted_rul",
        color="risk_bucket",
        title="Predicted RUL at Last Cycle by Unit",
    )
    st.plotly_chart(fig, use_container_width=True)


def _unit_detail(
    pred_df: pd.DataFrame, feature_cols: list[str], raw_test_df: pd.DataFrame
) -> None:
    unit_ids = sorted(pred_df["unit_id"].unique())
    unit_id = st.selectbox("Unit", unit_ids)
    unit_df = pred_df[pred_df["unit_id"] == unit_id].sort_values("cycle")

    fig = px.line(
        unit_df,
        x="cycle",
        y="predicted_rul",
        title=f"Predicted RUL Over Cycle for Unit {unit_id}",
    )
    st.plotly_chart(fig, use_container_width=True)

    drivers = unit_driver_scores(unit_df, feature_cols)
    drivers["direction"] = drivers["z_score"].apply(
        lambda v: "higher than unit mean" if v > 0 else "lower than unit mean" if v < 0 else "at unit mean"
    )
    st.subheader("Top drivers proxy")
    st.dataframe(drivers[["feature", "z_score", "direction"]], use_container_width=True)

    raw_unit = raw_test_df[raw_test_df["unit_id"] == unit_id].sort_values("cycle")
    raw_last = raw_unit.iloc[[-1]]
    sensor_cols = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    existing = [c for c in sensor_cols if c in raw_last.columns]
    st.subheader("Last cycle sensor snapshot")
    if not existing:
        st.warning(f"Expected sensor columns not found. Available columns: {list(raw_last.columns)}")
    else:
        st.dataframe(raw_last[existing], use_container_width=True)


def _quality_checks(df: pd.DataFrame) -> None:
    key_cols = ["unit_id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    missing = df[key_cols].isna().sum().reset_index()
    missing.columns = ["column", "missing"]
    missing = missing[missing["missing"] > 0]

    violations = []
    if (df["unit_id"] < 1).any():
        violations.append({"rule": "unit_id >= 1", "count": int((df["unit_id"] < 1).sum())})
    if (df["cycle"] < 1).any():
        violations.append({"rule": "cycle >= 1", "count": int((df["cycle"] < 1).sum())})
    violations_df = pd.DataFrame(violations)

    st.subheader("Missingness")
    if missing.empty:
        st.write("No missing values in key columns.")
    else:
        st.dataframe(missing, use_container_width=True)

    st.subheader("Range sanity")
    if violations_df.empty:
        st.write("No violations detected.")
    else:
        st.dataframe(violations_df, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="CMAPSS FD001 RUL Triage", layout="wide")
    st.title("CMAPSS FD001 RUL Triage Dashboard")

    st.sidebar.header("Artifacts")
    subset = st.sidebar.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"], index=0)
    artifact_dir_input = st.sidebar.text_input("Artifact path", "demo_assets/baseline_fd001_w30")
    load_clicked = st.sidebar.button("Load artifacts")

    if "artifacts" not in st.session_state:
        st.session_state["artifacts"] = None

    if load_clicked:
        artifact_dir = Path(artifact_dir_input)
        st.session_state["artifacts"] = _load_artifacts(artifact_dir)

    artifacts = st.session_state["artifacts"]
    if artifacts is None:
        st.info("Load artifacts to begin.")
        return

    window = artifacts["metadata"].get("window")
    st.sidebar.markdown(f"Window: {window}")

    project_root = Path.cwd()
    train_df, test_raw = _load_dataset(project_root, subset)

    train_stats = _dataset_stats(train_df)
    test_stats = _dataset_stats(test_raw)

    st.subheader("Dataset overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(
            f"Train units: {train_stats['units']} | rows: {train_stats['rows']} | "
            f"cycle range: {train_stats['cycle_min']} - {train_stats['cycle_max']}"
        )
    with col2:
        st.write(
            f"Test units: {test_stats['units']} | rows: {test_stats['rows']} | "
            f"cycle range: {test_stats['cycle_min']} - {test_stats['cycle_max']}"
        )

    st.subheader("How to read this")
    st.write(
        "Risk buckets are based on predicted RUL at the last observed cycle for each unit. "
        "High risk is 30 cycles or less, Medium is 31 to 80 cycles, and Low is above 80. "
        "CMAPSS is a benchmark proxy for real asset health, so treat outputs as triage signals."
    )

    test_feat = make_window_features(test_raw, window=int(window))
    feature_cols = artifacts["feature_cols"]
    preds = artifacts["model"].predict(test_feat[feature_cols])
    pred_df = test_feat.copy()
    pred_df["predicted_rul"] = preds

    st.subheader("Global feature importance")
    try:
        importance = global_feature_importance(artifacts["model"], feature_cols).head(20)
        st.dataframe(importance, use_container_width=True)
    except Exception as exc:
        st.warning(str(exc))

    tab1, tab2, tab3 = st.tabs(["Fleet overview", "Unit detail", "Quality"])
    with tab1:
        _fleet_overview(pred_df, subset)
    with tab2:
        _unit_detail(pred_df, feature_cols, test_raw)
    with tab3:
        _quality_checks(test_raw)


if __name__ == "__main__":
    main()
