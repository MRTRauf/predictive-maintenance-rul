from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn


def save_model_artifacts(
    out_dir: Path,
    model,
    metrics: dict,
    feature_cols: list[str],
    subset: str,
    window: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    feature_cols_path = out_dir / "feature_cols.json"
    metadata_path = out_dir / "run_metadata.json"

    joblib.dump(model, model_path)

    _write_json(metrics_path, metrics)
    _write_json(feature_cols_path, feature_cols)

    metadata = {
        "subset": subset,
        "window": window,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "train_rows": metrics.get("train_rows"),
        "valid_rows": metrics.get("valid_rows"),
        "train_units": metrics.get("train_units"),
        "valid_units": metrics.get("valid_units"),
    }
    _write_json(metadata_path, metadata)

    return {
        "model": str(model_path),
        "metrics": str(metrics_path),
        "feature_cols": str(feature_cols_path),
        "run_metadata": str(metadata_path),
    }


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
