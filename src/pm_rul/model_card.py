from __future__ import annotations

from pathlib import Path


def write_model_card(
    out_dir: Path, subset: str, window: int, metrics: dict, notes: dict
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "MODEL_CARD.md"

    mae = metrics.get("mae")
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")

    note_lines = "\n".join([f"- {k}: {v}" for k, v in notes.items()]) if notes else ""

    content = f"""# Model Card

## Objective
Predict remaining useful life for CMAPSS units to support maintenance triage.

## Data
CMAPSS {subset} with unit-based train/validation split.

## Features
Rolling mean, std, min, max, and slope for sensors s1 to s21 with window size {window}, plus cycle and normalized cycle.

## Model
HistGradientBoostingRegressor baseline.

## Metrics
- MAE: {mae}
- RMSE: {rmse}
- R2: {r2}

## Risk buckets
High if predicted RUL <= 30, Medium if <= 80, else Low.

## Notes
{note_lines if note_lines else "None"}

## Limitations and next steps
CMAPSS is a benchmark proxy and may not reflect real operational distributions. Validate on in-domain data, monitor drift, and consider calibration for decision thresholds.
"""

    path.write_text(content, encoding="utf-8")
    return path
