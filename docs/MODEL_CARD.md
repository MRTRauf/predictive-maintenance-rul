# Model Card

## Objective
Predict remaining useful life for CMAPSS units to support maintenance triage.

## Data
CMAPSS FD001 with unit-based train/validation split.

## Features
Rolling mean, std, min, max, and slope for sensors s1 to s21 with window size 30, plus cycle and normalized cycle.

## Model
HistGradientBoostingRegressor baseline.

## Metrics
- MAE: 5.221675425457838
- RMSE: 12.484389865147959
- R2: 0.9698077417559168

## Risk buckets
High if predicted RUL <= 30, Medium if <= 80, else Low.

## Notes
- train_frac: 0.8
- seed: 42

## Limitations and next steps
CMAPSS is a benchmark proxy and may not reflect real operational distributions. Validate on in-domain data, monitor drift, and consider calibration for decision thresholds.
