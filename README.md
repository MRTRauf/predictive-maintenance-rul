# Predictive Maintenance RUL

Predictive maintenance remaining useful life modeling using the NASA CMAPSS dataset.

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

## Data placement

Place the NASA CMAPSS archive at `data/raw/CMAPSSData.zip`.

## CLI health check

After `pip install -e .`, run:

```powershell
pm-rul health
```

## Next step

Run ingestion with either command:

```powershell
python -m pm_rul ingest-cmapss --subset FD001
```

```powershell
python scripts/ingest_cmapss.py --subset FD001
```

## Baseline training

Train a baseline model with unit-based split:

```powershell
python -m pm_rul train-baseline --subset FD001 --window 30
```

Artifacts are written to `artifacts/baseline_fd001_w30` and include:
- model.joblib
- metrics.json
- feature_cols.json
- run_metadata.json
- MODEL_CARD.md

Exports from the dashboard are written to `artifacts/baseline_fd001_w30/exports/alerts_FD001.csv`.

CMAPSS is a benchmark proxy dataset and not a production distribution.

## Dashboard

Run the Streamlit dashboard:

```powershell
python -m streamlit run src/pm_rul/dashboard.py
```

It reads artifacts from `artifacts/baseline_fd001_w30` by default.

## Quality

CI runs unit tests on Python 3.12.

```powershell
python -m pytest -q
```
