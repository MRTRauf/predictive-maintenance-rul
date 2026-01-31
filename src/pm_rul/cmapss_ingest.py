from pathlib import Path
import json
import shutil
import zipfile

import pandas as pd
from rich.console import Console
from rich.table import Table


def ingest_cmapss(subset: str, project_root: Path | None = None, force: bool = False) -> dict:
    root = _resolve_project_root(project_root)
    raw_zip, interim_dir, processed_dir = _build_paths(root)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_zip.exists():
        raise FileNotFoundError(f"Expected CMAPSSData.zip at {raw_zip}")

    extract_root = interim_dir / "CMAPSSData"
    if force and extract_root.exists():
        shutil.rmtree(extract_root)

    if not extract_root.exists():
        with zipfile.ZipFile(raw_zip, "r") as zf:
            zf.extractall(extract_root)

    train_path = _find_unique_file(extract_root, f"train_{subset}.txt")
    test_path = _find_unique_file(extract_root, f"test_{subset}.txt")
    rul_path = _find_unique_file(extract_root, f"RUL_{subset}.txt")

    train = _load_cmapss_frame(train_path)
    test = _load_cmapss_frame(test_path)

    train = _attach_train_rul(train)
    test, rul_last_table = _attach_test_rul(test, rul_path)

    _validate_rul(train["rul"])
    _validate_rul(test["rul"])

    train_out = processed_dir / f"train_{subset}.parquet"
    test_out = processed_dir / f"test_{subset}.parquet"
    rul_out = processed_dir / f"rul_{subset}.csv"
    schema_out = processed_dir / f"schema_{subset}.json"

    _write_parquet(train, train_out)
    _write_parquet(test, test_out)
    rul_last_table.to_csv(rul_out, index=False)
    _write_schema(train, schema_out)

    stats = _collect_stats(train, test)
    _print_summary(subset, stats)

    return {
        "subset": subset,
        "paths": {
            "train_parquet": str(train_out),
            "test_parquet": str(test_out),
            "rul_csv": str(rul_out),
            "schema_json": str(schema_out),
        },
        "stats": stats,
    }


def _resolve_project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root
    return Path(__file__).resolve().parents[2]


def _build_paths(root: Path) -> tuple[Path, Path, Path]:
    raw_zip = root / "data" / "raw" / "CMAPSSData.zip"
    interim_dir = root / "data" / "interim" / "cmapss"
    processed_dir = root / "data" / "processed" / "cmapss"
    return raw_zip, interim_dir, processed_dir


def _find_unique_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match for {name}, found {len(matches)}")
    return matches[0]


def _load_cmapss_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 26:
        raise ValueError(f"Expected 26 columns in {path}, found {df.shape[1]}")

    columns = ["unit_id", "cycle", "setting1", "setting2", "setting3"]
    columns += [f"s{i}" for i in range(1, 22)]
    df.columns = columns

    if df["unit_id"].isna().any() or df["cycle"].isna().any():
        raise ValueError(f"Missing values found in unit_id or cycle for {path}")

    df["unit_id"] = df["unit_id"].astype("int64")
    df["cycle"] = df["cycle"].astype("int64")

    float_cols = [c for c in df.columns if c not in {"unit_id", "cycle"}]
    df[float_cols] = df[float_cols].astype("float64")

    if (df["unit_id"] < 1).any():
        raise ValueError(f"unit_id must be >= 1 in {path}")
    if (df["cycle"] < 1).any():
        raise ValueError(f"cycle must be >= 1 in {path}")

    return df


def _attach_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train.groupby("unit_id")["cycle"].transform("max")
    train = train.copy()
    train["rul"] = (max_cycle - train["cycle"]).astype("int64")
    return train


def _attach_test_rul(test: pd.DataFrame, rul_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rul_vec = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")
    rul_last_values = rul_vec.values.flatten().astype("int64")

    unit_ids = sorted(test["unit_id"].unique())
    if len(rul_last_values) != len(unit_ids):
        raise ValueError(
            f"RUL length {len(rul_last_values)} does not match test units {len(unit_ids)}"
        )

    rul_last_map = dict(zip(unit_ids, rul_last_values))
    last_cycle = test.groupby("unit_id")["cycle"].max()

    test = test.copy()
    test["rul_last"] = test["unit_id"].map(rul_last_map).astype("int64")
    test["last_cycle"] = test["unit_id"].map(last_cycle).astype("int64")
    test["rul"] = (test["rul_last"] + (test["last_cycle"] - test["cycle"])).astype("int64")
    test = test.drop(columns=["rul_last", "last_cycle"])

    rul_last_table = pd.DataFrame({"unit_id": unit_ids, "rul_last": rul_last_values})
    rul_last_table["unit_id"] = rul_last_table["unit_id"].astype("int64")
    rul_last_table["rul_last"] = rul_last_table["rul_last"].astype("int64")

    return test, rul_last_table


def _validate_rul(series: pd.Series) -> None:
    if series.isna().any():
        raise ValueError("RUL contains missing values")
    if (series < 0).any():
        raise ValueError("RUL contains negative values")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        import pyarrow
        df.to_parquet(path, engine="pyarrow", index=False)
    except Exception as exc:
        if not isinstance(exc, ModuleNotFoundError):
            raise
        import duckdb
        con = duckdb.connect(database=":memory:")
        con.register("df", df)
        con.execute(f"COPY df TO '{path.as_posix()}' (FORMAT 'parquet')")
        con.close()


def _write_schema(df: pd.DataFrame, path: Path) -> None:
    descriptions = {
        "unit_id": "Engine unit identifier",
        "cycle": "Operational cycle index",
        "setting1": "Operational setting 1",
        "setting2": "Operational setting 2",
        "setting3": "Operational setting 3",
    }
    for i in range(1, 22):
        descriptions[f"s{i}"] = f"Sensor measurement {i}"
    descriptions["rul"] = "Remaining useful life in cycles"

    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "descriptions": {col: descriptions[col] for col in df.columns},
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def _collect_stats(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cycle_min = int(min(train["cycle"].min(), test["cycle"].min()))
    cycle_max = int(max(train["cycle"].max(), test["cycle"].max()))

    train_rul = train["rul"]
    test_rul = test["rul"]

    return {
        "train_rows": int(train.shape[0]),
        "train_units": int(train["unit_id"].nunique()),
        "test_rows": int(test.shape[0]),
        "test_units": int(test["unit_id"].nunique()),
        "cycle_min": cycle_min,
        "cycle_max": cycle_max,
        "train_rul_min": int(train_rul.min()),
        "train_rul_median": int(train_rul.median()),
        "train_rul_max": int(train_rul.max()),
        "test_rul_min": int(test_rul.min()),
        "test_rul_median": int(test_rul.median()),
        "test_rul_max": int(test_rul.max()),
    }


def _print_summary(subset: str, stats: dict) -> None:
    console = Console()
    table = Table(show_header=False)
    table.add_row("subset", subset)
    table.add_row("train rows", str(stats["train_rows"]))
    table.add_row("train units", str(stats["train_units"]))
    table.add_row("test rows", str(stats["test_rows"]))
    table.add_row("test units", str(stats["test_units"]))
    table.add_row("cycle min/max", f"{stats['cycle_min']} / {stats['cycle_max']}")
    table.add_row(
        "train RUL min/med/max",
        f"{stats['train_rul_min']} / {stats['train_rul_median']} / {stats['train_rul_max']}",
    )
    table.add_row(
        "test RUL min/med/max",
        f"{stats['test_rul_min']} / {stats['test_rul_median']} / {stats['test_rul_max']}",
    )
    console.print(table)
