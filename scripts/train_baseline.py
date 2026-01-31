from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from pm_rul.artifacts import save_model_artifacts
from pm_rul.features import make_window_features
from pm_rul.modeling import split_by_unit, train_baseline_regressor
from pm_rul.model_card import write_model_card


app = typer.Typer(add_completion=False)


@app.command()
def main(
    subset: str = "FD001",
    window: int = 30,
    train_frac: float = 0.8,
    seed: int = 42,
    out: str = "artifacts/baseline_fd001_w30",
) -> None:
    try:
        project_root = Path.cwd()
        data_path = project_root / "data" / "processed" / "cmapss" / f"train_{subset}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing processed data at {data_path}")

        raw = pd.read_parquet(data_path)
        features = make_window_features(raw, window=window)
        feature_cols = [c for c in features.columns if c not in {"unit_id", "cycle", "rul"}]

        train_df, valid_df = split_by_unit(features, train_frac=train_frac, seed=seed)
        results = train_baseline_regressor(train_df, valid_df, feature_cols)
        model = results["model"]
        sample_size = min(1000, train_df.shape[0])
        model.pi_X_ = train_df[feature_cols].sample(n=sample_size, random_state=seed)
        model.pi_y_ = train_df["rul"].loc[model.pi_X_.index]

        out_dir = Path(out)
        paths = save_model_artifacts(
            out_dir=out_dir,
            model=model,
            metrics=results["metrics"],
            feature_cols=feature_cols,
            subset=subset,
            window=window,
        )
        write_model_card(
            out_dir=out_dir,
            subset=subset,
            window=window,
            metrics=results["metrics"],
            notes={"train_frac": train_frac, "seed": seed},
        )

        _print_summary(results["metrics"], paths)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise SystemExit(1)


def _print_summary(metrics: dict, paths: dict) -> None:
    console = Console()
    table = Table(show_header=False)
    table.add_row("MAE", f"{metrics['mae']:.4f}")
    table.add_row("RMSE", f"{metrics['rmse']:.4f}")
    table.add_row("R2", f"{metrics['r2']:.4f}")
    table.add_row("train rows", str(metrics["train_rows"]))
    table.add_row("valid rows", str(metrics["valid_rows"]))
    table.add_row("train units", str(metrics["train_units"]))
    table.add_row("valid units", str(metrics["valid_units"]))
    worst_units = ", ".join(str(item["unit_id"]) for item in metrics["worst_units"])
    table.add_row("worst units", worst_units)
    table.add_row("model", paths["model"])
    table.add_row("metrics", paths["metrics"])
    table.add_row("feature cols", paths["feature_cols"])
    table.add_row("run metadata", paths["run_metadata"])
    console.print(table)


if __name__ == "__main__":
    app()
