from pathlib import Path

import typer

from pm_rul.cmapss_ingest import ingest_cmapss


app = typer.Typer(add_completion=False)


@app.command()
def main(subset: str = "FD001", force: bool = False) -> None:
    try:
        ingest_cmapss(subset=subset, project_root=Path.cwd(), force=force)
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    app()
