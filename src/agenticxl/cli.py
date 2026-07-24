from __future__ import annotations

import asyncio
from pathlib import Path

import click
from dotenv import load_dotenv

from agenticxl.pipeline import run as run_pipeline


@click.group()
def cli():
    """AgenticXL: filter, consolidate, and enrich Excel trade-data exports."""


@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config/run_config.yaml",
    show_default=True,
    help="Path to the run config YAML file.",
)
@click.option(
    "--project-root",
    "project_root",
    default=None,
    help="Project root to resolve relative paths against "
    "(defaults to the config file's grandparent directory).",
)
@click.option(
    "--env-file",
    "env_file",
    default=".env",
    show_default=True,
    help="Path to a .env file to load before running.",
)
@click.option(
    "--skip-enrichment",
    is_flag=True,
    default=False,
    help="Stop after Stage 2 (consolidation) -- skip Stage 3 column enrichment "
    "entirely (no enrichment LLM call, no duplicate output file). "
    "Same effect as setting AGENTICXL_SKIP_ENRICHMENT=true.",
)
def run(config_path: str, project_root: str | None, env_file: str, skip_enrichment: bool):
    """Run the full filter -> consolidate -> enrich pipeline."""
    if Path(env_file).exists():
        load_dotenv(env_file)

    report = asyncio.run(
        run_pipeline(config_path, project_root=project_root, skip_enrichment=skip_enrichment or None)
    )

    click.echo(
        f"Done: {len(report.files_ok)}/{len(report.file_results)} files ok, "
        f"{report.total_rows_kept}/{report.total_rows_in} rows kept."
    )
    if report.enrichment_skipped:
        click.echo("Enrichment skipped -- consolidated.xlsx is the final output for this run.")
    if report.files_error:
        click.echo(
            f"Errors in {len(report.files_error)} file(s) -- see the run report for details.",
            err=True,
        )


if __name__ == "__main__":
    cli()
