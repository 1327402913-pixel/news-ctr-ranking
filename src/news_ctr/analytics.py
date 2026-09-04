"""Reproducible DuckDB product analytics for ranking datasets."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import duckdb
import pandas as pd

from news_ctr.data import DatasetBundle, expand_candidates, load_bundle
from news_ctr.evidence import classify_dataset_evidence, evidence_banner
from news_ctr.reporting import dataset_fingerprint

SQL_OUTPUTS = (
    "data_quality.sql",
    "kpi_summary.sql",
    "exposure_funnel.sql",
    "segment_kpis.sql",
    "user_cohorts.sql",
)


@dataclass(frozen=True)
class AnalyticsConfig:
    """Inputs and support threshold for the SQL analytics product."""

    data: Path
    output: Path
    split: str = "validation"
    min_cell_count: int = 5


def _validate_analytics_config(config: AnalyticsConfig) -> None:
    if not config.split.strip():
        raise ValueError("analytics split must not be empty")
    if config.min_cell_count < 1:
        raise ValueError("min_cell_count must be positive")
    if Path(config.output).exists():
        raise ValueError(f"analytics output already exists: {config.output}")


def _history_lengths(bundle: DatasetBundle) -> pd.DataFrame:
    history = bundle.history[["user_id", "article_id_fixed"]].copy()
    history["history_length"] = history["article_id_fixed"].map(len)
    return history.groupby("user_id", as_index=False)["history_length"].max()


def _prepare_analytics_frame(bundle: DatasetBundle) -> pd.DataFrame:
    candidates = expand_candidates(bundle.behaviors)
    article_columns = ["article_id", "published_time", "category", "premium"]
    candidates = candidates.merge(
        bundle.articles[article_columns], on="article_id", how="left", validate="many_to_one"
    )
    candidates = candidates.merge(
        _history_lengths(bundle), on="user_id", how="left", validate="many_to_one"
    )
    candidates["history_length"] = candidates["history_length"].fillna(0).astype(int)
    candidates["impression_time"] = pd.to_datetime(candidates["impression_time"])
    candidates["published_time"] = pd.to_datetime(candidates["published_time"])
    candidates["article_age_days"] = (
        candidates["impression_time"] - candidates["published_time"]
    ).dt.total_seconds() / 86_400
    if "device_type" not in candidates:
        candidates["device_type"] = "unknown"
    return candidates


def _execute_sql_query(connection: duckdb.DuckDBPyConnection, sql_name: str) -> pd.DataFrame:
    sql = files("news_ctr").joinpath("sql", sql_name).read_text(encoding="utf-8")
    return connection.execute(sql).fetchdf()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10f")


def _execute_sql_outputs(candidates: pd.DataFrame, output: Path, config: AnalyticsConfig) -> None:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("candidates", candidates)
        for sql_name in SQL_OUTPUTS:
            result = _execute_sql_query(connection, sql_name)
            if sql_name == "segment_kpis.sql":
                result["low_support"] = result["units"] < config.min_cell_count
            _write_csv(result, output / sql_name.replace(".sql", ".csv"))
    finally:
        connection.close()


def _write_analytics_report(
    output: Path,
    bundle: DatasetBundle,
    config: AnalyticsConfig,
    fingerprint: str,
) -> None:
    summary = pd.read_csv(output / "kpi_summary.csv").iloc[0]
    segments = pd.read_csv(output / "segment_kpis.csv")
    low_support = int(segments["low_support"].sum())
    tier = classify_dataset_evidence(bundle.source_name)
    report = f"""# Product Analytics Report

> {evidence_banner(tier)}

## Scope

- Dataset split: `{bundle.source_name}`
- Dataset fingerprint: `{fingerprint}`
- Impressions: {int(summary["impressions"]):,}
- Candidate exposures: {int(summary["candidates"]):,}
- Clicks: {int(summary["clicks"]):,}
- Candidate CTR: {float(summary["ctr"]):.4f}

## Denominator contract

CTR uses candidate exposures as its denominator. `units` in segment outputs is the
number of distinct impressions represented by the cell; `candidates` is the exposure
denominator and `clicks` is the numerator. Cells with fewer than
{config.min_cell_count} impression units are marked `low_support` and should not drive
a product decision. This run contains {low_support} such cells.

## Deliverables

- `data_quality.csv`: contract checks and anomaly counts.
- `kpi_summary.csv`: top-line traffic and engagement measures.
- `exposure_funnel.csv`: users → impressions → candidates → clicks.
- `segment_kpis.csv`: device, position, slate-size, history, and freshness cuts.
- `user_cohorts.csv`: weekly retention when at least two activity weeks exist.

## Interpretation boundary

These tables describe observed exposure and click logs. They do not identify a causal
treatment effect; use the randomized-experiment workflow for launch decisions.
"""
    (output / "analytics_report.md").write_text(report, encoding="utf-8")


def run_analytics(config: AnalyticsConfig) -> Path:
    """Execute reviewed SQL and publish the complete analytics directory atomically."""

    _validate_analytics_config(config)
    data = Path(config.data)
    output = Path(config.output)
    bundle = load_bundle(data, split=config.split)
    candidates = _prepare_analytics_frame(bundle)
    fingerprint = dataset_fingerprint(data)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging-"
    ) as temporary_directory:
        staging = Path(temporary_directory)
        _execute_sql_outputs(candidates, staging, config)
        _write_analytics_report(staging, bundle, config, fingerprint)
        os.replace(staging, output)
    return output
