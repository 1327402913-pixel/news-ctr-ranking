"""Reproducible experiment artifacts and model-card rendering."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def dataset_fingerprint(path: str | Path) -> str:
    """Fingerprint relative paths, sizes, and complete file contents."""

    root = Path(path)
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    files = sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix in {".parquet", ".json"}
    )
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(root)).encode())
        digest.update(str(stat.st_size).encode())
        with item.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    return digest.hexdigest()


def write_run_artifacts(
    output: str | Path,
    *,
    dataset: str,
    dataset_summary: Mapping[str, Any],
    model_kind: str,
    metrics: Mapping[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    run_config: Mapping[str, Any],
) -> Path:
    """Write the complete reproducibility record for one experiment."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "dataset": dataset,
        "model": model_kind,
        "metrics": dict(metrics),
        "split_summary": dict(dataset_summary),
    }
    _write_json(output_path / "metrics.json", metrics_payload)
    _write_json(output_path / "run_config.json", dict(run_config))
    expected_prediction_columns = ["impression_id", "article_id", "label", "score"]
    if predictions.columns.tolist() != expected_prediction_columns:
        raise ValueError(
            "predictions must have columns in this order: " + ", ".join(expected_prediction_columns)
        )
    predictions.to_parquet(output_path / "predictions.parquet", index=False)
    importance.to_csv(output_path / "feature_importance.csv", index=False)
    (output_path / "model_card.md").write_text(
        render_model_card(
            dataset=dataset,
            model_kind=model_kind,
            metrics=metrics,
            run_config=run_config,
        ),
        encoding="utf-8",
    )
    return output_path


def render_model_card(
    *,
    dataset: str,
    model_kind: str,
    metrics: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> str:
    """Render a candid model card that distinguishes synthetic and real evaluation."""

    is_synthetic = dataset.startswith("synthetic")
    evaluation_note = (
        "This is an engineering smoke test on deterministic synthetic data. "
        "It is not an EB-NeRD benchmark and must not be used to claim real-world quality."
        if is_synthetic
        else "This run uses an EB-NeRD validation split obtained under the dataset's license."
    )
    metric_lines = "\n".join(
        f"| `{name}` | {float(value):.6f} |"
        for name, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and name not in {"groups", "auc_groups", "skipped_auc_groups"}
    )
    return f"""# Model Card

## Model

- Estimator: `{model_kind}`
- Dataset split: `{dataset}`
- Random seed: `{run_config.get("seed")}`
- Dataset fingerprint: `{run_config.get("dataset_fingerprint")}`

## Intended use

Rank candidate news articles within an observed impression for offline research
and portfolio demonstration. The model is not intended for production editorial
decisions.

## Evaluation status

{evaluation_note}

| Metric | Value |
| --- | ---: |
{metric_lines}

## Important limitations

- Offline clicks reflect exposure and position bias, not pure user preference.
- No causal claim can be made from these observational logs.
- The baseline does not optimize diversity, novelty, fairness, or editorial values.
- User-history representations are fitted from the supplied training history only.

## Reproduce

```bash
news-ctr train \\
  --data {run_config.get("data")} \\
  --output {run_config.get("output")} \\
  --model {model_kind} \\
  --seed {run_config.get("seed")} \\
  --text-components {run_config.get("text_components")}
```
"""


def render_benchmark_report(
    *,
    dataset: str,
    leaderboard: pd.DataFrame,
    intervals: pd.DataFrame,
    ablations: pd.DataFrame,
    segments: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping[str, Any],
) -> str:
    """Render a self-contained, recruiter-readable benchmark evidence report."""

    status = (
        "Synthetic engineering evidence — this is not an EB-NeRD benchmark and does not "
        "measure real-world recommendation lift."
        if dataset.startswith("synthetic")
        else "Evaluation on an EB-NeRD split obtained under the dataset license."
    )
    ranked = leaderboard.sort_values(
        ["ndcg@10", "model"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    leaderboard_table = _markdown_table(
        ["Rank", "Model", "Group AUC", "MRR", "NDCG@5", "NDCG@10"],
        [
            [
                index + 1,
                row["model"],
                row["group_auc"],
                row["mrr"],
                row["ndcg@5"],
                row["ndcg@10"],
            ]
            for index, row in ranked.iterrows()
        ],
    )
    interval_table = _markdown_table(
        ["Model", "Metric", "Mean", "95% lower", "95% upper", "Valid samples"],
        [
            [
                row["model"],
                row["metric"],
                row["mean"],
                row["lower"],
                row["upper"],
                f"{int(row['valid_samples'])}/{int(row['requested_samples'])}",
            ]
            for _, row in intervals.iterrows()
        ],
    )

    full_rows = ablations.loc[ablations["ablation"] == "full"]
    full = full_rows.iloc[0] if not full_rows.empty else None
    ablation_table = _markdown_table(
        [
            "Ablation",
            "Features",
            "Group AUC",
            "Δ vs full",
            "MRR",
            "Δ vs full",
            "NDCG@5",
            "Δ vs full",
            "NDCG@10",
            "Δ vs full",
        ],
        [
            [
                row["ablation"],
                row["feature_count"],
                row["group_auc"],
                None if full is None else row["group_auc"] - full["group_auc"],
                row["mrr"],
                None if full is None else row["mrr"] - full["mrr"],
                row["ndcg@5"],
                None if full is None else row["ndcg@5"] - full["ndcg@5"],
                row["ndcg@10"],
                None if full is None else row["ndcg@10"] - full["ndcg@10"],
            ]
            for _, row in ablations.iterrows()
        ],
    )
    segment_table = _markdown_table(
        ["Segment", "Value", "Impressions", "Support", "Group AUC", "MRR", "NDCG@10"],
        [
            [
                row["segment"],
                row["value"],
                row["impressions"],
                "LOW" if row["low_support"] else "OK",
                row["group_auc"],
                row["mrr"],
                row["ndcg@10"],
            ]
            for _, row in segments.iterrows()
        ],
    )
    diagnostic_table = _markdown_table(
        ["Diagnostic", "Bucket", "Candidates", "Clicks", "Click rate"],
        [
            [
                row["diagnostic"],
                row["bucket"],
                row["candidates"],
                row["clicks"],
                row["click_rate"],
            ]
            for _, row in diagnostics.iterrows()
        ],
    )
    operations_table = _markdown_table(
        ["Model", "Fit seconds", "Score ms / 1k", "Serialized bytes"],
        [
            [
                row["model"],
                row["fit_seconds"],
                row["score_ms_per_1000"],
                row["model_bytes"],
            ]
            for _, row in leaderboard.iterrows()
        ],
    )
    models = ",".join(str(item) for item in config.get("models", []))
    synthetic_generation = config.get("synthetic_generation", {})
    synthetic_setup = ""
    if dataset.startswith("synthetic"):
        synthetic_setup = (
            f"news-ctr make-synthetic --output {config.get('data')} "
            f"--seed {synthetic_generation.get('seed', config.get('seed'))} "
            f"--users {synthetic_generation.get('users', 30)} "
            f"--articles {synthetic_generation.get('articles', 80)} "
            f"--impressions {synthetic_generation.get('impressions', 180)}\n\n"
        )
    segment_models = (
        ", ".join(str(item) for item in pd.unique(segments["model"]))
        if "model" in segments
        else "unspecified"
    )
    reproduction_output = config.get("reproduction_output") or (
        f"{config.get('output')}-reproduced"
    )
    return f"""# Ranking Benchmark Report

> **Evaluation status:** {status}

## 60-second summary

- Dataset split: `{dataset}`.
- Every model is evaluated on the same validation impressions.
- Popularity statistics and feature fitting use training-visible rows only.
- Uncertainty uses `{config.get("bootstrap_samples")}` paired, impression-level bootstrap samples.
- This is offline observational evidence; it does not establish causal product impact.

## Leaderboard

{leaderboard_table}

## 95% confidence intervals

Whole impressions are resampled, and identical draws are reused for every model.

{interval_table}

## Feature ablations

Deltas compare each deterministic logistic-regression variant with the full feature set.

{ablation_table}

## Segment checks

Analysis model: `{segment_models}`.

Slices marked `LOW` contain fewer than five impressions and should not drive conclusions.

{segment_table}

## Exposure-bias diagnostic

These are candidate-level empirical click rates, not ranking metrics or causal effects.
Position differences are evidence of exposure bias in logged feedback.

{diagnostic_table}

## Runtime and artifact size

Timing is environment-dependent and is included for operational context only.

{operations_table}

## Leakage safeguards

- Temporal ordering keeps validation impressions strictly after training impressions.
- Candidate lists remain intact for ranking metrics and bootstrap resampling.
- Smoothed article popularity is estimated from training labels only.
- Text vocabulary, imputers, encoders, and user representations are fitted
  without validation clicks.

## Limitations

- Logged clicks mix user preference with exposure and position effects.
- Synthetic runs validate engineering behavior, not recommender quality on EB-NeRD.
- Offline metrics do not measure diversity, novelty, fairness, editorial quality, or online lift.
- Segment differences are descriptive and must not be interpreted as causal.
- Joblib artifacts use pickle semantics; load only artifacts you created or trust.

## Reproduce exactly

```bash
{synthetic_setup}news-ctr benchmark \\
  --data {config.get("data")} \\
  --output {reproduction_output} \\
  --models {models} \\
  --bootstrap-samples {config.get("bootstrap_samples")} \\
  --text-components {config.get("text_components")} \\
  --valid-fraction {config.get("valid_fraction")} \\
  --seed {config.get("seed")}
```

The reproduction destination must not already exist. The adjacent CSV files contain
the complete machine-readable evidence.
"""


def _format_report_value(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        numeric = float(value)
        return f"{numeric:.6f}" if math.isfinite(numeric) else "N/A"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_format_report_value(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value
