"""Reproducible experiment artifacts and model-card rendering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def dataset_fingerprint(path: str | Path) -> str:
    """Fingerprint file names, sizes, and modification times without reading huge datasets."""

    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix in {".parquet", ".json"}
    )
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(root)).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
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
