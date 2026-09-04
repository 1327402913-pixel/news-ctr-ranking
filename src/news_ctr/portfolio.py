"""Atomic assembly of aggregate, public-safe portfolio evidence."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ANALYTICS_ARTIFACTS = (
    "analytics_report.md",
    "data_quality.csv",
    "exposure_funnel.csv",
    "kpi_summary.csv",
    "segment_kpis.csv",
    "user_cohorts.csv",
)

EXPERIMENT_ARTIFACTS = (
    "cuped_effects.csv",
    "decision_report.md",
    "effects.csv",
    "effects.png",
    "experiment_config.json",
    "experiment_summary.csv",
    "heterogeneous_effects.csv",
    "run_manifest.json",
    "srm.json",
)


def _require_artifacts(source: Path, names: tuple[str, ...], label: str) -> None:
    if not source.is_dir():
        raise ValueError(f"{label} artifact directory does not exist: {source}")
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise ValueError(f"{label} artifacts are incomplete: {', '.join(missing)}")


def assemble_portfolio(analytics: Path, experiment: Path, output: Path) -> Path:
    """Copy the declared aggregate evidence set and publish it atomically."""

    analytics = Path(analytics)
    experiment = Path(experiment)
    output = Path(output)
    if output.exists():
        raise ValueError(f"portfolio output already exists: {output}")
    _require_artifacts(analytics, ANALYTICS_ARTIFACTS, "analytics")
    _require_artifacts(experiment, EXPERIMENT_ARTIFACTS, "experiment")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging-"
    ) as temporary_directory:
        staging = Path(temporary_directory)
        for source, names in (
            (analytics, ANALYTICS_ARTIFACTS),
            (experiment, EXPERIMENT_ARTIFACTS),
        ):
            for name in names:
                shutil.copy2(source / name, staging / name)
        os.replace(staging, output)
    return output
