from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import news_ctr.experiments as experiments
from news_ctr.experiment_data import write_synthetic_experiment
from news_ctr.experiments import run_experiment
from news_ctr.portfolio import ANALYTICS_ARTIFACTS, EXPERIMENT_ARTIFACTS, assemble_portfolio


def _fixture(tmp_path: Path, *, users: int = 6_000) -> tuple[Path, Path]:
    experiment = write_synthetic_experiment(tmp_path / "experiment.parquet", seed=42, users=users)
    config = Path(__file__).parents[1] / "configs" / "experiment-v3.json"
    return experiment, config


def test_run_experiment_writes_complete_decision_evidence(tmp_path: Path) -> None:
    experiment, config = _fixture(tmp_path)
    output = run_experiment(experiment, tmp_path / "study", config)

    assert {item.name for item in output.iterdir()} == {
        "cuped_effects.csv",
        "decision_report.md",
        "effects.csv",
        "effects.png",
        "experiment_config.json",
        "experiment_summary.csv",
        "heterogeneous_effects.csv",
        "run_manifest.json",
        "srm.json",
    }
    srm = json.loads((output / "srm.json").read_text(encoding="utf-8"))
    assert srm["passed"] is True
    assert sum(srm["observed_counts"].values()) == 6_000
    assert pd.read_csv(output / "effects.csv")["metric"].tolist() == [
        "click",
        "dwell_seconds",
        "latency_ms",
    ]
    report = (output / "decision_report.md").read_text(encoding="utf-8")
    assert "Synthetic randomized teaching evidence" in report
    assert "not production lift" in report.lower()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input"]["sha256"] == hashlib.sha256(experiment.read_bytes()).hexdigest()
    assert manifest["config"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert (
        manifest["metadata"]["sha256"]
        == hashlib.sha256(experiment.with_suffix(".metadata.json").read_bytes()).hexdigest()
    )
    assert "Input SHA-256" in report
    assert (output / "experiment_config.json").read_bytes() == config.read_bytes()


def test_run_experiment_refuses_existing_output_and_cleans_failed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, config = _fixture(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run_experiment(experiment, existing, config)

    output = tmp_path / "failed"

    def fail_outputs(*args, **kwargs):
        raise RuntimeError("injected workflow failure")

    monkeypatch.setattr(experiments, "_write_experiment_outputs", fail_outputs)
    with pytest.raises(RuntimeError, match="injected workflow failure"):
        run_experiment(experiment, output, config)
    assert not output.exists()


def test_run_experiment_rejects_observational_evidence_tier(tmp_path: Path) -> None:
    experiment, config = _fixture(tmp_path)
    metadata_path = experiment.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["evidence_tier"] = "licensed-ebnerd"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="randomized evidence"):
        run_experiment(experiment, tmp_path / "study", config)


@pytest.mark.parametrize("field", ["randomization_unit", "units", "allocation"])
def test_run_experiment_validates_fixture_metadata(tmp_path: Path, field: str) -> None:
    experiment, config = _fixture(tmp_path)
    metadata_path = experiment.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = "invalid"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        run_experiment(experiment, tmp_path / "study", config)


def test_run_experiment_reports_schema_errors_before_metadata_checks(tmp_path: Path) -> None:
    experiment, config = _fixture(tmp_path)
    pd.read_parquet(experiment).drop(columns="variant").to_parquet(experiment, index=False)

    with pytest.raises(ValueError, match="missing configured columns: variant"):
        run_experiment(experiment, tmp_path / "study", config)


def test_committed_v3_evidence_is_truthful_and_reproducible() -> None:
    root = Path(__file__).parents[1] / "artifacts" / "portfolio-v3"
    assert {item.name for item in root.iterdir()} == {
        *ANALYTICS_ARTIFACTS,
        *EXPERIMENT_ARTIFACTS,
    }
    report = (root / "decision_report.md").read_text(encoding="utf-8")
    assert "synthetic-rct" in report
    assert "not production lift" in report.lower()
    assert "Recommendation" in report
    assert (root / "effects.png").is_file()
    assert (root / "analytics_report.md").is_file()
    assert (root / "kpi_summary.csv").is_file()


def test_portfolio_assembly_is_complete_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analytics = tmp_path / "analytics"
    experiment = tmp_path / "experiment"
    analytics.mkdir()
    experiment.mkdir()
    for name in ANALYTICS_ARTIFACTS:
        (analytics / name).write_text(f"analytics:{name}", encoding="utf-8")
    for name in EXPERIMENT_ARTIFACTS:
        (experiment / name).write_text(f"experiment:{name}", encoding="utf-8")

    output = assemble_portfolio(analytics, experiment, tmp_path / "portfolio")
    assert {item.name for item in output.iterdir()} == {
        *ANALYTICS_ARTIFACTS,
        *EXPERIMENT_ARTIFACTS,
    }

    def fail_copy(*args, **kwargs):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr("news_ctr.portfolio.shutil.copy2", fail_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        assemble_portfolio(analytics, experiment, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
