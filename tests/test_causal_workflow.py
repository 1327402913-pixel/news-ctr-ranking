from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import news_ctr.causal_workflow as causal_workflow
from news_ctr.causal_workflow import CAUSAL_ARTIFACTS, run_causal_impact
from news_ctr.quasi_data import write_synthetic_market_panel


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    panel = write_synthetic_market_panel(
        tmp_path / "market-week.parquet",
        seed=42,
        markets=60,
        pre_weeks=20,
        post_weeks=12,
    )
    config = Path(__file__).parents[1] / "configs" / "causal-impact-v4.json"
    return panel, config


def test_causal_workflow_writes_complete_truthful_evidence(tmp_path: Path) -> None:
    """Catches incomplete artifacts or provenance that cannot identify the analyzed inputs."""

    panel, config = _fixture(tmp_path)
    metadata_path = panel.with_suffix(".metadata.json")
    output = run_causal_impact(panel, tmp_path / "study", config)

    assert tuple(sorted(item.name for item in output.iterdir())) == CAUSAL_ARTIFACTS
    assert (output / "config_snapshot.json").read_bytes() == config.read_bytes()
    report = (output / "causal_report.md").read_text(encoding="utf-8")
    assert "synthetic-quasi-experiment" in report
    assert "not production lift" in report.lower()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["evidence_tier"] == "synthetic-quasi-experiment"
    assert manifest["artifacts"] == list(CAUSAL_ARTIFACTS)
    assert manifest["input"]["sha256"] == hashlib.sha256(panel.read_bytes()).hexdigest()
    assert manifest["metadata"]["sha256"] == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    assert manifest["config"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert manifest["analysis_contract"]["weighting"] == "candidate_exposures"
    assert manifest["analysis_contract"]["covariance"] == "cluster:market_id"


def test_causal_workflow_rejects_missing_input_or_sidecar(tmp_path: Path) -> None:
    """Catches analyzing an absent or untraceable evidence source."""

    config = Path(__file__).parents[1] / "configs" / "causal-impact-v4.json"
    with pytest.raises(ValueError, match="input does not exist"):
        run_causal_impact(tmp_path / "missing.parquet", tmp_path / "study", config)

    panel, config = _fixture(tmp_path)
    panel.with_suffix(".metadata.json").unlink()
    with pytest.raises(ValueError, match="metadata does not exist"):
        run_causal_impact(panel, tmp_path / "study", config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("panel_sha256", "0" * 64),
        ("evidence_tier", "synthetic-rct"),
        ("rows", 1),
        ("markets", 1),
        ("pre_weeks", 1),
        ("post_weeks", 1),
        ("known_effects", {"ctr_absolute": 99.0}),
    ],
)
def test_causal_workflow_validates_sidecar_contract(
    tmp_path: Path, field: str, value: object
) -> None:
    """Catches stale or mislabeled metadata reaching the published report."""

    panel, config = _fixture(tmp_path)
    metadata_path = panel.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        run_causal_impact(panel, tmp_path / "study", config)


def test_causal_workflow_reports_structure_before_sidecar_mismatch(tmp_path: Path) -> None:
    """Catches provenance checks hiding a more actionable panel-schema error."""

    panel, config = _fixture(tmp_path)
    pd.read_parquet(panel).drop(columns="policy_active").to_parquet(panel, index=False)

    with pytest.raises(ValueError, match="missing configured columns: policy_active"):
        run_causal_impact(panel, tmp_path / "study", config)


def test_causal_workflow_is_atomic_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches failed runs leaving partial evidence or reruns replacing prior results."""

    panel, config = _fixture(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run_causal_impact(panel, existing, config)

    output = tmp_path / "failed"

    def fail_outputs(*args, **kwargs):
        raise RuntimeError("injected causal writer failure")

    monkeypatch.setattr(causal_workflow, "_write_causal_outputs", fail_outputs)
    with pytest.raises(RuntimeError, match="injected causal writer failure"):
        run_causal_impact(panel, output, config)

    assert not output.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))
