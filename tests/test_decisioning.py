from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from news_ctr.decisioning import (
    DecisionStatus,
    evaluate_decision,
    make_decision,
    render_decision_report,
)
from news_ctr.evidence import EvidenceTier
from news_ctr.experiment_data import write_synthetic_experiment
from news_ctr.experiments import ExperimentConfig, analyze_experiment_frame


@pytest.mark.parametrize(
    ("srm_passed", "primary_interval", "guardrail_state", "expected"),
    [
        (False, (0.01, 0.03), "pass", DecisionStatus.INVALID_EXPERIMENT),
        (True, (0.01, 0.03), "pass", DecisionStatus.LAUNCH),
        (True, (-0.01, 0.02), "pass", DecisionStatus.CONTINUE_EXPERIMENT),
        (True, (-0.03, -0.01), "pass", DecisionStatus.DO_NOT_LAUNCH),
        (True, (0.01, 0.03), "fail", DecisionStatus.DO_NOT_LAUNCH),
        (True, (0.01, 0.03), "inconclusive", DecisionStatus.CONTINUE_EXPERIMENT),
    ],
)
def test_decision_policy(
    srm_passed: bool,
    primary_interval: tuple[float, float],
    guardrail_state: str,
    expected: DecisionStatus,
) -> None:
    assert (
        evaluate_decision(
            srm_passed=srm_passed,
            primary_lower=primary_interval[0],
            primary_upper=primary_interval[1],
            practical_threshold=0.005,
            guardrail_states=[guardrail_state],
        )
        is expected
    )


def test_decision_report_names_evidence_rule_and_guardrails(tmp_path: Path) -> None:
    config = ExperimentConfig.from_json(
        Path(__file__).parents[1] / "configs" / "experiment-v3.json"
    )
    experiment = write_synthetic_experiment(tmp_path / "experiment.parquet", seed=42, users=20_000)
    analysis = analyze_experiment_frame(pd.read_parquet(experiment), config)
    result = make_decision(analysis, config)
    report = render_decision_report(result, config, evidence_tier=EvidenceTier.SYNTHETIC_RCT)

    assert "Synthetic randomized teaching evidence" in report
    assert "not production lift" in report.lower()
    assert "Guardrails" in report
    assert "CUPED" in report
    assert "SRM" in report
    assert "news-ctr experiment" in report
    assert result.status in set(DecisionStatus)


def test_make_decision_marks_both_guardrails(tmp_path: Path) -> None:
    config = ExperimentConfig.from_json(
        Path(__file__).parents[1] / "configs" / "experiment-v3.json"
    )
    experiment = write_synthetic_experiment(tmp_path / "experiment.parquet", seed=7, users=5_000)
    analysis = analyze_experiment_frame(pd.read_parquet(experiment), config)
    result = make_decision(analysis, config)
    assert set(result.guardrails["metric"]) == {"dwell_seconds", "latency_ms"}
    assert set(result.guardrails["state"]) <= {"pass", "fail", "inconclusive"}


def test_decision_report_labels_non_default_confidence_level(tmp_path: Path) -> None:
    config = ExperimentConfig.from_json(
        Path(__file__).parents[1] / "configs" / "experiment-v3.json"
    )
    config = replace(config, alpha=0.10)
    experiment = write_synthetic_experiment(tmp_path / "experiment.parquet", seed=12, users=4_000)
    analysis = analyze_experiment_frame(pd.read_parquet(experiment), config)
    result = make_decision(analysis, config)
    report = render_decision_report(result, config, evidence_tier=EvidenceTier.SYNTHETIC_RCT)
    assert "90% CI" in report
    assert "95% CI" not in report
