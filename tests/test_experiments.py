from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_ctr.experiment_data import write_synthetic_experiment
from news_ctr.experiments import (
    ExperimentConfig,
    ExperimentDesign,
    MetricSpec,
    analyze_experiment_frame,
    estimate_cuped,
    estimate_itt,
    estimate_metric_itt,
    estimate_segment_effects,
    required_sample_size,
    sample_ratio_mismatch,
    validate_experiment,
    validate_experiment_config,
)


def _config() -> ExperimentConfig:
    path = Path(__file__).parents[1] / "configs" / "experiment-v3.json"
    return ExperimentConfig.from_json(path)


@pytest.fixture
def experiment_frame(tmp_path: Path) -> pd.DataFrame:
    path = write_synthetic_experiment(tmp_path / "experiment.parquet", seed=42, users=4_000)
    return pd.read_parquet(path)


def test_required_sample_size_is_monotonic() -> None:
    easy = ExperimentDesign(0.12, 0.02, alpha=0.05, power=0.80)
    hard = ExperimentDesign(0.12, 0.01, alpha=0.05, power=0.80)
    assert required_sample_size(hard) > required_sample_size(easy) > 0


@pytest.mark.parametrize(
    "design",
    [
        ExperimentDesign(0.0, 0.01),
        ExperimentDesign(1.0, -0.01),
        ExperimentDesign(0.99, 0.02),
        ExperimentDesign(0.12, 0.0),
        ExperimentDesign(0.12, 0.01, alpha=0.0),
        ExperimentDesign(0.12, 0.01, power=1.0),
    ],
)
def test_required_sample_size_validates_probability_domains(design: ExperimentDesign) -> None:
    with pytest.raises(ValueError):
        required_sample_size(design)


def test_committed_experiment_config_is_predeclared() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "experiment-v3.json"
    config = ExperimentConfig.from_json(config_path)

    assert config.unit_column == "user_id"
    assert config.primary.name == "click"
    assert config.primary.kind == "binary"
    assert config.cuped_covariate == "pre_ctr"
    assert len(config.guardrails) == 2
    assert config.expected_allocation == {"control": 0.5, "treatment": 0.5}
    assert config.segment_columns == ("device_type", "history_segment")


def test_duplicate_randomization_units_are_rejected(experiment_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([experiment_frame, experiment_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate user_id"):
        validate_experiment(duplicated, _config())


def test_invalid_experiment_values_are_rejected(experiment_frame: pd.DataFrame) -> None:
    cases = [
        (experiment_frame.drop(columns=["click"]), "missing configured columns"),
        (experiment_frame.assign(click=2), "binary metric click"),
        (experiment_frame.assign(latency_ms=np.inf), "finite"),
        (experiment_frame.assign(variant="control"), "both configured arms"),
    ]
    for invalid, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_experiment(invalid, _config())


def test_srm_passes_expected_and_detects_corrupted_assignment(tmp_path: Path) -> None:
    good = write_synthetic_experiment(tmp_path / "good.parquet", seed=42, users=10_000)
    bad = write_synthetic_experiment(
        tmp_path / "bad.parquet", seed=42, users=10_000, corrupt_allocation=True
    )

    good_result = sample_ratio_mismatch(pd.read_parquet(good), _config())
    bad_result = sample_ratio_mismatch(pd.read_parquet(bad), _config())
    assert good_result.passed is True
    assert good_result.p_value >= _config().alpha
    assert bad_result.p_value < _config().alpha
    assert bad_result.passed is False
    assert sum(bad_result.observed_counts.values()) == 10_000


def test_binary_itt_matches_difference_in_means() -> None:
    frame = pd.DataFrame(
        {
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "click": [0, 0, 1, 0, 0, 1, 1, 1],
        }
    )
    result = estimate_metric_itt(
        frame, metric=MetricSpec("click", "binary", "increase"), config=_config()
    )
    assert result.effect == pytest.approx(0.50)
    assert result.control_mean == pytest.approx(0.25)
    assert result.treatment_mean == pytest.approx(0.75)
    assert result.control_units == result.treatment_units == 4
    assert result.ci_lower < result.effect < result.ci_upper


def test_estimate_and_analysis_return_predeclared_metric_rows(
    experiment_frame: pd.DataFrame,
) -> None:
    effects = estimate_itt(experiment_frame, _config())
    assert effects["metric"].tolist() == ["click", "dwell_seconds", "latency_ms"]
    assert effects["effect"].notna().all()

    analysis = analyze_experiment_frame(experiment_frame, _config())
    assert analysis.srm.passed is True
    pd.testing.assert_frame_equal(analysis.effects, effects)
    assert analysis.cuped_effects["metric"].tolist() == ["click"]
    assert not analysis.heterogeneous_effects.empty


def test_cuped_reduces_standard_error_for_correlated_pre_metric(
    experiment_frame: pd.DataFrame,
) -> None:
    config = _config()
    raw = estimate_metric_itt(experiment_frame, metric=config.primary, config=config)
    adjusted = estimate_cuped(
        experiment_frame,
        metric=config.primary,
        covariate=config.cuped_covariate,
        config=config,
    )
    assert adjusted.standard_error < raw.standard_error
    assert adjusted.variance_reduction is not None
    assert adjusted.variance_reduction > 0
    assert adjusted.theta is not None


def test_cuped_with_constant_covariate_is_unchanged(experiment_frame: pd.DataFrame) -> None:
    config = _config()
    constant = experiment_frame.assign(pre_ctr=0.1)
    raw = estimate_metric_itt(constant, metric=config.primary, config=config)
    adjusted = estimate_cuped(
        constant,
        metric=config.primary,
        covariate=config.cuped_covariate,
        config=config,
    )
    assert adjusted.effect == pytest.approx(raw.effect)
    assert adjusted.standard_error == pytest.approx(raw.standard_error)
    assert adjusted.theta == 0
    assert adjusted.variance_reduction == 0


def test_post_treatment_covariate_is_rejected() -> None:
    bad = replace(_config(), cuped_covariate="latency_ms")
    with pytest.raises(ValueError, match="pre-treatment"):
        validate_experiment_config(bad)

    self_adjusting = replace(
        _config(),
        cuped_covariate="click",
        pre_treatment_columns=("pre_ctr", "click"),
    )
    with pytest.raises(ValueError, match="outcome metric"):
        validate_experiment_config(self_adjusting)


def test_experiment_config_rejects_overlapping_metric_roles() -> None:
    config = _config()
    duplicate_guardrail = replace(config, guardrails=(config.primary, *config.guardrails))
    with pytest.raises(ValueError, match="unique"):
        validate_experiment_config(duplicate_guardrail)


def test_experiment_config_restricts_primary_to_increase_direction() -> None:
    config = _config()
    unsupported = replace(config, primary=replace(config.primary, direction="decrease"))
    with pytest.raises(ValueError, match="primary direction"):
        validate_experiment_config(unsupported)


def test_segment_effects_are_exploratory_and_flag_low_support(
    experiment_frame: pd.DataFrame,
) -> None:
    config = replace(_config(), minimum_segment_units=1_500)
    result = estimate_segment_effects(experiment_frame, config)
    assert set(result["analysis_role"]) == {"exploratory"}
    assert result.loc[result["units"] < config.min_segment_units, "low_support"].all()
    assert not result.loc[result["units"] >= config.min_segment_units, "low_support"].any()
    assert set(result["segment"]) == {"device_type", "history_segment"}


def test_confirmatory_segment_intervals_use_holm_thresholds(
    experiment_frame: pd.DataFrame,
) -> None:
    config = replace(_config(), confirmatory_segments=("device_type",))
    result = estimate_segment_effects(experiment_frame, config)
    confirmatory = result.loc[result["analysis_role"] == "confirmatory"]
    assert len(confirmatory) == experiment_frame["device_type"].nunique()
    assert confirmatory["multiplicity_alpha"].notna().all()
    assert (confirmatory["multiplicity_alpha"] <= config.alpha).all()
    assert confirmatory["holm_adjusted_p_value"].notna().all()
    assert set(confirmatory["interval_method"]) == {"bonferroni"}
    exploratory = result.loc[result["analysis_role"] == "exploratory"]
    assert set(exploratory["interval_method"]) == {"pointwise"}
