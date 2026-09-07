from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from news_ctr.causal import CausalAnalysis, CausalConfig
from news_ctr.causal_reporting import (
    CausalDecisionStatus,
    evaluate_causal_decision,
    render_causal_report,
)


def _config() -> CausalConfig:
    return CausalConfig(
        unit_column="market_id",
        time_column="relative_week",
        group_column="treated_market",
        active_column="policy_active",
        exposure_column="candidate_exposures",
        click_column="clicks",
        outcome_column="ctr",
        outcome_kind="rate",
        balance_columns=("market_size_index", "candidate_exposures", "ctr"),
        rollout_week=0,
        reference_week=-1,
        event_window=(-12, 8),
        placebo_week=-8,
        alpha=0.05,
        practical_threshold=0.002,
        minimum_markets=20,
        minimum_treated_markets=8,
        minimum_control_markets=8,
        minimum_pre_weeks=8,
        minimum_post_weeks=4,
        maximum_missing_market_week_share=0.0,
        business_exposure_scale=1_000_000,
    )


def _analysis(
    *,
    parallel: bool = True,
    placebo: bool = True,
    effect: float = 0.006,
    lower: float = 0.004,
    upper: float = 0.008,
) -> CausalAnalysis:
    did = pd.DataFrame(
        [
            {
                "term": "policy_active",
                "effect": effect,
                "standard_error": 0.001,
                "statistic": 6.0,
                "p_value": 0.0001,
                "ci_lower": lower,
                "ci_upper": upper,
                "treated_markets": 30,
                "control_markets": 30,
                "clusters": 60,
                "observations": 1_920,
                "total_exposure": 25_000_000,
            }
        ]
    )
    placebo_estimate = pd.DataFrame(
        [
            {
                "term": "placebo_policy_active",
                "effect": 0.0001,
                "ci_lower": -0.001,
                "ci_upper": 0.0012,
            }
        ]
    )
    business = pd.DataFrame(
        [
            {
                "scale_exposures": 1_000_000,
                "incremental_clicks": effect * 1_000_000,
                "ci_lower_clicks": lower * 1_000_000,
                "ci_upper_clicks": upper * 1_000_000,
            }
        ]
    )
    return CausalAnalysis(
        did_estimate=did,
        event_study=pd.DataFrame(
            [
                {
                    "relative_week": 0,
                    "effect": effect,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            ]
        ),
        placebo_estimate=placebo_estimate,
        balance=pd.DataFrame(
            [
                {
                    "metric": "market_size_index",
                    "treated_mean": 1.1,
                    "control_mean": 0.9,
                    "standardized_mean_difference": 0.2,
                }
            ]
        ),
        business_impact=business,
        diagnostics={
            "integrity": {
                "integrity_passed": True,
                "markets": 60,
                "pre_weeks": 20,
                "post_weeks": 12,
                "missing_market_week_share": 0.0,
            },
            "parallel_trends": {
                "passed": parallel,
                "p_value": 0.40 if parallel else 0.01,
                "lead_terms": 11,
            },
            "placebo": {
                "passed": placebo,
                "ci_contains_zero": placebo,
                "placebo_week": -8,
            },
        },
    )


@pytest.mark.parametrize(
    ("parallel", "placebo", "lower", "upper", "expected"),
    [
        (False, True, 0.004, 0.008, "invalid_design"),
        (True, False, 0.004, 0.008, "invalid_design"),
        (True, True, 0.003, 0.008, "supports_incremental_impact"),
        (True, True, -0.008, 0.0, "evidence_of_no_benefit"),
        (True, True, -0.001, 0.006, "inconclusive"),
    ],
)
def test_causal_decision_applies_the_predeclared_four_state_policy(
    parallel: bool,
    placebo: bool,
    lower: float,
    upper: float,
    expected: str,
) -> None:
    """Catches selectively interpreting a positive estimate after diagnostics fail."""

    decision = evaluate_causal_decision(
        _analysis(parallel=parallel, placebo=placebo, lower=lower, upper=upper),
        _config(),
    )

    assert decision.status is CausalDecisionStatus(expected)


def test_causal_decision_rejects_non_finite_intervals() -> None:
    """Catches a malformed estimate being silently converted into a decision."""

    analysis = _analysis()
    broken = replace(
        analysis,
        did_estimate=analysis.did_estimate.assign(ci_lower=float("nan")),
    )

    with pytest.raises(ValueError, match="finite"):
        evaluate_causal_decision(broken, _config())


def test_causal_report_is_candid_recruiter_readable_and_reproducible() -> None:
    """Catches a portfolio report that overclaims or omits the identification logic."""

    config = _config()
    analysis = _analysis()
    decision = evaluate_causal_decision(analysis, config)
    provenance = {
        "input_path": "data/quasi/market_week.parquet",
        "input_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "config_sha256": "c" * 64,
    }

    report = render_causal_report(
        decision,
        analysis,
        config,
        evidence_tier="synthetic-quasi-experiment",
        provenance=provenance,
    )

    required = [
        "synthetic-quasi-experiment",
        "not production lift",
        "Difference-in-Differences",
        "market fixed effects",
        "week fixed effects",
        "clustered by `market_id`",
        "Parallel trends",
        "Placebo rollout",
        "per 1,000,000 candidate exposures",
        "supports_incremental_impact",
        "unobserved time-varying confounding",
        "make causal-impact-v4",
    ]
    assert all(item.lower() in report.lower() for item in required)
    assert provenance["input_sha256"] in report
    assert "0.6000 percentage points" in report
    assert "6,000 incremental clicks" in report
