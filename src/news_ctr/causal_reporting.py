"""Deterministic interpretation and reporting for quasi-experimental evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from news_ctr.causal import CausalAnalysis, CausalConfig


class CausalDecisionStatus(str, Enum):
    """Mutually exclusive outcomes from the predeclared interpretation policy."""

    INVALID_DESIGN = "invalid_design"
    SUPPORTS_INCREMENTAL_IMPACT = "supports_incremental_impact"
    EVIDENCE_OF_NO_BENEFIT = "evidence_of_no_benefit"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CausalDecision:
    """Compact policy result with the evidence needed to audit it."""

    status: CausalDecisionStatus
    diagnostics: dict[str, object]
    did_estimate: dict[str, object]
    placebo_estimate: dict[str, object]
    business_impact: dict[str, object]


def _require_bool(mapping: dict[str, object], key: str, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context} diagnostic {key} must be boolean")
    return value


def _single_record(frame, context: str) -> dict[str, object]:
    if len(frame) != 1:
        raise ValueError(f"{context} must contain exactly one row")
    return frame.iloc[0].to_dict()


def evaluate_causal_decision(analysis: CausalAnalysis, config: CausalConfig) -> CausalDecision:
    """Apply the predeclared validity-first four-state interpretation policy."""

    parallel = analysis.diagnostics.get("parallel_trends")
    placebo = analysis.diagnostics.get("placebo")
    if not isinstance(parallel, dict) or not isinstance(placebo, dict):
        raise ValueError("causal diagnostics must include parallel_trends and placebo mappings")
    parallel_passed = _require_bool(parallel, "passed", "parallel_trends")
    placebo_passed = _require_bool(placebo, "passed", "placebo")

    did = _single_record(analysis.did_estimate, "DiD estimate")
    placebo_estimate = _single_record(analysis.placebo_estimate, "placebo estimate")
    business = _single_record(analysis.business_impact, "business impact")
    try:
        lower = float(did["ci_lower"])
        upper = float(did["ci_upper"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("DiD estimate must contain a numeric confidence interval") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("DiD confidence interval must contain finite values")
    if lower > upper:
        raise ValueError("DiD confidence interval lower bound exceeds upper bound")

    if not parallel_passed or not placebo_passed:
        status = CausalDecisionStatus.INVALID_DESIGN
    elif upper <= 0:
        status = CausalDecisionStatus.EVIDENCE_OF_NO_BENEFIT
    elif lower > config.practical_threshold:
        status = CausalDecisionStatus.SUPPORTS_INCREMENTAL_IMPACT
    else:
        status = CausalDecisionStatus.INCONCLUSIVE
    return CausalDecision(
        status=status,
        diagnostics={
            "parallel_trends": dict(parallel),
            "placebo": dict(placebo),
        },
        did_estimate=did,
        placebo_estimate=placebo_estimate,
        business_impact=business,
    )


def _balance_markdown(analysis: CausalAnalysis) -> str:
    lines = [
        "| Metric | Treated mean | Control mean | Standardized difference |",
        "|---|---:|---:|---:|",
    ]
    for row in analysis.balance.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {float(row['treated_mean']):.4f} | "
            f"{float(row['control_mean']):.4f} | "
            f"{float(row['standardized_mean_difference']):.4f} |"
        )
    return "\n".join(lines)


def render_causal_report(
    decision: CausalDecision,
    analysis: CausalAnalysis,
    config: CausalConfig,
    *,
    evidence_tier: str,
    provenance: dict[str, object],
) -> str:
    """Render a candid, recruiter-readable Markdown causal impact brief."""

    did = decision.did_estimate
    placebo = decision.placebo_estimate
    impact = decision.business_impact
    integrity = analysis.diagnostics["integrity"]
    parallel = decision.diagnostics["parallel_trends"]
    effect_pp = float(did["effect"]) * 100
    lower_pp = float(did["ci_lower"]) * 100
    upper_pp = float(did["ci_upper"]) * 100
    provenance_lines = "\n".join(
        f"- `{key}`: `{value}`" for key, value in sorted(provenance.items())
    )

    return f"""# Causal Impact V4: News Ranking Rollout

> Evidence tier: `{evidence_tier}`. This is deterministic synthetic quasi-experimental
> teaching evidence, not production lift and not a real deployment recommendation.

## Decision question

Did activating the candidate ranking policy increase market-week CTR by more than the
predeclared practical threshold of {config.practical_threshold * 100:.4f} percentage points?

**Policy result:** `{decision.status.value}`

## Identification strategy

The primary Difference-in-Differences model uses candidate-exposure-weighted least squares,
market fixed effects, week fixed effects, and uncertainty clustered by `{config.unit_column}`.
The estimand is the treated-versus-control change after rollout, conditional on common time
shocks and time-invariant market differences.

## Integrity summary

- Markets: {int(integrity["markets"])}
- Pre/post weeks: {int(integrity["pre_weeks"])}/{int(integrity["post_weeks"])}
- Missing market-week share: {float(integrity["missing_market_week_share"]):.4f}
- Structural integrity passed: {bool(integrity["integrity_passed"])}

## Primary estimate

The estimated absolute CTR effect is **{effect_pp:.4f} percentage points**
(confidence interval: {lower_pp:.4f} to {upper_pp:.4f} percentage points;
p={float(did["p_value"]):.4f}). The interval, not the point estimate alone, drives the
predeclared decision status.

## Event study and Parallel trends

The event study omits relative week {config.reference_week} and estimates treated-market
interactions across weeks {config.event_window[0]} through {config.event_window[1]}.
The joint Parallel trends test covers {int(parallel["lead_terms"])} pre-treatment leads:
p={float(parallel["p_value"]):.4f}; passed={bool(parallel["passed"])}. Passing is supportive
evidence for the identifying assumption, not proof.

## Placebo rollout

The Placebo rollout at week {config.placebo_week} uses only observations before the real
rollout. Its estimate is {float(placebo["effect"]) * 100:.4f} percentage points with interval
[{float(placebo["ci_lower"]) * 100:.4f}, {float(placebo["ci_upper"]) * 100:.4f}];
passed={bool(decision.diagnostics["placebo"]["passed"])}.

## Pre-period balance

{_balance_markdown(analysis)}

Balance is descriptive. It does not establish that post-treatment trends are unconfounded.

## Business translation

Per {int(impact["scale_exposures"]):,} candidate exposures, the model implies
**{float(impact["incremental_clicks"]):,.0f} incremental clicks** with interval
[{float(impact["ci_lower_clicks"]):,.0f}, {float(impact["ci_upper_clicks"]):,.0f}]. No revenue,
margin, or lifetime-value assumption is added.

## Interpretation, assumptions, and limitations

The result is `{decision.status.value}` under the predeclared policy. Interpretation requires
parallel untreated potential-outcome trends, no rollout anticipation, stable treatment, and no
spillovers between markets. Event studies and placebos can reveal some violations but cannot
exclude unobserved time-varying confounding. The data are synthetic, so the analysis demonstrates
workflow competence rather than external validity or production impact.

## Provenance

{provenance_lines}

## Reproduce

Run `make causal-impact-v4` from a clean checkout. Stored CSV files retain full numeric precision;
rounding above is presentation-only.
"""
