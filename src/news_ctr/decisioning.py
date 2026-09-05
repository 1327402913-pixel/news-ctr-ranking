"""Transparent experiment launch policy and recruiter-readable decision memo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from news_ctr.evidence import EvidenceTier, evidence_banner
from news_ctr.experiments import ExperimentAnalysis, ExperimentConfig, MetricSpec, SRMResult


class DecisionStatus(str, Enum):
    """Mutually exclusive recommendation states."""

    INVALID_EXPERIMENT = "invalid_experiment"
    LAUNCH = "launch"
    CONTINUE_EXPERIMENT = "continue_experiment"
    DO_NOT_LAUNCH = "do_not_launch"


@dataclass(frozen=True)
class DecisionResult:
    """All evidence used by the deterministic decision policy."""

    status: DecisionStatus
    srm: SRMResult
    raw_primary: dict[str, object]
    decision_primary: dict[str, object]
    guardrails: pd.DataFrame
    segment_effects: pd.DataFrame


def evaluate_decision(
    *,
    srm_passed: bool,
    primary_lower: float,
    primary_upper: float,
    practical_threshold: float,
    guardrail_states: Sequence[str],
) -> DecisionStatus:
    """Apply the predeclared integrity, benefit, and guardrail rule."""

    unknown = set(guardrail_states) - {"pass", "fail", "inconclusive"}
    if unknown:
        raise ValueError(f"unknown guardrail states: {', '.join(sorted(unknown))}")
    if not srm_passed:
        return DecisionStatus.INVALID_EXPERIMENT
    if "fail" in guardrail_states or primary_upper < 0:
        return DecisionStatus.DO_NOT_LAUNCH
    if "inconclusive" in guardrail_states:
        return DecisionStatus.CONTINUE_EXPERIMENT
    if primary_lower > practical_threshold:
        return DecisionStatus.LAUNCH
    return DecisionStatus.CONTINUE_EXPERIMENT


def _guardrail_state(row: pd.Series, metric: MetricSpec) -> str:
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    margin = metric.non_inferiority_margin
    if metric.direction in {"increase", "non_decrease"}:
        if lower >= margin:
            return "pass"
        if upper < margin:
            return "fail"
        return "inconclusive"
    if metric.direction in {"decrease", "non_increase"}:
        if upper <= margin:
            return "pass"
        if lower > margin:
            return "fail"
        return "inconclusive"
    raise ValueError(f"unsupported guardrail direction: {metric.direction}")


def _metric_row(frame: pd.DataFrame, name: str) -> pd.Series:
    selected = frame.loc[frame["metric"] == name]
    if len(selected) != 1:
        raise ValueError(f"analysis must contain exactly one row for metric {name}")
    return selected.iloc[0]


def make_decision(analysis: ExperimentAnalysis, config: ExperimentConfig) -> DecisionResult:
    """Turn experiment estimates into an auditable product recommendation."""

    raw_primary = _metric_row(analysis.effects, config.primary.name)
    decision_primary = (
        _metric_row(analysis.cuped_effects, config.primary.name)
        if not analysis.cuped_effects.empty
        else raw_primary
    )
    guardrail_rows: list[dict[str, object]] = []
    for metric in config.guardrails:
        estimate = _metric_row(analysis.effects, metric.name)
        guardrail_rows.append(
            {
                "metric": metric.name,
                "direction": metric.direction,
                "margin": metric.non_inferiority_margin,
                "effect": float(estimate["effect"]),
                "ci_lower": float(estimate["ci_lower"]),
                "ci_upper": float(estimate["ci_upper"]),
                "state": _guardrail_state(estimate, metric),
            }
        )
    guardrails = pd.DataFrame(guardrail_rows)
    status = evaluate_decision(
        srm_passed=analysis.srm.passed,
        primary_lower=float(decision_primary["ci_lower"]),
        primary_upper=float(decision_primary["ci_upper"]),
        practical_threshold=config.practical_threshold,
        guardrail_states=guardrails["state"].tolist(),
    )
    return DecisionResult(
        status=status,
        srm=analysis.srm,
        raw_primary=raw_primary.to_dict(),
        decision_primary=decision_primary.to_dict(),
        guardrails=guardrails,
        segment_effects=analysis.heterogeneous_effects.copy(),
    )


def _format_interval(row: dict[str, object] | pd.Series) -> str:
    return (
        f"{float(row['effect']):.4f} [{float(row['ci_lower']):.4f}, {float(row['ci_upper']):.4f}]"
    )


def render_decision_report(
    result: DecisionResult,
    config: ExperimentConfig,
    *,
    evidence_tier: EvidenceTier,
    provenance: Mapping[str, str] | None = None,
) -> str:
    """Render the complete decision rule, evidence, and limitations as Markdown."""

    guardrail_lines = "\n".join(
        f"| `{row.metric}` | {row.direction} | {row.margin:.3f} | "
        f"{row.effect:.4f} [{row.ci_lower:.4f}, {row.ci_upper:.4f}] | **{row.state}** |"
        for row in result.guardrails.itertuples(index=False)
    )
    low_support = (
        int(result.segment_effects["low_support"].sum()) if not result.segment_effects.empty else 0
    )
    total_segments = len(result.segment_effects)
    confidence_label = f"{100 * (1 - config.alpha):g}% CI"
    raw_primary_interval = _format_interval(result.raw_primary)
    decision_primary_interval = _format_interval(result.decision_primary)
    recommendation = {
        DecisionStatus.INVALID_EXPERIMENT: "Do not interpret effects; repair experiment validity.",
        DecisionStatus.LAUNCH: "Launch under the declared policy and monitor guardrails.",
        DecisionStatus.CONTINUE_EXPERIMENT: "Continue the experiment; evidence is not decisive.",
        DecisionStatus.DO_NOT_LAUNCH: "Do not launch under the declared policy.",
    }[result.status]
    provenance_lines = (
        "\n".join(
            [
                f"- Input SHA-256: `{provenance['input_sha256']}`",
                f"- Metadata SHA-256: `{provenance['metadata_sha256']}`",
                f"- Config SHA-256: `{provenance['config_sha256']}`",
            ]
        )
        if provenance is not None
        else "- In-memory report; persisted workflows add input, metadata, and config hashes."
    )
    return f"""# Experiment Decision Report

> {evidence_banner(evidence_tier)}

- Evidence tier: `{evidence_tier.value}`

## Provenance

{provenance_lines}

## Recommendation

**{result.status.value}** — {recommendation}

## Experiment validity

- Randomization unit: `{config.unit_column}`
- SRM status: **{"pass" if result.srm.passed else "fail"}**
- SRM p-value: {result.srm.p_value:.6g}
- Observed allocation: `{result.srm.observed_counts}`
- Expected allocation: `{result.srm.expected_counts}`

## Primary metric

- Raw ITT ({confidence_label}): {raw_primary_interval}
- CUPED-adjusted ITT used for the decision ({confidence_label}): {decision_primary_interval}
- Minimum practically important effect: {config.practical_threshold:.4f}

## Guardrails

| Metric | Direction | Margin | Effect and {confidence_label} | State |
| --- | --- | ---: | ---: | --- |
{guardrail_lines}

## Heterogeneous effects

All undeclared segment cuts are exploratory. {total_segments} cells were estimated;
{low_support} are marked low support at the threshold of {config.minimum_segment_units}
randomization units.

## Predeclared decision rule

1. SRM failure makes the experiment invalid.
2. Launch requires the CUPED primary lower confidence bound to exceed
   {config.practical_threshold:.4f}.
3. A credibly harmful primary effect or failed guardrail means do not launch.
4. Any interval crossing a decision margin means continue the experiment.

## Reproduce

```bash
news-ctr make-experiment --output data/synthetic-rct.parquet --seed 42 --users 20000
news-ctr experiment --input data/synthetic-rct.parquet \\
  --output artifacts/experiment-v3 --config configs/experiment-v3.json
```

## Limitations

- This workflow estimates user-level intent-to-treat effects for a randomized fixture.
- Segment effects are exploratory unless explicitly predeclared as confirmatory.
- Product, editorial, novelty, diversity, and longer-term effects remain out of scope.
"""
