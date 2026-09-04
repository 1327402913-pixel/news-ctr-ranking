"""Experiment design and analysis contracts for randomized ranking tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm

from news_ctr.evidence import EvidenceTier


@dataclass(frozen=True)
class ExperimentDesign:
    """Inputs for a two-arm binary-outcome power calculation."""

    baseline_rate: float
    absolute_mde: float
    alpha: float = 0.05
    power: float = 0.80


def _validate_design(design: ExperimentDesign) -> None:
    if not 0 < design.baseline_rate < 1:
        raise ValueError("baseline_rate must be between zero and one")
    if design.absolute_mde <= 0:
        raise ValueError("absolute_mde must be positive")
    if not 0 < design.baseline_rate + design.absolute_mde < 1:
        raise ValueError("baseline_rate plus absolute_mde must be between zero and one")
    if not 0 < design.alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not 0 < design.power < 1:
        raise ValueError("power must be between zero and one")


def required_sample_size(design: ExperimentDesign) -> int:
    """Return required randomization units per arm using a normal approximation."""

    _validate_design(design)
    p1 = design.baseline_rate
    p2 = p1 + design.absolute_mde
    pooled = (p1 + p2) / 2
    numerator = (
        norm.ppf(1 - design.alpha / 2) * math.sqrt(2 * pooled * (1 - pooled))
        + norm.ppf(design.power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / design.absolute_mde**2)


@dataclass(frozen=True)
class MetricSpec:
    """Predeclared experiment metric and its launch interpretation."""

    name: str
    kind: str
    direction: str
    non_inferiority_margin: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricSpec:
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            direction=str(payload["direction"]),
            non_inferiority_margin=float(payload.get("non_inferiority_margin", 0.0)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable pre-analysis decisions loaded from reviewed JSON."""

    unit_column: str
    variant_column: str
    control_label: str
    treatment_label: str
    expected_allocation: dict[str, float]
    primary: MetricSpec
    guardrails: tuple[MetricSpec, ...]
    cuped_covariate: str
    pre_treatment_columns: tuple[str, ...]
    segment_columns: tuple[str, ...]
    alpha: float
    practical_threshold: float
    minimum_segment_units: int
    confirmatory_segments: tuple[str, ...] = ()

    @property
    def min_segment_units(self) -> int:
        """Compatibility alias used in result interpretation."""

        return self.minimum_segment_units

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            unit_column=str(payload["unit_column"]),
            variant_column=str(payload["variant_column"]),
            control_label=str(payload["control_label"]),
            treatment_label=str(payload["treatment_label"]),
            expected_allocation={
                str(label): float(share) for label, share in payload["expected_allocation"].items()
            },
            primary=MetricSpec.from_dict(payload["primary"]),
            guardrails=tuple(MetricSpec.from_dict(item) for item in payload["guardrails"]),
            cuped_covariate=str(payload["cuped_covariate"]),
            pre_treatment_columns=tuple(str(item) for item in payload["pre_treatment_columns"]),
            segment_columns=tuple(str(item) for item in payload["segment_columns"]),
            alpha=float(payload["alpha"]),
            practical_threshold=float(payload["practical_threshold"]),
            minimum_segment_units=int(payload["minimum_segment_units"]),
            confirmatory_segments=tuple(
                str(item) for item in payload.get("confirmatory_segments", [])
            ),
        )
        validate_experiment_config(config)
        return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate predeclared metrics, allocation, and analysis boundaries."""

    if not config.unit_column or not config.variant_column:
        raise ValueError("unit_column and variant_column must not be empty")
    if config.control_label == config.treatment_label:
        raise ValueError("control and treatment labels must differ")
    expected_labels = {config.control_label, config.treatment_label}
    if set(config.expected_allocation) != expected_labels:
        raise ValueError("expected_allocation must contain the configured experiment arms")
    if any(share <= 0 for share in config.expected_allocation.values()) or not math.isclose(
        sum(config.expected_allocation.values()), 1.0
    ):
        raise ValueError("expected_allocation shares must be positive and sum to one")
    if config.primary.kind not in {"binary", "continuous"}:
        raise ValueError("primary metric kind must be binary or continuous")
    if config.primary.direction != "increase":
        raise ValueError("primary direction must be increase for this launch policy")
    if any(metric.kind not in {"binary", "continuous"} for metric in config.guardrails):
        raise ValueError("guardrail metric kind must be binary or continuous")
    guardrail_directions = {"increase", "decrease", "non_decrease", "non_increase"}
    if any(metric.direction not in guardrail_directions for metric in config.guardrails):
        raise ValueError("guardrail metric direction is unsupported")
    metric_names = [metric.name for metric in _configured_metrics(config)]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("primary and guardrail metric names must be unique")
    if config.cuped_covariate not in config.pre_treatment_columns:
        raise ValueError("CUPED covariate must be declared as pre-treatment")
    if set(config.pre_treatment_columns) & set(metric_names):
        raise ValueError("pre-treatment columns must not overlap an outcome metric")
    if not set(config.confirmatory_segments).issubset(config.segment_columns):
        raise ValueError("confirmatory segments must be configured segment columns")
    if not 0 < config.alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if config.practical_threshold < 0:
        raise ValueError("practical_threshold must not be negative")
    if config.minimum_segment_units < 2:
        raise ValueError("minimum_segment_units must be at least two")


@dataclass(frozen=True)
class SRMResult:
    """Pearson sample-ratio-mismatch result for configured allocation."""

    observed_counts: dict[str, int]
    expected_counts: dict[str, float]
    statistic: float
    p_value: float
    passed: bool


@dataclass(frozen=True)
class EffectEstimate:
    """Unadjusted intent-to-treat difference in arm means."""

    metric: str
    kind: str
    control_units: int
    treatment_units: int
    control_mean: float
    treatment_mean: float
    effect: float
    relative_effect: float | None
    standard_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    theta: float | None = None
    raw_standard_error: float | None = None
    variance_reduction: float | None = None


@dataclass(frozen=True)
class ExperimentAnalysis:
    """Structured in-memory experiment result before report rendering."""

    srm: SRMResult
    effects: pd.DataFrame
    cuped_effects: pd.DataFrame
    heterogeneous_effects: pd.DataFrame


def _configured_metrics(config: ExperimentConfig) -> tuple[MetricSpec, ...]:
    return (config.primary, *config.guardrails)


def validate_experiment(frame: pd.DataFrame, config: ExperimentConfig) -> None:
    """Reject experiment frames that violate the predeclared analysis contract."""

    required = {
        config.unit_column,
        config.variant_column,
        config.cuped_covariate,
        *config.pre_treatment_columns,
        *config.segment_columns,
        *(metric.name for metric in _configured_metrics(config)),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing configured columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("experiment frame must not be empty")
    if frame[config.unit_column].isna().any():
        raise ValueError(f"{config.unit_column} must not contain null values")
    if frame[config.unit_column].duplicated().any():
        raise ValueError(f"duplicate {config.unit_column} randomization units")
    if frame[config.variant_column].isna().any():
        raise ValueError(f"{config.variant_column} must not contain null values")

    arms = set(frame[config.variant_column].unique())
    configured_arms = {config.control_label, config.treatment_label}
    if arms != configured_arms:
        raise ValueError("experiment must contain exactly both configured arms")

    numeric_columns = [
        *(metric.name for metric in _configured_metrics(config)),
        config.cuped_covariate,
    ]
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"configured metric {column} must be numeric")
        if frame[column].isna().any() or not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"configured metric {column} must contain finite values")
    for metric in _configured_metrics(config):
        if metric.kind == "binary" and not frame[metric.name].isin([0, 1]).all():
            raise ValueError(f"binary metric {metric.name} must contain only zero or one")
    for column in config.segment_columns:
        if frame[column].isna().any():
            raise ValueError(f"segment column {column} must not contain null values")


def sample_ratio_mismatch(frame: pd.DataFrame, config: ExperimentConfig) -> SRMResult:
    """Test observed assignment counts against predeclared arm shares."""

    validate_experiment(frame, config)
    labels = (config.control_label, config.treatment_label)
    counts = frame[config.variant_column].value_counts()
    observed = np.array([counts[label] for label in labels], dtype=float)
    expected = np.array([len(frame) * config.expected_allocation[label] for label in labels])
    statistic, p_value = chisquare(observed, f_exp=expected)
    return SRMResult(
        observed_counts={label: int(value) for label, value in zip(labels, observed, strict=True)},
        expected_counts={
            label: float(value) for label, value in zip(labels, expected, strict=True)
        },
        statistic=float(statistic),
        p_value=float(p_value),
        passed=bool(p_value >= config.alpha),
    )


def estimate_metric_itt(
    frame: pd.DataFrame,
    *,
    metric: MetricSpec,
    config: ExperimentConfig,
) -> EffectEstimate:
    """Estimate a two-sided normal interval for a treatment-minus-control mean."""

    needed = {config.variant_column, metric.name}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns for ITT estimate: {', '.join(missing)}")
    if not pd.api.types.is_numeric_dtype(frame[metric.name]):
        raise ValueError(f"configured metric {metric.name} must be numeric")
    if frame[metric.name].isna().any() or not np.isfinite(frame[metric.name].to_numpy()).all():
        raise ValueError(f"configured metric {metric.name} must contain finite values")
    if metric.kind == "binary" and not frame[metric.name].isin([0, 1]).all():
        raise ValueError(f"binary metric {metric.name} must contain only zero or one")

    control = frame.loc[frame[config.variant_column] == config.control_label, metric.name].astype(
        float
    )
    treatment = frame.loc[
        frame[config.variant_column] == config.treatment_label, metric.name
    ].astype(float)
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("ITT estimation requires at least two units in each arm")

    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    effect = treatment_mean - control_mean
    standard_error = math.sqrt(
        control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment)
    )
    critical_value = float(norm.ppf(1 - config.alpha / 2))
    if standard_error == 0:
        p_value = 1.0 if effect == 0 else 0.0
    else:
        p_value = float(2 * norm.sf(abs(effect / standard_error)))
    return EffectEstimate(
        metric=metric.name,
        kind=metric.kind,
        control_units=len(control),
        treatment_units=len(treatment),
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        effect=effect,
        relative_effect=effect / control_mean if control_mean != 0 else None,
        standard_error=standard_error,
        ci_lower=effect - critical_value * standard_error,
        ci_upper=effect + critical_value * standard_error,
        p_value=p_value,
    )


def estimate_itt(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Estimate raw ITT effects in the predeclared metric order."""

    validate_experiment(frame, config)
    return pd.DataFrame(
        [
            asdict(estimate_metric_itt(frame, metric=metric, config=config))
            for metric in _configured_metrics(config)
        ]
    )


def estimate_cuped(
    frame: pd.DataFrame,
    *,
    metric: MetricSpec,
    covariate: str,
    config: ExperimentConfig,
) -> EffectEstimate:
    """Estimate a CUPED-adjusted effect using a declared pre-treatment covariate."""

    validate_experiment_config(config)
    validate_experiment(frame, config)
    if covariate not in config.pre_treatment_columns:
        raise ValueError(f"CUPED covariate {covariate} must be declared as pre-treatment")
    if covariate not in frame:
        raise ValueError(f"CUPED covariate is missing: {covariate}")
    if not pd.api.types.is_numeric_dtype(frame[covariate]):
        raise ValueError(f"CUPED covariate {covariate} must be numeric")

    raw = estimate_metric_itt(frame, metric=metric, config=config)
    covariate_values = frame[covariate].astype(float)
    covariate_variance = float(covariate_values.var(ddof=1))
    if np.isclose(covariate_variance, 0.0):
        return replace(
            raw,
            theta=0.0,
            raw_standard_error=raw.standard_error,
            variance_reduction=0.0,
        )

    outcome = frame[metric.name].astype(float)
    theta = float(outcome.cov(covariate_values) / covariate_variance)
    adjusted_name = "__cuped_adjusted_outcome__"
    adjusted_frame = frame[[config.variant_column]].copy()
    adjusted_frame[adjusted_name] = outcome - theta * (covariate_values - covariate_values.mean())
    adjusted = estimate_metric_itt(
        adjusted_frame,
        metric=MetricSpec(adjusted_name, "continuous", metric.direction),
        config=config,
    )
    raw_variance = raw.standard_error**2
    adjusted_variance = adjusted.standard_error**2
    variance_reduction = 0.0 if raw_variance == 0 else 1 - adjusted_variance / raw_variance
    return replace(
        adjusted,
        metric=metric.name,
        kind=metric.kind,
        theta=theta,
        raw_standard_error=raw.standard_error,
        variance_reduction=variance_reduction,
    )


def estimate_segment_effects(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Estimate transparent primary effects for each predeclared segment cell."""

    validate_experiment(frame, config)
    rows: list[dict[str, Any]] = []
    for segment in config.segment_columns:
        values = sorted(frame[segment].drop_duplicates().tolist(), key=str)
        for value in values:
            subset = frame.loc[frame[segment] == value]
            counts = subset[config.variant_column].value_counts()
            control_units = int(counts.get(config.control_label, 0))
            treatment_units = int(counts.get(config.treatment_label, 0))
            role = "confirmatory" if segment in config.confirmatory_segments else "exploratory"
            row: dict[str, Any] = {
                "segment": segment,
                "value": str(value),
                "units": len(subset),
                "control_units": control_units,
                "treatment_units": treatment_units,
                "analysis_role": role,
                "low_support": len(subset) < config.minimum_segment_units,
                "multiplicity_alpha": np.nan,
                "holm_adjusted_p_value": np.nan,
                "interval_method": "pointwise",
                "significant": False,
            }
            if control_units >= 2 and treatment_units >= 2:
                estimate = estimate_metric_itt(
                    subset,
                    metric=config.primary,
                    config=config,
                )
                row.update(
                    {
                        "effect": estimate.effect,
                        "standard_error": estimate.standard_error,
                        "ci_lower": estimate.ci_lower,
                        "ci_upper": estimate.ci_upper,
                        "p_value": estimate.p_value,
                    }
                )
            else:
                row.update(
                    {
                        "effect": np.nan,
                        "standard_error": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                    }
                )
            rows.append(row)

    result = (
        pd.DataFrame(rows).sort_values(["segment", "value"], kind="stable").reset_index(drop=True)
    )
    confirmatory = result.index[result["analysis_role"] == "confirmatory"].tolist()
    ordered = sorted(
        confirmatory,
        key=lambda index: (
            result.at[index, "p_value"] if np.isfinite(result.at[index, "p_value"]) else math.inf,
            index,
        ),
    )
    tests = len(ordered)
    rejection_chain_open = True
    cumulative_adjusted_p = 0.0
    for rank, index in enumerate(ordered):
        threshold = config.alpha / (tests - rank)
        result.at[index, "multiplicity_alpha"] = threshold
        p_value = result.at[index, "p_value"]
        if np.isfinite(p_value):
            cumulative_adjusted_p = max(cumulative_adjusted_p, (tests - rank) * p_value)
            result.at[index, "holm_adjusted_p_value"] = min(1.0, cumulative_adjusted_p)
        rejects = bool(rejection_chain_open and np.isfinite(p_value) and p_value <= threshold)
        result.at[index, "significant"] = rejects
        rejection_chain_open = rejection_chain_open and rejects
        standard_error = result.at[index, "standard_error"]
        if np.isfinite(standard_error):
            bonferroni_alpha = config.alpha / tests
            critical_value = float(norm.ppf(1 - bonferroni_alpha / 2))
            effect = result.at[index, "effect"]
            result.at[index, "ci_lower"] = effect - critical_value * standard_error
            result.at[index, "ci_upper"] = effect + critical_value * standard_error
            result.at[index, "interval_method"] = "bonferroni"
    return result


def analyze_experiment_frame(frame: pd.DataFrame, config: ExperimentConfig) -> ExperimentAnalysis:
    """Run integrity, raw, CUPED, and segment experiment analyses."""

    validate_experiment(frame, config)
    return ExperimentAnalysis(
        srm=sample_ratio_mismatch(frame, config),
        effects=estimate_itt(frame, config),
        cuped_effects=pd.DataFrame(
            [
                asdict(
                    estimate_cuped(
                        frame,
                        metric=config.primary,
                        covariate=config.cuped_covariate,
                        config=config,
                    )
                )
            ]
        ),
        heterogeneous_effects=estimate_segment_effects(frame, config),
    )


def _write_experiment_outputs(
    output: Path,
    frame: pd.DataFrame,
    analysis: ExperimentAnalysis,
    config: ExperimentConfig,
    evidence_tier: EvidenceTier,
    provenance: dict[str, str],
    manifest: dict[str, Any],
    config_bytes: bytes,
) -> None:
    from news_ctr.decisioning import make_decision, render_decision_report
    from news_ctr.visualization import write_effect_plot

    metrics = [metric.name for metric in _configured_metrics(config)]
    summary = (
        frame.groupby(config.variant_column, sort=False)
        .agg(
            units=(config.unit_column, "size"),
            **{f"{name}_mean": (name, "mean") for name in metrics},
        )
        .reindex([config.control_label, config.treatment_label])
        .reset_index()
    )
    summary.to_csv(output / "experiment_summary.csv", index=False, float_format="%.10f")
    (output / "srm.json").write_text(
        json.dumps(asdict(analysis.srm), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "experiment_config.json").write_bytes(config_bytes)
    analysis.effects.to_csv(output / "effects.csv", index=False, float_format="%.10f")
    analysis.cuped_effects.to_csv(output / "cuped_effects.csv", index=False, float_format="%.10f")
    analysis.heterogeneous_effects.to_csv(
        output / "heterogeneous_effects.csv", index=False, float_format="%.10f"
    )
    decision = make_decision(analysis, config)
    (output / "decision_report.md").write_text(
        render_decision_report(
            decision,
            config,
            evidence_tier=evidence_tier,
            provenance=provenance,
        ),
        encoding="utf-8",
    )
    write_effect_plot(
        analysis.effects,
        output / "effects.png",
        confidence_level=1 - config.alpha,
    )


def _validate_experiment_metadata(
    metadata: dict[str, Any],
    frame: pd.DataFrame,
    config: ExperimentConfig,
    evidence_tier: EvidenceTier,
) -> None:
    if evidence_tier is not EvidenceTier.SYNTHETIC_RCT:
        raise ValueError(
            "experiment decisions require randomized evidence; observational tiers are rejected"
        )
    if metadata.get("randomization_unit") != config.unit_column:
        raise ValueError("experiment metadata randomization_unit does not match the config")
    if metadata.get("units") != len(frame):
        raise ValueError("experiment metadata units do not match the input frame")
    observed = {
        str(label): int(count)
        for label, count in frame[config.variant_column].value_counts().sort_index().items()
    }
    if metadata.get("allocation") != observed:
        raise ValueError("experiment metadata allocation does not match the input frame")


def run_experiment(input_path: Path, output: Path, config_path: Path) -> Path:
    """Analyze an experiment and atomically publish complete decision evidence."""

    input_path = Path(input_path)
    output = Path(output)
    if output.exists():
        raise ValueError(f"experiment output already exists: {output}")
    if not input_path.is_file():
        raise ValueError(f"experiment input does not exist: {input_path}")
    metadata_path = input_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"experiment evidence metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        evidence_tier = EvidenceTier(str(metadata["evidence_tier"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("experiment metadata has an invalid evidence_tier") from exc

    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    config = ExperimentConfig.from_json(config_path)
    frame = pd.read_parquet(input_path)
    validate_experiment(frame, config)
    _validate_experiment_metadata(metadata, frame, config, evidence_tier)
    analysis = analyze_experiment_frame(frame, config)
    provenance = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evidence_tier": evidence_tier.value,
        "randomization_unit": config.unit_column,
        "units": len(frame),
        "allocation": metadata["allocation"],
        "analysis_contract": {
            "alpha": config.alpha,
            "primary_metric": config.primary.name,
            "guardrail_metrics": [metric.name for metric in config.guardrails],
            "cuped_covariate": config.cuped_covariate,
            "practical_threshold": config.practical_threshold,
        },
        "input": {"sha256": provenance["input_sha256"]},
        "metadata": {"sha256": provenance["metadata_sha256"]},
        "config": {"sha256": provenance["config_sha256"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging-"
    ) as temporary_directory:
        staging = Path(temporary_directory)
        _write_experiment_outputs(
            staging,
            frame,
            analysis,
            config,
            evidence_tier,
            provenance,
            manifest,
            config_bytes,
        )
        os.replace(staging, output)
    return output
