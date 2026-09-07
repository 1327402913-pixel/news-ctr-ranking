"""Contracts and estimators for market-level quasi-experimental impact analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ESTIMATE_COLUMNS = [
    "term",
    "effect",
    "standard_error",
    "statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
    "alpha",
    "treated_markets",
    "control_markets",
    "clusters",
    "observations",
    "total_exposure",
    "weighting",
    "covariance",
]

_CONFIG_FIELDS = {
    "active_column",
    "alpha",
    "balance_columns",
    "business_exposure_scale",
    "click_column",
    "event_window",
    "exposure_column",
    "group_column",
    "maximum_missing_market_week_share",
    "minimum_control_markets",
    "minimum_markets",
    "minimum_post_weeks",
    "minimum_pre_weeks",
    "minimum_treated_markets",
    "outcome_column",
    "outcome_kind",
    "placebo_week",
    "practical_threshold",
    "reference_week",
    "rollout_week",
    "time_column",
    "unit_column",
}


@dataclass(frozen=True)
class CausalConfig:
    """Immutable identification and reporting choices loaded before analysis."""

    unit_column: str
    time_column: str
    group_column: str
    active_column: str
    exposure_column: str
    click_column: str
    outcome_column: str
    outcome_kind: str
    balance_columns: tuple[str, ...]
    rollout_week: int
    reference_week: int
    event_window: tuple[int, int]
    placebo_week: int
    alpha: float
    practical_threshold: float
    minimum_markets: int
    minimum_treated_markets: int
    minimum_control_markets: int
    minimum_pre_weeks: int
    minimum_post_weeks: int
    maximum_missing_market_week_share: float
    business_exposure_scale: int

    @classmethod
    def from_json(cls, path: str | Path) -> CausalConfig:
        """Load and validate an exact causal-analysis JSON contract."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("causal config must be a JSON object")
        unknown = sorted(set(payload) - _CONFIG_FIELDS)
        missing = sorted(_CONFIG_FIELDS - set(payload))
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"missing config fields: {', '.join(missing)}")
        try:
            event_window = tuple(int(value) for value in payload["event_window"])
            config = cls(
                unit_column=str(payload["unit_column"]),
                time_column=str(payload["time_column"]),
                group_column=str(payload["group_column"]),
                active_column=str(payload["active_column"]),
                exposure_column=str(payload["exposure_column"]),
                click_column=str(payload["click_column"]),
                outcome_column=str(payload["outcome_column"]),
                outcome_kind=str(payload["outcome_kind"]),
                balance_columns=tuple(str(value) for value in payload["balance_columns"]),
                rollout_week=int(payload["rollout_week"]),
                reference_week=int(payload["reference_week"]),
                event_window=event_window,
                placebo_week=int(payload["placebo_week"]),
                alpha=float(payload["alpha"]),
                practical_threshold=float(payload["practical_threshold"]),
                minimum_markets=int(payload["minimum_markets"]),
                minimum_treated_markets=int(payload["minimum_treated_markets"]),
                minimum_control_markets=int(payload["minimum_control_markets"]),
                minimum_pre_weeks=int(payload["minimum_pre_weeks"]),
                minimum_post_weeks=int(payload["minimum_post_weeks"]),
                maximum_missing_market_week_share=float(
                    payload["maximum_missing_market_week_share"]
                ),
                business_exposure_scale=int(payload["business_exposure_scale"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("causal config contains an invalid value type") from exc
        validate_causal_config(config)
        return config


@dataclass(frozen=True)
class CausalAnalysis:
    """Complete in-memory result from one validated causal analysis."""

    did_estimate: pd.DataFrame
    event_study: pd.DataFrame
    placebo_estimate: pd.DataFrame
    balance: pd.DataFrame
    business_impact: pd.DataFrame
    diagnostics: dict[str, object]


def validate_causal_config(config: CausalConfig) -> None:
    """Reject internally inconsistent identification choices."""

    named_columns = (
        config.unit_column,
        config.time_column,
        config.group_column,
        config.active_column,
        config.exposure_column,
        config.click_column,
        config.outcome_column,
    )
    if any(not column for column in named_columns):
        raise ValueError("configured column names must not be empty")
    if len(set(named_columns)) != len(named_columns):
        raise ValueError("configured role columns must be unique")
    if not config.balance_columns or len(set(config.balance_columns)) != len(
        config.balance_columns
    ):
        raise ValueError("balance_columns must be non-empty and unique")
    if config.outcome_kind != "rate":
        raise ValueError("outcome_kind must be rate")
    if not 0 < config.alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if config.practical_threshold < 0:
        raise ValueError("practical_threshold must not be negative")
    if config.business_exposure_scale <= 0:
        raise ValueError("business_exposure_scale must be positive")
    if config.minimum_markets < 4:
        raise ValueError("minimum_markets must be at least four")
    if config.minimum_treated_markets < 2:
        raise ValueError("minimum_treated_markets must be at least two")
    if config.minimum_control_markets < 2:
        raise ValueError("minimum_control_markets must be at least two")
    if config.minimum_markets < (config.minimum_treated_markets + config.minimum_control_markets):
        raise ValueError("minimum_markets must cover both arm minima")
    if config.minimum_pre_weeks < 2:
        raise ValueError("minimum_pre_weeks must be at least two")
    if config.minimum_post_weeks < 2:
        raise ValueError("minimum_post_weeks must be at least two")
    if not 0 <= config.maximum_missing_market_week_share < 1:
        raise ValueError("maximum missing market-week share must be between zero and one")
    if len(config.event_window) != 2:
        raise ValueError("event_window must contain exactly two weeks")
    if config.reference_week >= config.rollout_week:
        raise ValueError("reference_week must precede rollout_week")
    window_start, window_end = config.event_window
    if not window_start < config.reference_week < config.rollout_week <= window_end:
        raise ValueError("event_window must include pre-period leads, reference, and rollout")
    if config.placebo_week >= config.rollout_week:
        raise ValueError("placebo_week must precede rollout_week")


def _require_numeric(frame: pd.DataFrame, columns: set[str]) -> None:
    for column in sorted(columns):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"configured field {column} must be numeric")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"configured fields must contain finite values: {column}")


def validate_market_panel(frame: pd.DataFrame, config: CausalConfig) -> dict[str, object]:
    """Validate a market-week panel and return JSON-safe integrity diagnostics."""

    required = {
        config.unit_column,
        config.time_column,
        config.group_column,
        config.active_column,
        config.exposure_column,
        config.click_column,
        config.outcome_column,
        *config.balance_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing configured columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("market panel must not be empty")
    if frame[config.unit_column].isna().any():
        raise ValueError(f"{config.unit_column} must not contain null values")
    if frame[config.group_column].isna().any():
        raise ValueError(f"{config.group_column} must not contain null values")
    if frame[[config.unit_column, config.time_column]].duplicated().any():
        raise ValueError("duplicate market-week rows")

    numeric = {
        config.time_column,
        config.group_column,
        config.active_column,
        config.exposure_column,
        config.click_column,
        config.outcome_column,
        *config.balance_columns,
    }
    _require_numeric(frame, numeric)
    weeks = frame[config.time_column].to_numpy(dtype=float)
    if not np.equal(weeks, np.floor(weeks)).all():
        raise ValueError(f"{config.time_column} must contain integer weeks")
    groups = set(frame[config.group_column].astype(int).unique())
    if (
        groups != {0, 1}
        or not np.equal(frame[config.group_column], frame[config.group_column].astype(int)).all()
    ):
        raise ValueError(f"{config.group_column} must contain exactly zero and one")
    membership_counts = frame.groupby(config.unit_column)[config.group_column].nunique()
    if not membership_counts.eq(1).all():
        raise ValueError("treatment membership must remain constant within each market")
    expected_active = frame[config.group_column].astype(int) * (
        frame[config.time_column] >= config.rollout_week
    ).astype(int)
    if not np.array_equal(frame[config.active_column].to_numpy(), expected_active.to_numpy()):
        raise ValueError(f"{config.active_column} disagrees with the configured rollout")

    exposures = frame[config.exposure_column].to_numpy(dtype=float)
    clicks = frame[config.click_column].to_numpy(dtype=float)
    if (exposures <= 0).any() or not np.equal(exposures, np.floor(exposures)).all():
        raise ValueError(f"{config.exposure_column} must contain positive integers")
    if (
        (clicks < 0).any()
        or (clicks > exposures).any()
        or not np.equal(clicks, np.floor(clicks)).all()
    ):
        raise ValueError(f"{config.click_column} must be integer counts within exposures")
    observed_rate = frame[config.outcome_column].to_numpy(dtype=float)
    if not np.isclose(observed_rate, clicks / exposures, atol=1e-12, rtol=0).all():
        raise ValueError(f"{config.outcome_column} must equal clicks divided by exposures")

    market_groups = frame.groupby(config.unit_column)[config.group_column].first()
    markets = len(market_groups)
    treated_markets = int((market_groups == 1).sum())
    control_markets = int((market_groups == 0).sum())
    if markets < config.minimum_markets:
        raise ValueError("market panel has fewer than minimum_markets")
    if treated_markets < config.minimum_treated_markets:
        raise ValueError("market panel has too few treated markets")
    if control_markets < config.minimum_control_markets:
        raise ValueError("market panel has too few control markets")

    integer_weeks = frame[config.time_column].astype(int)
    unique_weeks = np.sort(integer_weeks.unique())
    pre_weeks = int((unique_weeks < config.rollout_week).sum())
    post_weeks = int((unique_weeks >= config.rollout_week).sum())
    if pre_weeks < config.minimum_pre_weeks:
        raise ValueError("market panel has too few pre-treatment weeks")
    if post_weeks < config.minimum_post_weeks:
        raise ValueError("market panel has too few post-treatment weeks")
    expected_rows = markets * (int(unique_weeks[-1]) - int(unique_weeks[0]) + 1)
    missing_share = 1 - len(frame) / expected_rows
    if missing_share > config.maximum_missing_market_week_share and not math.isclose(
        missing_share, config.maximum_missing_market_week_share
    ):
        raise ValueError("missing market-week share exceeds the configured maximum")

    pre_mask = integer_weeks < config.rollout_week
    return {
        "control_markets": control_markets,
        "integrity_passed": True,
        "markets": markets,
        "missing_market_week_share": float(missing_share),
        "post_observations": int((~pre_mask).sum()),
        "post_weeks": post_weeks,
        "pre_observations": int(pre_mask.sum()),
        "pre_weeks": pre_weeks,
        "rows": len(frame),
        "total_exposure": int(frame[config.exposure_column].sum()),
        "treated_markets": treated_markets,
        "week_max": int(unique_weeks[-1]),
        "week_min": int(unique_weeks[0]),
    }


def _fit_fixed_effect_model(
    frame: pd.DataFrame,
    config: CausalConfig,
    treatment_columns: tuple[str, ...],
):
    """Fit an exposure-weighted two-way fixed-effect model with clustered covariance."""

    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra
        raise RuntimeError(
            "causal analysis requires the optional dependency group: pip install '.[causal]'"
        ) from exc

    market_categories = sorted(frame[config.unit_column].unique())
    week_categories = sorted(frame[config.time_column].unique())
    market_values = pd.Categorical(frame[config.unit_column], categories=market_categories)
    week_values = pd.Categorical(frame[config.time_column], categories=week_categories)
    market_dummies = pd.get_dummies(
        market_values,
        prefix="market",
        drop_first=True,
        dtype=float,
    )
    week_dummies = pd.get_dummies(
        week_values,
        prefix="week",
        drop_first=True,
        dtype=float,
    )
    market_dummies.index = frame.index
    week_dummies.index = frame.index
    design = pd.concat(
        [
            pd.Series(1.0, index=frame.index, name="const"),
            market_dummies,
            week_dummies,
            frame.loc[:, list(treatment_columns)].astype(float),
        ],
        axis=1,
    )
    design_values = design.to_numpy(dtype=float)
    if not np.isfinite(design_values).all():
        raise ValueError("fixed-effect design matrix contains non-finite values")
    if np.linalg.matrix_rank(design_values) != design_values.shape[1]:
        raise ValueError("fixed-effect design matrix is rank deficient")

    model = sm.WLS(
        frame[config.outcome_column].astype(float),
        design.astype(float),
        weights=frame[config.exposure_column].astype(float),
    )
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": frame[config.unit_column],
            "use_correction": True,
        },
    )
    inference = np.concatenate(
        [
            np.asarray(result.params),
            np.asarray(result.bse),
            np.asarray(result.tvalues),
            np.asarray(result.pvalues),
            np.asarray(result.conf_int(alpha=config.alpha)).ravel(),
        ]
    )
    if not np.isfinite(inference).all():
        raise ValueError("fixed-effect model produced non-finite inference")
    return result


def fit_difference_in_differences(frame: pd.DataFrame, config: CausalConfig) -> pd.DataFrame:
    """Estimate the rollout effect with weighted two-way fixed effects."""

    diagnostics = validate_market_panel(frame, config)
    result = _fit_fixed_effect_model(frame, config, (config.active_column,))
    interval = result.conf_int(alpha=config.alpha).loc[config.active_column]
    row = {
        "term": config.active_column,
        "effect": float(result.params[config.active_column]),
        "standard_error": float(result.bse[config.active_column]),
        "statistic": float(result.tvalues[config.active_column]),
        "p_value": float(result.pvalues[config.active_column]),
        "ci_lower": float(interval.iloc[0]),
        "ci_upper": float(interval.iloc[1]),
        "alpha": float(config.alpha),
        "treated_markets": int(diagnostics["treated_markets"]),
        "control_markets": int(diagnostics["control_markets"]),
        "clusters": int(diagnostics["markets"]),
        "observations": len(frame),
        "total_exposure": int(frame[config.exposure_column].sum()),
        "weighting": config.exposure_column,
        "covariance": f"cluster:{config.unit_column}",
    }
    return pd.DataFrame([row], columns=_ESTIMATE_COLUMNS)


def _event_term(week: int) -> str:
    direction = "m" if week < 0 else "p"
    return f"event_{direction}{abs(week)}"


def estimate_event_study(
    frame: pd.DataFrame, config: CausalConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate dynamic treatment effects and jointly test pre-treatment leads."""

    validate_market_panel(frame, config)
    window_start, window_end = config.event_window
    event_frame = frame.loc[frame[config.time_column].between(window_start, window_end)].copy()
    weeks = [week for week in range(window_start, window_end + 1) if week != config.reference_week]
    terms = [_event_term(week) for week in weeks]
    treated = event_frame[config.group_column].astype(float)
    for week, term in zip(weeks, terms, strict=True):
        event_frame[term] = treated * event_frame[config.time_column].eq(week).astype(float)

    result = _fit_fixed_effect_model(event_frame, config, tuple(terms))
    intervals = result.conf_int(alpha=config.alpha)
    rows = []
    for week, term in zip(weeks, terms, strict=True):
        rows.append(
            {
                "relative_week": week,
                "term": term,
                "effect": float(result.params[term]),
                "standard_error": float(result.bse[term]),
                "statistic": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "ci_lower": float(intervals.loc[term].iloc[0]),
                "ci_upper": float(intervals.loc[term].iloc[1]),
            }
        )
    event = pd.DataFrame(
        rows,
        columns=[
            "relative_week",
            "term",
            "effect",
            "standard_error",
            "statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
        ],
    )

    lead_terms = [_event_term(week) for week in weeks if week < config.rollout_week]
    lead_effects = result.params.loc[lead_terms].to_numpy(dtype=float)
    lead_covariance = result.cov_params().loc[lead_terms, lead_terms].to_numpy(dtype=float)
    if np.linalg.matrix_rank(lead_covariance) != len(lead_terms):
        raise ValueError("pre-treatment covariance matrix is rank deficient")
    statistic = float(lead_effects @ np.linalg.solve(lead_covariance, lead_effects))
    try:
        from scipy.stats import chi2
    except ImportError as exc:  # pragma: no cover - installed by the causal extra
        raise RuntimeError(
            "causal analysis requires the optional dependency group: pip install '.[causal]'"
        ) from exc
    p_value = float(chi2.sf(statistic, len(lead_terms)))
    parallel = {
        "statistic": statistic,
        "degrees_of_freedom": len(lead_terms),
        "p_value": p_value,
        "lead_terms": len(lead_terms),
        "alpha": float(config.alpha),
        "passed": bool(p_value >= config.alpha),
    }
    return event, parallel


def estimate_placebo(
    frame: pd.DataFrame, config: CausalConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate a fake rollout using only observations before the real rollout."""

    integrity = validate_market_panel(frame, config)
    placebo_frame = frame.loc[frame[config.time_column] < config.rollout_week].copy()
    term = "placebo_policy_active"
    placebo_frame[term] = placebo_frame[config.group_column].astype(int) * (
        placebo_frame[config.time_column] >= config.placebo_week
    ).astype(int)
    if placebo_frame[term].nunique() != 2:
        raise ValueError("placebo window does not identify a fake rollout effect")

    result = _fit_fixed_effect_model(placebo_frame, config, (term,))
    interval = result.conf_int(alpha=config.alpha).loc[term]
    row = {
        "term": term,
        "effect": float(result.params[term]),
        "standard_error": float(result.bse[term]),
        "statistic": float(result.tvalues[term]),
        "p_value": float(result.pvalues[term]),
        "ci_lower": float(interval.iloc[0]),
        "ci_upper": float(interval.iloc[1]),
        "alpha": float(config.alpha),
        "treated_markets": int(integrity["treated_markets"]),
        "control_markets": int(integrity["control_markets"]),
        "clusters": int(integrity["markets"]),
        "observations": len(placebo_frame),
        "total_exposure": int(placebo_frame[config.exposure_column].sum()),
        "weighting": config.exposure_column,
        "covariance": f"cluster:{config.unit_column}",
    }
    estimate = pd.DataFrame([row], columns=_ESTIMATE_COLUMNS)
    contains_zero = bool(row["ci_lower"] <= 0 <= row["ci_upper"])
    diagnostics = {
        "alpha": float(config.alpha),
        "ci_contains_zero": contains_zero,
        "placebo_week": int(config.placebo_week),
        "passed": contains_zero,
    }
    return estimate, diagnostics


def summarize_preperiod_balance(frame: pd.DataFrame, config: CausalConfig) -> pd.DataFrame:
    """Compare treated and control markets using pre-period market-level means."""

    validate_market_panel(frame, config)
    preperiod = frame.loc[frame[config.time_column] < config.rollout_week]
    market_means = (
        preperiod.groupby(config.unit_column, sort=True)
        .agg(
            {
                config.group_column: "first",
                **{column: "mean" for column in config.balance_columns},
            }
        )
        .reset_index()
    )
    treated = market_means.loc[market_means[config.group_column] == 1]
    control = market_means.loc[market_means[config.group_column] == 0]
    rows = []
    for metric in config.balance_columns:
        treated_values = treated[metric].to_numpy(dtype=float)
        control_values = control[metric].to_numpy(dtype=float)
        treated_mean = float(treated_values.mean())
        control_mean = float(control_values.mean())
        pooled_variance = (
            float(treated_values.var(ddof=1)) + float(control_values.var(ddof=1))
        ) / 2
        denominator = math.sqrt(pooled_variance)
        if denominator == 0:
            standardized_difference = 0.0 if treated_mean == control_mean else math.inf
        else:
            standardized_difference = (treated_mean - control_mean) / denominator
        if not math.isfinite(standardized_difference):
            raise ValueError(f"balance metric {metric} has zero pooled variance")
        rows.append(
            {
                "metric": metric,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "standardized_mean_difference": float(standardized_difference),
                "absolute_standardized_mean_difference": abs(float(standardized_difference)),
                "treated_markets": len(treated_values),
                "control_markets": len(control_values),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "treated_mean",
            "control_mean",
            "standardized_mean_difference",
            "absolute_standardized_mean_difference",
            "treated_markets",
            "control_markets",
        ],
    )


def translate_business_impact(did_estimate: pd.DataFrame, config: CausalConfig) -> pd.DataFrame:
    """Translate an absolute CTR effect into clicks at a declared exposure scale."""

    required = {"effect", "ci_lower", "ci_upper"}
    missing = sorted(required - set(did_estimate.columns))
    if missing or len(did_estimate) != 1:
        detail = f"; missing columns: {', '.join(missing)}" if missing else ""
        raise ValueError(f"DiD estimate must contain exactly one complete row{detail}")
    row = did_estimate.iloc[0]
    values = np.array([row["effect"], row["ci_lower"], row["ci_upper"]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("DiD estimate contains non-finite impact values")
    scale = config.business_exposure_scale
    return pd.DataFrame(
        [
            {
                "scale_exposures": scale,
                "incremental_clicks": float(row["effect"] * scale),
                "ci_lower_clicks": float(row["ci_lower"] * scale),
                "ci_upper_clicks": float(row["ci_upper"] * scale),
            }
        ],
        columns=[
            "scale_exposures",
            "incremental_clicks",
            "ci_lower_clicks",
            "ci_upper_clicks",
        ],
    )


def analyze_causal_frame(frame: pd.DataFrame, config: CausalConfig) -> CausalAnalysis:
    """Run all declared estimators against one structurally validated panel."""

    integrity = validate_market_panel(frame, config)
    did_estimate = fit_difference_in_differences(frame, config)
    event_study, parallel_trends = estimate_event_study(frame, config)
    placebo_estimate, placebo = estimate_placebo(frame, config)
    balance = summarize_preperiod_balance(frame, config)
    business_impact = translate_business_impact(did_estimate, config)
    return CausalAnalysis(
        did_estimate=did_estimate,
        event_study=event_study,
        placebo_estimate=placebo_estimate,
        balance=balance,
        business_impact=business_impact,
        diagnostics={
            "integrity": integrity,
            "parallel_trends": parallel_trends,
            "placebo": placebo,
        },
    )
