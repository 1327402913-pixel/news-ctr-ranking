"""Contracts and estimators for market-level quasi-experimental impact analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
