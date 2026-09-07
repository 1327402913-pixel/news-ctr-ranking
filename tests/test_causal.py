from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_ctr.causal import CausalConfig, validate_market_panel
from news_ctr.quasi_data import write_synthetic_market_panel


def _valid_payload() -> dict[str, object]:
    return {
        "active_column": "policy_active",
        "alpha": 0.05,
        "balance_columns": ["market_size_index", "candidate_exposures", "ctr"],
        "business_exposure_scale": 1_000_000,
        "click_column": "clicks",
        "event_window": [-12, 8],
        "exposure_column": "candidate_exposures",
        "group_column": "treated_market",
        "maximum_missing_market_week_share": 0.0,
        "minimum_control_markets": 8,
        "minimum_markets": 20,
        "minimum_post_weeks": 4,
        "minimum_pre_weeks": 8,
        "minimum_treated_markets": 8,
        "outcome_column": "ctr",
        "outcome_kind": "rate",
        "placebo_week": -8,
        "practical_threshold": 0.002,
        "reference_week": -1,
        "rollout_week": 0,
        "time_column": "relative_week",
        "unit_column": "market_id",
    }


def _write_config(path: Path, payload: dict[str, object] | None = None) -> Path:
    path.write_text(json.dumps(payload or _valid_payload()), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> CausalConfig:
    return CausalConfig.from_json(_write_config(tmp_path / "causal.json"))


@pytest.fixture
def panel(tmp_path: Path) -> pd.DataFrame:
    path = write_synthetic_market_panel(
        tmp_path / "panel.parquet", seed=42, markets=60, pre_weeks=20, post_weeks=12
    )
    return pd.read_parquet(path)


def test_causal_config_loads_the_predeclared_analysis_contract(tmp_path: Path) -> None:
    """Catches configuration fields being dropped, renamed, or silently defaulted."""

    config = CausalConfig.from_json(_write_config(tmp_path / "causal.json"))

    assert config.unit_column == "market_id"
    assert config.time_column == "relative_week"
    assert config.group_column == "treated_market"
    assert config.active_column == "policy_active"
    assert config.exposure_column == "candidate_exposures"
    assert config.click_column == "clicks"
    assert config.outcome_column == "ctr"
    assert config.outcome_kind == "rate"
    assert config.balance_columns == ("market_size_index", "candidate_exposures", "ctr")
    assert config.rollout_week == 0
    assert config.reference_week == -1
    assert config.event_window == (-12, 8)
    assert config.placebo_week == -8
    assert config.alpha == 0.05
    assert config.practical_threshold == 0.002
    assert config.business_exposure_scale == 1_000_000


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_week", 0, "reference_week"),
        ("event_window", [-1, 8], "event_window"),
        ("placebo_week", 0, "placebo_week"),
        ("alpha", 0.0, "alpha"),
        ("practical_threshold", -0.1, "practical_threshold"),
        ("business_exposure_scale", 0, "business_exposure_scale"),
        ("minimum_markets", 3, "minimum_markets"),
        ("minimum_treated_markets", 1, "minimum_treated_markets"),
        ("minimum_control_markets", 1, "minimum_control_markets"),
        ("minimum_pre_weeks", 1, "minimum_pre_weeks"),
        ("minimum_post_weeks", 1, "minimum_post_weeks"),
        ("maximum_missing_market_week_share", -0.01, "missing"),
        ("outcome_kind", "continuous", "outcome_kind"),
    ],
)
def test_causal_config_rejects_contracts_that_change_identification(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Catches invalid analysis choices reaching the estimator."""

    payload = _valid_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        CausalConfig.from_json(_write_config(tmp_path / "invalid.json", payload))


def test_causal_config_rejects_unknown_fields(tmp_path: Path) -> None:
    """Catches misspelled configuration keys being silently ignored."""

    payload = _valid_payload()
    payload["alphaa"] = 0.05
    with pytest.raises(ValueError, match="unknown config fields: alphaa"):
        CausalConfig.from_json(_write_config(tmp_path / "invalid.json", payload))


def test_market_panel_validation_returns_a_json_safe_integrity_summary(
    panel: pd.DataFrame, config: CausalConfig
) -> None:
    """Catches validation succeeding without publishing the design's audit fields."""

    diagnostics = validate_market_panel(panel, config)

    assert diagnostics == {
        "control_markets": 30,
        "integrity_passed": True,
        "markets": 60,
        "missing_market_week_share": 0.0,
        "post_observations": 720,
        "post_weeks": 12,
        "pre_observations": 1_200,
        "pre_weeks": 20,
        "rows": 1_920,
        "total_exposure": int(panel["candidate_exposures"].sum()),
        "treated_markets": 30,
        "week_max": 11,
        "week_min": -20,
    }
    json.dumps(diagnostics)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="policy_active"), "missing configured columns"),
        (lambda frame: frame.iloc[0:0], "must not be empty"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (
            lambda frame: frame.assign(market_id=np.where(frame.index == 0, None, frame.market_id)),
            "market_id",
        ),
        (
            lambda frame: frame.assign(
                treated_market=np.where(
                    frame.index == 0,
                    1 - frame.treated_market,
                    frame.treated_market,
                )
            ),
            "treatment membership",
        ),
        (lambda frame: frame.assign(policy_active=1 - frame.policy_active), "policy_active"),
        (
            lambda frame: frame.assign(
                relative_week=np.where(frame.index == 0, -19.5, frame.relative_week)
            ),
            "integer",
        ),
        (
            lambda frame: frame.assign(
                candidate_exposures=np.where(frame.index == 0, 0, frame.candidate_exposures)
            ),
            "candidate_exposures",
        ),
        (lambda frame: frame.assign(clicks=np.where(frame.index == 0, -1, frame.clicks)), "clicks"),
        (lambda frame: frame.assign(ctr=np.where(frame.index == 0, 0.9, frame.ctr)), "ctr"),
        (
            lambda frame: frame.assign(
                market_size_index=np.where(frame.index == 0, np.inf, frame.market_size_index)
            ),
            "finite",
        ),
        (lambda frame: frame.drop(index=frame.index[0]), "missing market-week"),
    ],
)
def test_market_panel_validation_rejects_broken_design_contracts(
    panel: pd.DataFrame,
    config: CausalConfig,
    mutation,
    message: str,
) -> None:
    """Catches malformed panel state being interpreted as estimable evidence."""

    with pytest.raises(ValueError, match=message):
        validate_market_panel(mutation(panel.copy()), config)


def test_market_panel_validation_enforces_cluster_and_time_minima(
    panel: pd.DataFrame, config: CausalConfig, tmp_path: Path
) -> None:
    """Catches under-supported arms or windows reaching clustered inference."""

    treated_ids = panel.loc[panel["treated_market"] == 1, "market_id"].unique()[:7]
    too_few_treated = panel.loc[
        (panel["treated_market"] == 0) | panel["market_id"].isin(treated_ids)
    ]
    with pytest.raises(ValueError, match="treated markets"):
        validate_market_panel(too_few_treated, config)

    short_window = panel.loc[panel["relative_week"].between(-7, 3)]
    permissive_payload = _valid_payload()
    permissive_payload["maximum_missing_market_week_share"] = 0.5
    short_config = CausalConfig.from_json(
        _write_config(tmp_path / "short.json", permissive_payload)
    )
    with pytest.raises(ValueError, match="pre-treatment weeks"):
        validate_market_panel(short_window, short_config)
