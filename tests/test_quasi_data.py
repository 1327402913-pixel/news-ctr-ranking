from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from news_ctr.quasi_data import write_synthetic_market_panel


def test_market_panel_is_deterministic_and_preserves_the_rollout_contract(
    tmp_path: Path,
) -> None:
    """Catches unstable fixtures or treatment flags that disagree with rollout timing."""

    first = write_synthetic_market_panel(
        tmp_path / "a.parquet", seed=42, markets=60, pre_weeks=20, post_weeks=12
    )
    second = write_synthetic_market_panel(
        tmp_path / "b.parquet", seed=42, markets=60, pre_weeks=20, post_weeks=12
    )

    first_frame = pd.read_parquet(first)
    second_frame = pd.read_parquet(second)
    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert first_frame.columns.tolist() == [
        "market_id",
        "relative_week",
        "treated_market",
        "policy_active",
        "candidate_exposures",
        "clicks",
        "ctr",
        "market_size_index",
    ]
    assert len(first_frame) == 60 * 32
    assert not first_frame[["market_id", "relative_week"]].duplicated().any()
    assert first_frame.groupby("market_id")["treated_market"].nunique().eq(1).all()
    assert first_frame.groupby("market_id")["treated_market"].first().sum() == 30
    expected_active = first_frame["treated_market"] * (first_frame["relative_week"] >= 0).astype(
        "int8"
    )
    pd.testing.assert_series_equal(first_frame["policy_active"], expected_active, check_names=False)
    assert (first_frame["candidate_exposures"] > 0).all()
    assert first_frame["clicks"].between(0, first_frame["candidate_exposures"]).all()
    assert (first_frame["ctr"] == first_frame["clicks"] / first_frame["candidate_exposures"]).all()

    first_metadata = json.loads(first.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    second_metadata = json.loads(second.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert first_metadata == second_metadata
    assert first_metadata == {
        "evidence_tier": "synthetic-quasi-experiment",
        "known_effects": {"ctr_absolute": 0.006},
        "markets": 60,
        "panel_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "post_weeks": 12,
        "pre_weeks": 20,
        "rows": 1_920,
        "seed": 42,
        "treated_markets": 30,
    }


@pytest.mark.parametrize(
    ("markets", "pre_weeks", "post_weeks", "message"),
    [
        (1, 20, 12, "markets"),
        (5, 20, 12, "even"),
        (60, 0, 12, "pre_weeks"),
        (60, 20, 0, "post_weeks"),
    ],
)
def test_market_panel_rejects_dimensions_that_cannot_support_two_period_comparison(
    tmp_path: Path,
    markets: int,
    pre_weeks: int,
    post_weeks: int,
    message: str,
) -> None:
    """Catches invalid fixtures that cannot identify a two-arm pre/post contrast."""

    with pytest.raises(ValueError, match=message):
        write_synthetic_market_panel(
            tmp_path / "panel.parquet",
            seed=42,
            markets=markets,
            pre_weeks=pre_weeks,
            post_weeks=post_weeks,
        )


def test_market_panel_refuses_to_overwrite_data_or_its_sidecar(tmp_path: Path) -> None:
    """Catches a rerun silently replacing evidence or separating it from its metadata."""

    output = write_synthetic_market_panel(
        tmp_path / "panel.parquet", seed=42, markets=10, pre_weeks=3, post_weeks=2
    )
    with pytest.raises(ValueError, match="already exists"):
        write_synthetic_market_panel(output, seed=42, markets=10, pre_weeks=3, post_weeks=2)

    data_only = tmp_path / "metadata-blocked.parquet"
    data_only.with_suffix(".metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        write_synthetic_market_panel(data_only, seed=42, markets=10, pre_weeks=3, post_weeks=2)


def test_market_panel_removes_a_partial_sidecar_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches metadata write failures leaving a retry-blocking partial sidecar."""

    output = tmp_path / "panel.parquet"
    metadata_path = output.with_suffix(".metadata.json")
    original_write_text = Path.write_text

    def fail_metadata_write(path: Path, data: str, **kwargs):
        if path == metadata_path:
            original_write_text(path, "partial", encoding="utf-8")
            raise OSError("injected metadata write failure")
        return original_write_text(path, data, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata_write)

    with pytest.raises(OSError, match="injected metadata write failure"):
        write_synthetic_market_panel(
            output,
            seed=42,
            markets=10,
            pre_weeks=3,
            post_weeks=2,
        )

    assert not output.exists()
    assert not metadata_path.exists()
