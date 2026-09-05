from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from news_ctr.experiment_data import write_synthetic_experiment


def test_synthetic_experiment_is_deterministic_and_user_randomized(tmp_path: Path) -> None:
    first = write_synthetic_experiment(tmp_path / "a.parquet", seed=42, users=2_000)
    second = write_synthetic_experiment(tmp_path / "b.parquet", seed=42, users=2_000)

    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))
    frame = pd.read_parquet(first)
    assert frame["user_id"].is_unique
    assert frame.columns.tolist() == [
        "user_id",
        "variant",
        "pre_ctr",
        "click",
        "dwell_seconds",
        "latency_ms",
        "device_type",
        "history_segment",
    ]
    assert set(frame["variant"]) == {"control", "treatment"}

    first_metadata = json.loads((tmp_path / "a.metadata.json").read_text(encoding="utf-8"))
    second_metadata = json.loads((tmp_path / "b.metadata.json").read_text(encoding="utf-8"))
    assert first_metadata == second_metadata
    assert first_metadata["evidence_tier"] == "synthetic-rct"
    assert first_metadata["randomization_unit"] == "user_id"
    assert first_metadata["units"] == 2_000
    assert first_metadata["declared_effects"]["click_absolute"] > 0


@pytest.mark.parametrize("users", [0, 1, -5])
def test_synthetic_experiment_rejects_too_few_users(tmp_path: Path, users: int) -> None:
    with pytest.raises(ValueError, match="users"):
        write_synthetic_experiment(tmp_path / "experiment.parquet", seed=42, users=users)


def test_synthetic_experiment_refuses_overwrite(tmp_path: Path) -> None:
    path = write_synthetic_experiment(tmp_path / "experiment.parquet", seed=42, users=100)
    with pytest.raises(ValueError, match="already exists"):
        write_synthetic_experiment(path, seed=42, users=100)
