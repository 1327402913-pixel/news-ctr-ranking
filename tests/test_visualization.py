from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL.Image")

from PIL import Image

from news_ctr.visualization import write_effect_plot


@pytest.fixture
def effects() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["click", "dwell_seconds", "latency_ms"],
            "effect": [0.018, 0.3, 3.0],
            "ci_lower": [0.009, -0.4, 2.1],
            "ci_upper": [0.027, 1.0, 3.9],
        }
    )


def test_effect_plot_is_reproducible(tmp_path: Path, effects: pd.DataFrame) -> None:
    first = write_effect_plot(effects, tmp_path / "a.png")
    second = write_effect_plot(effects, tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size > 5_000


def test_effect_plot_validates_schema_and_destination(
    tmp_path: Path, effects: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="columns"):
        write_effect_plot(effects.drop(columns=["ci_upper"]), tmp_path / "missing.png")
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        write_effect_plot(effects, existing)


def test_effect_plot_labels_non_default_confidence_level(
    tmp_path: Path, effects: pd.DataFrame
) -> None:
    output = write_effect_plot(effects, tmp_path / "effect-90.png", confidence_level=0.90)
    with Image.open(output) as rendered:
        assert rendered.info["Description"] == "Treatment effects with 90% confidence intervals"
