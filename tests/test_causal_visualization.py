from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from news_ctr.causal_visualization import write_event_study_plot


def _event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "relative_week": [-2, 0, 1],
            "effect": [-0.001, 0.006, 0.007],
            "ci_lower": [-0.003, 0.003, 0.004],
            "ci_upper": [0.001, 0.009, 0.010],
        }
    )


def test_event_study_plot_is_deterministic_and_described(tmp_path: Path) -> None:
    """Catches unstable visual evidence or a plot without accessibility metadata."""

    first = write_event_study_plot(_event_frame(), tmp_path / "first.png")
    second = write_event_study_plot(_event_frame(), tmp_path / "second.png")

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size > 5_000
    with Image.open(first) as image:
        assert image.size == (900, 500)
        assert (
            image.info["Description"]
            == "Synthetic quasi-experiment event study with 95% confidence intervals"
        )


@pytest.mark.parametrize(
    ("frame", "confidence", "message"),
    [
        (_event_frame().drop(columns="ci_upper"), 0.95, "missing"),
        (_event_frame().iloc[0:0], 0.95, "empty"),
        (_event_frame(), 0.0, "confidence"),
        (_event_frame(), 1.0, "confidence"),
    ],
)
def test_event_study_plot_rejects_invalid_inputs(
    tmp_path: Path,
    frame: pd.DataFrame,
    confidence: float,
    message: str,
) -> None:
    """Catches malformed visual evidence being written as if analysis succeeded."""

    with pytest.raises(ValueError, match=message):
        write_event_study_plot(frame, tmp_path / "event.png", confidence_level=confidence)


def test_event_study_plot_refuses_to_overwrite(tmp_path: Path) -> None:
    """Catches reruns silently replacing a published evidence artifact."""

    destination = tmp_path / "event.png"
    destination.write_bytes(b"existing")

    with pytest.raises(ValueError, match="already exists"):
        write_event_study_plot(_event_frame(), destination)

    assert destination.read_bytes() == b"existing"
