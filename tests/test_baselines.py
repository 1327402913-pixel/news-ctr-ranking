from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_ctr.baselines import fit_baseline, predict_baseline


def test_position_baseline_prefers_earlier_candidates() -> None:
    train = pd.DataFrame({"article_id": [1, 2], "candidate_position": [0, 1], "label": [0, 1]})

    model = fit_baseline("position", train)

    np.testing.assert_array_equal(predict_baseline(model, train), np.array([0.0, -1.0]))


def test_popularity_uses_smoothed_training_ctr_and_global_fallback() -> None:
    train = pd.DataFrame(
        {
            "article_id": [1, 1, 2, 2],
            "candidate_position": [0, 1, 0, 1],
            "label": [1, 1, 0, 0],
        }
    )
    valid = pd.DataFrame({"article_id": [1, 2, 999], "candidate_position": [0, 1, 2]})

    model = fit_baseline("popularity", train, alpha=2.0)
    scores = predict_baseline(model, valid)

    assert scores[0] > scores[2] > scores[1]
    assert scores[2] == pytest.approx(0.5)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf"), float("nan")])
def test_popularity_rejects_nonpositive_or_nonfinite_alpha(alpha: float) -> None:
    train = pd.DataFrame({"article_id": [1, 2], "candidate_position": [0, 1], "label": [1, 0]})

    with pytest.raises(ValueError, match="alpha must be positive and finite"):
        fit_baseline("popularity", train, alpha=alpha)


def test_fit_rejects_empty_nonbinary_or_incomplete_training_data() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        fit_baseline(
            "position",
            pd.DataFrame(columns=["article_id", "candidate_position", "label"]),
        )
    with pytest.raises(ValueError, match="binary"):
        fit_baseline(
            "popularity",
            pd.DataFrame({"article_id": [1, 2], "candidate_position": [0, 1], "label": [0, 2]}),
        )
    with pytest.raises(ValueError, match="missing required columns"):
        fit_baseline("position", pd.DataFrame({"candidate_position": [0], "label": [1]}))


def test_prediction_validates_columns_and_model_kind() -> None:
    train = pd.DataFrame({"article_id": [1, 2], "candidate_position": [0, 1], "label": [1, 0]})
    popularity = fit_baseline("popularity", train)
    position = fit_baseline("position", train)

    with pytest.raises(ValueError, match="missing required columns"):
        predict_baseline(popularity, pd.DataFrame({"candidate_position": [0]}))
    with pytest.raises(ValueError, match="missing required columns"):
        predict_baseline(position, pd.DataFrame({"article_id": [1]}))
    with pytest.raises(ValueError, match="baseline kind must be one of"):
        fit_baseline("random", train)
