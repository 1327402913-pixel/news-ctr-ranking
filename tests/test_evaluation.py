from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_ctr.evaluation import (
    _bootstrap_indices,
    bootstrap_ranking_intervals,
    candidate_bias_diagnostics,
    run_logistic_ablations,
    segment_ranking_metrics,
)
from news_ctr.features import NewsFeatureBuilder
from news_ctr.metrics import RANKING_METRICS


def test_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.array([1, 0, 0, 1, 1, 0])
    groups = np.array([10, 10, 20, 20, 30, 30])
    scores = {
        "good": np.array([0.9, 0.1, 0.1, 0.9, 0.8, 0.2]),
        "bad": np.array([0.1, 0.9, 0.9, 0.1, 0.2, 0.8]),
    }

    first = bootstrap_ranking_intervals(labels, scores, groups, samples=25, seed=7)
    second = bootstrap_ranking_intervals(labels, scores, groups, samples=25, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["model"]) == {"good", "bad"}
    assert set(first["metric"]) == set(RANKING_METRICS)
    assert (first["lower"] <= first["mean"]).all()
    assert (first["mean"] <= first["upper"]).all()
    assert first["requested_samples"].eq(25).all()
    assert first["valid_samples"].eq(25).all()
    means = first.pivot(index="metric", columns="model", values="mean")
    assert (means["good"] > means["bad"]).all()


def test_bootstrap_relabels_duplicate_group_draws() -> None:
    class FixedRng:
        def integers(self, low: int, high: int, size: int) -> np.ndarray:
            assert (low, high, size) == (0, 3, 3)
            return np.array([0, 0, 2])

    rows, remapped = _bootstrap_indices(np.array([10, 10, 20, 20, 30, 30]), FixedRng())

    np.testing.assert_array_equal(rows, np.array([0, 1, 0, 1, 4, 5]))
    np.testing.assert_array_equal(remapped, np.array([0, 0, 1, 1, 2, 2]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 1}, "at least 2"),
        ({"confidence": 0.0}, "between 0 and 1"),
        ({"confidence": 1.0}, "between 0 and 1"),
    ],
)
def test_bootstrap_rejects_invalid_configuration(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_ranking_intervals([1, 0], {"model": [0.9, 0.1]}, [1, 1], **kwargs)


def test_bootstrap_rejects_empty_misaligned_or_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        bootstrap_ranking_intervals([1, 0], {}, [1, 1])
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ranking_intervals([1, 0], {"model": [0.9]}, [1, 1])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ranking_intervals([1, 0], {"model": [0.9, np.nan]}, [1, 1])
    with pytest.raises(ValueError, match="binary"):
        bootstrap_ranking_intervals([1, 2], {"model": [0.9, 0.1]}, [1, 1])


def _ablation_features(rows: int) -> pd.DataFrame:
    values = np.arange(rows * len(NewsFeatureBuilder.feature_names_), dtype=float)
    return pd.DataFrame(values.reshape(rows, -1), columns=NewsFeatureBuilder.feature_names_)


def test_ablations_use_declared_feature_subsets() -> None:
    X_train = _ablation_features(8)
    X_valid = _ablation_features(6) + 0.5
    y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    train_groups = np.repeat([10, 20, 30, 40], 2)
    y_valid = np.array([1, 0, 0, 1, 1, 0])
    valid_groups = np.repeat([50, 60, 70], 2)

    table = run_logistic_ablations(
        X_train,
        y_train,
        train_groups,
        X_valid,
        y_valid,
        valid_groups,
        seed=3,
    )

    assert table["ablation"].tolist() == [
        "context_only",
        "no_position",
        "no_semantic",
        "no_personalization",
        "full",
    ]
    assert (
        table.loc[table["ablation"] == "no_position", "feature_count"].item()
        == X_train.shape[1] - 1
    )
    assert set(RANKING_METRICS) <= set(table.columns)


def _segment_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    candidates = pd.DataFrame(
        {
            "impression_id": np.repeat([1, 2, 3], 2),
            "label": [1, 0, 0, 1, 1, 0],
            "device_type": [1, 1, 2, 2, 1, 1],
            "candidate_count": [2, 2, 7, 7, 12, 12],
            "candidate_position": [0, 1, 0, 1, 0, 1],
        }
    )
    scores = np.array([0.9, 0.1, 0.2, 0.8, 0.7, 0.3])
    features = pd.DataFrame(
        {
            "history_length": [0, 0, 3, 3, 8, 8],
            "publication_age_hours": [3, 30, 80, 169, 12, 48],
        }
    )
    return candidates, scores, features


def test_segment_metrics_flag_low_support_without_splitting_impressions() -> None:
    candidates, scores, features = _segment_inputs()

    result = segment_ranking_metrics(candidates, scores, features, min_impressions=5)

    assert {"device_type", "candidate_count", "history_length"} <= set(result["segment"])
    assert result["low_support"].dtype == bool
    assert result["low_support"].all()
    assert result.groupby("segment")["impressions"].sum().eq(3).all()
    assert result["candidates"].mod(2).eq(0).all()
    assert set(RANKING_METRICS) <= set(result.columns)


def test_segment_metrics_reject_nonconstant_impression_signals() -> None:
    candidates, scores, features = _segment_inputs()
    candidates.loc[1, "device_type"] = 9

    with pytest.raises(ValueError, match="constant within each impression"):
        segment_ranking_metrics(candidates, scores, features)

    candidates, scores, features = _segment_inputs()
    candidates.loc[1, "candidate_count"] = 4
    with pytest.raises(ValueError, match="candidate_count must be constant"):
        segment_ranking_metrics(candidates, scores, features)

    candidates, scores, features = _segment_inputs()
    features.loc[3, "history_length"] = 4
    with pytest.raises(ValueError, match="history_length must be constant"):
        segment_ranking_metrics(candidates, scores, features)


def test_candidate_diagnostics_are_click_rates_not_ranking_metrics() -> None:
    candidates, _, features = _segment_inputs()

    result = candidate_bias_diagnostics(candidates, features)

    assert set(result["diagnostic"]) == {"candidate_position", "article_freshness"}
    assert "ndcg@5" not in result.columns
    assert result.columns.tolist() == [
        "diagnostic",
        "bucket",
        "candidates",
        "clicks",
        "click_rate",
    ]
    unseen = result.loc[(result["diagnostic"] == "article_freshness") & (result["bucket"] == "7d+")]
    assert unseen["candidates"].item() == 1
