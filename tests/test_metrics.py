from __future__ import annotations

import numpy as np
import pytest

from news_ctr.metrics import ranking_metrics


def test_ranking_metrics_match_hand_calculated_impressions() -> None:
    """Catches calculating global classification metrics instead of group rankings."""
    labels = np.array([1, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.1])
    groups = np.array(["a", "a", "a", "b", "b", "b"])

    result = ranking_metrics(labels, scores, groups, ks=(5, 10))

    assert result["mrr"] == pytest.approx(0.75)
    assert result["ndcg@5"] == pytest.approx((1.0 + 1 / np.log2(3)) / 2)
    assert result["ndcg@10"] == pytest.approx((1.0 + 1 / np.log2(3)) / 2)
    assert result["group_auc"] == pytest.approx(0.75)
    assert result["groups"] == 2
    assert result["auc_groups"] == 2
    assert result["skipped_auc_groups"] == 0


def test_metrics_are_invariant_to_row_order_within_groups() -> None:
    """Catches using candidate input position as predicted rank."""
    labels = np.array([1, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.1])
    groups = np.array([1, 1, 1, 2, 2, 2])
    shuffled = np.array([2, 0, 1, 5, 4, 3])

    original = ranking_metrics(labels, scores, groups)
    reordered = ranking_metrics(labels[shuffled], scores[shuffled], groups[shuffled])

    assert reordered == original


def test_group_auc_skips_single_class_impressions() -> None:
    """Catches crashes or misleading AUC values for groups without both classes."""
    result = ranking_metrics(
        labels=[1, 0, 0, 0],
        scores=[0.8, 0.2, 0.7, 0.1],
        groups=[1, 1, 2, 2],
    )

    assert result["group_auc"] == pytest.approx(1.0)
    assert result["auc_groups"] == 1
    assert result["skipped_auc_groups"] == 1
    assert result["mrr"] == pytest.approx(0.5)


def test_group_auc_is_null_when_no_impression_has_both_classes() -> None:
    """Catches emitting NaN that cannot be stored in strict JSON artifacts."""
    result = ranking_metrics(
        labels=[0, 0, 1, 1],
        scores=[0.8, 0.2, 0.7, 0.1],
        groups=[1, 1, 2, 2],
    )

    assert result["group_auc"] is None
    assert result["auc_groups"] == 0
    assert result["skipped_auc_groups"] == 2


@pytest.mark.parametrize(
    ("labels", "scores", "groups", "message"),
    [
        ([1, 0], [0.9], [1, 1], "same length"),
        ([1, 2], [0.9, 0.1], [1, 1], "binary"),
        ([], [], [], "at least one candidate"),
    ],
)
def test_metrics_reject_invalid_inputs(labels, scores, groups, message: str) -> None:
    """Catches silently truncating or accepting invalid relevance labels."""
    with pytest.raises(ValueError, match=message):
        ranking_metrics(labels, scores, groups)
