"""Impression-grouped ranking metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANKING_METRICS = ("group_auc", "mrr", "ndcg@5", "ndcg@10")


def ranking_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    ks: Iterable[int] = (5, 10),
) -> dict[str, float | int | None]:
    """Calculate mean ranking quality within each impression."""

    y = np.asarray(labels)
    predictions = np.asarray(scores, dtype=float)
    impression_ids = np.asarray(groups)
    if not (len(y) == len(predictions) == len(impression_ids)):
        raise ValueError("labels, scores, and groups must have the same length")
    if len(y) == 0:
        raise ValueError("ranking metrics require at least one candidate")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be binary values in {0, 1}")
    if not np.isfinite(predictions).all():
        raise ValueError("scores must be finite")
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("metric cutoffs must be positive integers")

    frame = pd.DataFrame({"label": y.astype(int), "score": predictions, "group": impression_ids})
    reciprocal_ranks: list[float] = []
    ndcg_values: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    auc_values: list[float] = []

    for _, group in frame.groupby("group", sort=False, dropna=False):
        ordered = group.sort_values("score", ascending=False, kind="mergesort")
        ordered_labels = ordered["label"].to_numpy(dtype=int)
        relevant_positions = np.flatnonzero(ordered_labels == 1)
        reciprocal_ranks.append(
            0.0 if len(relevant_positions) == 0 else 1.0 / float(relevant_positions[0] + 1)
        )
        ideal_labels = np.sort(ordered_labels)[::-1]
        for cutoff in cutoffs:
            dcg = _dcg(ordered_labels[:cutoff])
            ideal = _dcg(ideal_labels[:cutoff])
            ndcg_values[cutoff].append(0.0 if ideal == 0 else dcg / ideal)
        if group["label"].nunique() == 2:
            auc_values.append(float(roc_auc_score(group["label"], group["score"])))

    group_count = int(frame["group"].nunique(dropna=False))
    result: dict[str, float | int | None] = {
        "group_auc": float(np.mean(auc_values)) if auc_values else None,
        "mrr": float(np.mean(reciprocal_ranks)),
        "groups": group_count,
        "auc_groups": len(auc_values),
        "skipped_auc_groups": group_count - len(auc_values),
    }
    result.update(
        {f"ndcg@{cutoff}": float(np.mean(values)) for cutoff, values in ndcg_values.items()}
    )
    return result


def _dcg(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(labels) + 2))
    return float(np.sum(labels / discounts))
