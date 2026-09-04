"""Uncertainty estimates for impression-grouped ranking metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd

from news_ctr.features import FEATURE_GROUPS, select_feature_names
from news_ctr.metrics import RANKING_METRICS, ranking_metrics
from news_ctr.models import fit_model, predict_scores

ABLATIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "context_only": {"include_groups": ("context",)},
    "no_position": {"exclude_features": ("candidate_position",)},
    "no_semantic": {"exclude_groups": ("semantic",)},
    "no_personalization": {"exclude_groups": ("personalization",)},
    "full": {},
}


class _IntegerSampler(Protocol):
    def integers(self, low: int, high: int, size: int) -> np.ndarray: ...


def _bootstrap_indices(groups: np.ndarray, rng: _IntegerSampler) -> tuple[np.ndarray, np.ndarray]:
    """Draw whole impressions and assign each occurrence a fresh group ID."""

    unique_groups = pd.unique(groups)
    drawn = rng.integers(0, len(unique_groups), size=len(unique_groups))
    row_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for occurrence, group_index in enumerate(drawn):
        selected = unique_groups[group_index]
        mask = pd.isna(groups) if pd.isna(selected) else groups == selected
        rows = np.flatnonzero(mask)
        row_parts.append(rows)
        group_parts.append(np.full(len(rows), occurrence, dtype=int))
    return np.concatenate(row_parts), np.concatenate(group_parts)


def bootstrap_ranking_intervals(
    labels: Sequence[int] | np.ndarray,
    scores_by_model: Mapping[str, Sequence[float] | np.ndarray],
    groups: Sequence[Any] | np.ndarray,
    *,
    samples: int = 200,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate paired confidence intervals by resampling whole impressions."""

    if samples < 2:
        raise ValueError("bootstrap requires at least 2 samples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not scores_by_model:
        raise ValueError("scores_by_model must contain at least one model")
    if any(not isinstance(name, str) or not name.strip() for name in scores_by_model):
        raise ValueError("model names must be non-empty strings")

    y = np.asarray(labels)
    impression_ids = np.asarray(groups)
    if y.ndim != 1 or impression_ids.ndim != 1 or len(y) != len(impression_ids):
        raise ValueError("labels, scores, and groups must have the same length")
    if len(y) == 0:
        raise ValueError("bootstrap requires at least one candidate")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels must be binary values in {0, 1}")

    model_scores: dict[str, np.ndarray] = {}
    for name, scores in scores_by_model.items():
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or len(values) != len(y):
            raise ValueError("labels, scores, and groups must have the same length")
        if not np.isfinite(values).all():
            raise ValueError("scores must be finite")
        model_scores[name] = values

    distributions = {name: {metric: [] for metric in RANKING_METRICS} for name in model_scores}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        rows, resampled_groups = _bootstrap_indices(impression_ids, rng)
        for name, scores in model_scores.items():
            result = ranking_metrics(y[rows], scores[rows], resampled_groups)
            for metric in RANKING_METRICS:
                value = result[metric]
                if value is not None and np.isfinite(value):
                    distributions[name][metric].append(float(value))

    tail = (1 - confidence) / 2
    records: list[dict[str, str | float | int | None]] = []
    for name in sorted(distributions):
        for metric in sorted(RANKING_METRICS):
            values = distributions[name][metric]
            records.append(
                {
                    "model": name,
                    "metric": metric,
                    "mean": float(np.mean(values)) if values else None,
                    "lower": float(np.quantile(values, tail)) if values else None,
                    "upper": float(np.quantile(values, 1 - tail)) if values else None,
                    "valid_samples": len(values),
                    "requested_samples": samples,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "model",
            "metric",
            "mean",
            "lower",
            "upper",
            "valid_samples",
            "requested_samples",
        ],
    )


def run_logistic_ablations(
    X_train: pd.DataFrame,
    y_train: Sequence[int] | np.ndarray,
    train_groups: Sequence[Any] | np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: Sequence[int] | np.ndarray,
    valid_groups: Sequence[Any] | np.ndarray,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a fixed, deterministic logistic-regression feature ablation matrix."""

    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("training and validation features must have identical ordered columns")
    missing = sorted(
        {feature for group in FEATURE_GROUPS.values() for feature in group} - set(X_train.columns)
    )
    if missing:
        raise ValueError(f"feature frames are missing registered columns: {', '.join(missing)}")

    rows: list[dict[str, str | float | int | None]] = []
    all_features = list(X_train.columns)
    for name, config in ABLATIONS.items():
        if "include_groups" in config:
            selected = select_feature_names(include_groups=config["include_groups"])
        elif "exclude_groups" in config:
            selected = select_feature_names(exclude_groups=config["exclude_groups"])
        else:
            excluded = set(config.get("exclude_features", ()))
            selected = [feature for feature in all_features if feature not in excluded]
        model = fit_model("logistic", X_train[selected], y_train, train_groups, seed=seed)
        scores = predict_scores(model, X_valid[selected])
        metrics = ranking_metrics(y_valid, scores, valid_groups)
        rows.append(
            {
                "ablation": name,
                "feature_count": len(selected),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def _candidate_count_bucket(value: float) -> str:
    if value <= 5:
        return "small"
    if value <= 10:
        return "medium"
    return "large"


def _history_bucket(value: float) -> str:
    if value <= 0:
        return "cold"
    if value <= 5:
        return "short"
    return "long"


def _freshness_bucket(value: float) -> str:
    if value < 24:
        return "<24h"
    if value < 72:
        return "1-3d"
    if value < 168:
        return "3-7d"
    return "7d+"


def segment_ranking_metrics(
    candidates: pd.DataFrame,
    scores: Sequence[float] | np.ndarray,
    features: pd.DataFrame,
    *,
    min_impressions: int = 5,
) -> pd.DataFrame:
    """Calculate ranking metrics for whole-impression validation slices."""

    required_candidates = {
        "impression_id",
        "label",
        "device_type",
        "candidate_count",
    }
    missing_candidates = sorted(required_candidates - set(candidates.columns))
    if missing_candidates:
        raise ValueError(
            f"candidates are missing required columns: {', '.join(missing_candidates)}"
        )
    if "history_length" not in features:
        raise ValueError("features are missing required columns: history_length")
    values = np.asarray(scores, dtype=float)
    if len(candidates) != len(features) or len(candidates) != len(values):
        raise ValueError("candidates, scores, and features must have the same length")
    if min_impressions < 1:
        raise ValueError("min_impressions must be positive")

    frame = candidates.reset_index(drop=True).copy()
    frame["score"] = values
    frame["history_length"] = features["history_length"].reset_index(drop=True)
    frame["candidate_count_bucket"] = frame["candidate_count"].map(_candidate_count_bucket)
    frame["history_length_bucket"] = frame["history_length"].map(_history_bucket)

    segment_columns = {
        "device_type": "device_type",
        "candidate_count": "candidate_count_bucket",
        "history_length": "history_length_bucket",
    }
    raw_signal_columns = {
        "device_type": "device_type",
        "candidate_count": "candidate_count",
        "history_length": "history_length",
    }
    for segment, column in raw_signal_columns.items():
        counts = frame.groupby("impression_id", dropna=False)[column].nunique(dropna=False)
        if (counts > 1).any():
            raise ValueError(
                f"{segment} must be constant within each impression before segmentation"
            )

    records: list[dict[str, str | float | int | bool | None]] = []
    for segment, column in segment_columns.items():
        for value in pd.unique(frame[column]):
            mask = frame[column].isna() if pd.isna(value) else frame[column] == value
            sliced = frame.loc[mask]
            metrics = ranking_metrics(sliced["label"], sliced["score"], sliced["impression_id"])
            impression_count = int(sliced["impression_id"].nunique(dropna=False))
            records.append(
                {
                    "segment": segment,
                    "value": "<missing>" if pd.isna(value) else str(value),
                    "impressions": impression_count,
                    "candidates": len(sliced),
                    "low_support": impression_count < min_impressions,
                    **{metric: metrics[metric] for metric in RANKING_METRICS},
                }
            )
    return pd.DataFrame.from_records(records)


def candidate_bias_diagnostics(candidates: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Report empirical candidate click rates without implying causal effects."""

    required = {"candidate_position", "label"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates are missing required columns: {', '.join(missing)}")
    if "publication_age_hours" not in features:
        raise ValueError("features are missing required columns: publication_age_hours")
    if len(candidates) != len(features):
        raise ValueError("candidates and features must have the same length")
    if not candidates["label"].isin([0, 1]).all():
        raise ValueError("candidate labels must be binary")

    frame = candidates.reset_index(drop=True).copy()
    frame["article_freshness"] = (
        features["publication_age_hours"].reset_index(drop=True).map(_freshness_bucket)
    )
    records: list[dict[str, str | float | int]] = []
    diagnostics = (
        ("candidate_position", "candidate_position"),
        ("article_freshness", "article_freshness"),
    )
    for diagnostic, column in diagnostics:
        for bucket, group in frame.groupby(column, sort=False, dropna=False):
            clicks = int(group["label"].sum())
            records.append(
                {
                    "diagnostic": diagnostic,
                    "bucket": "<missing>" if pd.isna(bucket) else str(bucket),
                    "candidates": len(group),
                    "clicks": clicks,
                    "click_rate": clicks / len(group),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["diagnostic", "bucket", "candidates", "clicks", "click_rate"],
    )
