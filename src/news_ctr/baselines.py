"""Simple, training-only diagnostic baselines for news ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineScorer:
    """Immutable fitted state for a ranking diagnostic baseline."""

    kind: str
    global_rate: float | None = None
    article_rates: Mapping[Any, float] = field(default_factory=lambda: MappingProxyType({}))


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def fit_baseline(
    kind: str,
    train_candidates: pd.DataFrame,
    *,
    alpha: float = 20.0,
) -> BaselineScorer:
    """Fit a position or smoothed article-popularity baseline."""

    if kind not in {"popularity", "position"}:
        raise ValueError("baseline kind must be one of: popularity, position")
    _require_columns(
        train_candidates,
        {"article_id", "candidate_position", "label"},
        name="train_candidates",
    )
    if train_candidates.empty:
        raise ValueError("train_candidates must not be empty")
    if not train_candidates["label"].isin([0, 1]).all():
        raise ValueError("train_candidates labels must be binary")

    if kind == "position":
        return BaselineScorer(kind=kind)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")

    global_rate = float(train_candidates["label"].mean())
    grouped = train_candidates.groupby("article_id", dropna=False)["label"].agg(["sum", "count"])
    rates = (grouped["sum"] + alpha * global_rate) / (grouped["count"] + alpha)
    return BaselineScorer(
        kind=kind,
        global_rate=global_rate,
        article_rates=MappingProxyType(rates.to_dict()),
    )


def predict_baseline(model: BaselineScorer, candidates: pd.DataFrame) -> np.ndarray:
    """Score candidate rows without using validation or test labels."""

    if model.kind == "position":
        _require_columns(candidates, {"candidate_position"}, name="candidates")
        return -candidates["candidate_position"].to_numpy(dtype=float)
    if model.kind == "popularity":
        _require_columns(candidates, {"article_id"}, name="candidates")
        assert model.global_rate is not None
        return (
            candidates["article_id"]
            .map(model.article_rates)
            .fillna(model.global_rate)
            .to_numpy(dtype=float)
        )
    raise ValueError("baseline kind must be one of: popularity, position")
