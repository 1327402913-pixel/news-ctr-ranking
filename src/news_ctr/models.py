"""Pointwise and learning-to-rank model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrainedModel:
    """Small common wrapper for model-specific estimators."""

    kind: str
    estimator: Any


def fit_model(
    kind: str,
    X: pd.DataFrame,
    y: Sequence[int] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    seed: int = 42,
) -> TrainedModel:
    """Fit a deterministic pointwise baseline or an optional LambdaRank model."""

    allowed = {"logistic", "lightgbm"}
    if kind not in allowed:
        raise ValueError(f"model kind must be one of: {', '.join(sorted(allowed))}")
    labels = np.asarray(y, dtype=int)
    impression_ids = np.asarray(groups)
    if not (len(X) == len(labels) == len(impression_ids)):
        raise ValueError("X, y, and groups must have the same length")
    if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) < 2:
        raise ValueError("training labels must contain both binary classes")

    if kind == "logistic":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        estimator.fit(X, labels)
        return TrainedModel(kind=kind, estimator=estimator)

    try:
        from lightgbm import LGBMRanker
    except ImportError as exc:
        raise RuntimeError(
            'LightGBM support is optional; install it with pip install ".[ranking]"'
        ) from exc
    except OSError as exc:
        if "libomp" in str(exc).lower() or "openmp" in str(exc).lower():
            raise RuntimeError(
                "LightGBM is installed but the OpenMP runtime is missing; on macOS, "
                "install libomp with your package manager and retry"
            ) from exc
        raise RuntimeError(f"the LightGBM native library could not be loaded: {exc}") from exc

    order = (
        pd.DataFrame({"group": impression_ids, "position": np.arange(len(impression_ids))})
        .sort_values("group", kind="mergesort")["position"]
        .to_numpy()
    )
    sorted_groups = impression_ids[order]
    group_sizes = (
        pd.Series(sorted_groups).groupby(sorted_groups, sort=False, dropna=False).size().tolist()
    )
    estimator = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
    )
    estimator.fit(X.iloc[order], labels[order], group=group_sizes)
    return TrainedModel(kind=kind, estimator=estimator)


def predict_scores(model: TrainedModel, X: pd.DataFrame) -> np.ndarray:
    """Return click probabilities for logistic regression or rank scores otherwise."""

    if model.kind == "logistic":
        return np.asarray(model.estimator.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.estimator.predict(X), dtype=float)


def feature_importance(model: TrainedModel, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return absolute coefficients or gain importance aligned to feature names."""

    names = list(feature_names)
    if model.kind == "logistic":
        values = np.abs(model.estimator.named_steps["classifier"].coef_[0])
    else:
        values = np.asarray(model.estimator.feature_importances_, dtype=float)
    if len(values) != len(names):
        raise ValueError("model importance length does not match feature names")
    return (
        pd.DataFrame({"feature": names, "importance": values.astype(float)})
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
