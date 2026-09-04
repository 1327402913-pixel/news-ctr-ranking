"""Schema-versioned persistence for trusted, locally produced rankers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from news_ctr.features import NewsFeatureBuilder
from news_ctr.models import TrainedModel, predict_scores
from news_ctr.reporting import _write_json

SAVED_RANKER_SCHEMA_VERSION = 1


@dataclass
class SavedRanker:
    """Feature transformation, model state, and provenance needed for scoring."""

    schema_version: int
    model: TrainedModel
    feature_builder: NewsFeatureBuilder
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


def save_ranker(path: str | Path, ranker: SavedRanker) -> tuple[Path, Path]:
    """Atomically save a trusted fitted ranker plus human-readable metadata."""

    if ranker.schema_version != SAVED_RANKER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported saved-ranker schema: {ranker.schema_version}; "
            f"expected {SAVED_RANKER_SCHEMA_VERSION}"
        )
    destination = Path(path)
    if destination.suffix == ".joblib":
        model_path = destination
        output_dir = destination.parent
    else:
        output_dir = destination
        model_path = output_dir / "model.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "model_metadata.json"

    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=".model.joblib.", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(ranker, temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    _write_json(
        metadata_path,
        {
            "schema_version": ranker.schema_version,
            "feature_names": ranker.feature_names,
            "metadata": ranker.metadata,
        },
    )
    return model_path, metadata_path


def load_ranker(path: str | Path) -> SavedRanker:
    """Load a trusted local artifact after validating its schema and object type."""

    source = Path(path)
    model_path = source / "model.joblib" if source.is_dir() else source
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"could not load saved ranker from {model_path}: {exc}") from exc

    if isinstance(artifact, Mapping):
        schema_version = artifact.get("schema_version")
    else:
        schema_version = getattr(artifact, "schema_version", None)
    if schema_version != SAVED_RANKER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported saved-ranker schema: {schema_version}; "
            f"expected {SAVED_RANKER_SCHEMA_VERSION}"
        )
    if not isinstance(artifact, SavedRanker):
        raise ValueError("saved-ranker artifact has an invalid object type")
    return artifact


def rank_candidates(
    ranker: SavedRanker,
    candidates: pd.DataFrame,
    *,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Score and deterministically rank candidates within each impression."""

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    required = {
        "impression_id",
        "user_id",
        "impression_time",
        "article_id",
        "candidate_position",
        "candidate_count",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates are missing required scoring columns: {', '.join(missing)}")
    if candidates.empty:
        raise ValueError("candidates for scoring must not be empty")

    features = ranker.feature_builder.transform(candidates)
    missing_features = sorted(set(ranker.feature_names) - set(features.columns))
    if missing_features:
        raise ValueError(
            f"saved feature builder is missing model features: {', '.join(missing_features)}"
        )
    scores = predict_scores(ranker.model, features[list(ranker.feature_names)])
    ranked = candidates[["impression_id", "article_id"]].copy()
    ranked["score"] = np.asarray(scores, dtype=float)
    ranked["_candidate_order"] = np.arange(len(ranked))
    ranked = ranked.sort_values(
        ["impression_id", "score", "_candidate_order"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked["rank"] = ranked.groupby("impression_id", sort=False, dropna=False).cumcount() + 1
    if top_k is not None:
        ranked = ranked.loc[ranked["rank"] <= top_k]
    return ranked[["impression_id", "article_id", "score", "rank"]].reset_index(drop=True)
