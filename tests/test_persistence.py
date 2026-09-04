from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from news_ctr.features import NewsFeatureBuilder
from news_ctr.models import TrainedModel, fit_model, predict_scores
from news_ctr.persistence import SavedRanker, load_ranker, rank_candidates, save_ranker


def _fitted_components() -> tuple[NewsFeatureBuilder, TrainedModel, pd.DataFrame]:
    articles = pd.DataFrame(
        {
            "article_id": [10, 11, 12, 13],
            "title": ["Football win", "Bank rates", "Transfer news", "Science lab"],
            "subtitle": ["sport", "business", "player", "research"],
            "published_time": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 07:00",
                    "2026-01-01 06:00",
                    "2026-01-01 05:00",
                ]
            ),
            "category": [1, 2, 1, 3],
            "premium": [False, True, False, False],
        }
    )
    history = pd.DataFrame(
        {
            "user_id": [1, 2],
            "article_id_fixed": [[10, 12], []],
            "impression_time_fixed": [
                [pd.Timestamp("2026-01-01 09:00"), pd.Timestamp("2026-01-01 10:00")],
                [],
            ],
        }
    )
    candidates = pd.DataFrame(
        {
            "impression_id": [100, 100, 101, 101],
            "user_id": [1, 1, 2, 2],
            "impression_time": pd.to_datetime(["2026-01-02 08:00"] * 4),
            "article_id": [10, 11, 12, 13],
            "label": [1, 0, 0, 1],
            "candidate_position": [0, 1, 0, 1],
            "candidate_count": [2, 2, 2, 2],
            "device_type": [1, 1, 2, 2],
            "is_sso_user": [True, True, False, False],
            "is_subscriber": [False, False, True, True],
        }
    )
    builder = NewsFeatureBuilder(n_components=2, random_state=7).fit(articles, history, candidates)
    features = builder.transform(candidates)
    model = fit_model(
        "logistic",
        features,
        candidates["label"],
        candidates["impression_id"],
        seed=7,
    )
    return builder, model, candidates


def _saved_ranker() -> tuple[SavedRanker, pd.DataFrame]:
    builder, model, candidates = _fitted_components()
    return (
        SavedRanker(
            schema_version=1,
            model=model,
            feature_builder=builder,
            feature_names=tuple(builder.feature_names_),
            metadata={"dataset_fingerprint": "abc", "model": "logistic"},
        ),
        candidates,
    )


def test_saved_ranker_round_trip_preserves_scores(tmp_path: Path) -> None:
    saved, valid = _saved_ranker()
    builder = saved.feature_builder

    model_path, metadata_path = save_ranker(tmp_path, saved)
    restored = load_ranker(model_path)

    np.testing.assert_allclose(
        predict_scores(saved.model, saved.feature_builder.transform(valid)),
        predict_scores(restored.model, restored.feature_builder.transform(valid)),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["feature_names"] == builder.feature_names_


def test_load_accepts_run_directory(tmp_path: Path) -> None:
    builder, model, _ = _fitted_components()
    save_ranker(
        tmp_path,
        SavedRanker(1, model, builder, tuple(builder.feature_names_), {"model": "logistic"}),
    )

    assert load_ranker(tmp_path).metadata["model"] == "logistic"


def test_load_rejects_incompatible_or_corrupt_schema(tmp_path: Path) -> None:
    joblib.dump({"schema_version": 99}, tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="unsupported saved-ranker schema"):
        load_ranker(tmp_path / "model.joblib")

    (tmp_path / "model.joblib").write_bytes(b"not-a-joblib")
    with pytest.raises(ValueError, match="could not load saved ranker"):
        load_ranker(tmp_path / "model.joblib")


def test_rank_candidates_orders_each_impression_and_applies_top_k() -> None:
    saved, candidates = _saved_ranker()

    ranked = rank_candidates(saved, candidates, top_k=2)

    assert ranked.columns.tolist() == ["impression_id", "article_id", "score", "rank"]
    assert ranked.groupby("impression_id").size().eq(2).all()
    rank_lists = ranked.groupby("impression_id")["rank"].apply(list)
    assert rank_lists.map(lambda values: values == [1, 2]).all()
    assert (
        ranked.groupby("impression_id")["score"]
        .apply(lambda values: values.is_monotonic_decreasing)
        .all()
    )


def test_rank_candidates_rejects_invalid_top_k() -> None:
    saved, candidates = _saved_ranker()

    with pytest.raises(ValueError, match="top_k must be positive"):
        rank_candidates(saved, candidates, top_k=0)


def test_rank_candidates_rejects_missing_or_empty_scoring_schema() -> None:
    saved, candidates = _saved_ranker()

    with pytest.raises(ValueError, match="missing required scoring columns: impression_time"):
        rank_candidates(saved, candidates.drop(columns="impression_time"))
    with pytest.raises(ValueError, match="must not be empty"):
        rank_candidates(saved, candidates.iloc[0:0])
