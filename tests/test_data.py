from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from news_ctr.data import (
    DataContractError,
    DatasetBundle,
    audit_bundle,
    expand_candidates,
    load_bundle,
    temporal_split,
    write_synthetic_bundle,
)


def _valid_bundle() -> DatasetBundle:
    articles = pd.DataFrame(
        {
            "article_id": [10, 11, 12],
            "title": ["Sport today", "Market today", "Local news"],
            "subtitle": ["A", "B", "C"],
            "published_time": pd.to_datetime(
                ["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-01 10:00"]
            ),
            "category": [1, 2, 3],
            "premium": [False, True, False],
        }
    )
    behaviors = pd.DataFrame(
        {
            "impression_id": [100, 101, 102],
            "user_id": [1, 2, 1],
            "impression_time": pd.to_datetime(
                ["2026-01-02 08:00", "2026-01-02 09:00", "2026-01-02 10:00"]
            ),
            "article_ids_inview": [[10, 11], [11, 12], [10, 12]],
            "article_ids_clicked": [[10], [12], [12]],
            "device_type": [1, 2, 1],
        }
    )
    history = pd.DataFrame(
        {
            "user_id": [1, 2],
            "article_id_fixed": [[10], [11]],
            "impression_time_fixed": [
                [pd.Timestamp("2026-01-01 11:00")],
                [pd.Timestamp("2026-01-01 12:00")],
            ],
        }
    )
    return DatasetBundle(articles, behaviors, history, source_name="unit-test")


def test_expand_candidates_keeps_each_negative_in_its_impression() -> None:
    """Catches mixing candidates or labels across impression boundaries."""
    expanded = expand_candidates(_valid_bundle().behaviors)

    assert expanded[["impression_id", "article_id", "label", "candidate_position"]].to_dict(
        "records"
    ) == [
        {"impression_id": 100, "article_id": 10, "label": 1, "candidate_position": 0},
        {"impression_id": 100, "article_id": 11, "label": 0, "candidate_position": 1},
        {"impression_id": 101, "article_id": 11, "label": 0, "candidate_position": 0},
        {"impression_id": 101, "article_id": 12, "label": 1, "candidate_position": 1},
        {"impression_id": 102, "article_id": 10, "label": 0, "candidate_position": 0},
        {"impression_id": 102, "article_id": 12, "label": 1, "candidate_position": 1},
    ]
    assert expanded.groupby("impression_id")["candidate_count"].first().to_dict() == {
        100: 2,
        101: 2,
        102: 2,
    }


def test_audit_rejects_clicked_article_that_was_not_exposed() -> None:
    """Catches silently constructing an impossible positive example."""
    bundle = _valid_bundle()
    bundle.behaviors.at[0, "article_ids_clicked"] = [12]

    with pytest.raises(DataContractError, match="not present in article_ids_inview"):
        audit_bundle(bundle)


def test_audit_rejects_missing_required_article_column() -> None:
    """Catches accepting source data that cannot produce article features."""
    bundle = _valid_bundle()
    bundle.articles.drop(columns="title", inplace=True)

    with pytest.raises(DataContractError, match="articles is missing required columns: title"):
        audit_bundle(bundle)


def test_audit_rejects_article_published_after_its_impression() -> None:
    """Catches allowing content metadata that was unavailable at prediction time."""
    bundle = _valid_bundle()
    bundle.articles.loc[bundle.articles["article_id"] == 10, "published_time"] = pd.Timestamp(
        "2026-01-03"
    )

    with pytest.raises(DataContractError, match="published after impression 100"):
        audit_bundle(bundle)


def test_temporal_split_has_strictly_separated_timestamps() -> None:
    """Catches random or row-level splitting that leaks a timestamp across folds."""
    frame = pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "impression_time": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-04",
                ]
            ),
        }
    )

    train, valid = temporal_split(frame, valid_fraction=0.5)

    assert train["impression_id"].tolist() == [1, 1, 2, 2]
    assert valid["impression_id"].tolist() == [3, 3, 4, 4]
    assert train["impression_time"].max() < valid["impression_time"].min()


def test_temporal_split_requires_multiple_timestamps() -> None:
    """Catches returning empty or overlapping folds for unsplittable input."""
    frame = pd.DataFrame(
        {"impression_id": [1, 1], "impression_time": pd.to_datetime(["2026-01-01"] * 2)}
    )

    with pytest.raises(DataContractError, match="at least two distinct impression timestamps"):
        temporal_split(frame)


def test_synthetic_bundle_is_deterministic_and_loadable(tmp_path: Path) -> None:
    """Catches uncontrolled randomness or a generator that violates its own contract."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_synthetic_bundle(first, seed=17, n_users=8, n_articles=20, n_impressions=24)
    write_synthetic_bundle(second, seed=17, n_users=8, n_articles=20, n_impressions=24)
    bundle_one = load_bundle(first, split="train")
    bundle_two = load_bundle(second, split="train")

    assert_frame_equal(bundle_one.articles, bundle_two.articles)
    assert_frame_equal(bundle_one.behaviors, bundle_two.behaviors)
    assert_frame_equal(bundle_one.history, bundle_two.history)
    summary = audit_bundle(bundle_one)
    assert summary["source_name"] == "synthetic:train"
    assert summary["impressions"] == len(bundle_one.behaviors)
    assert summary["candidates"] > summary["impressions"]


def test_load_bundle_accepts_official_history_aliases(tmp_path: Path) -> None:
    """Catches coupling the loader only to the synthetic column spelling."""
    root = tmp_path / "bundle"
    split = root / "train"
    split.mkdir(parents=True)
    valid = _valid_bundle()
    valid.articles.rename(columns={"subtitle": "subtitle"}).to_parquet(root / "articles.parquet")
    valid.behaviors.to_parquet(split / "behaviors.parquet")
    valid.history.rename(columns={"article_id_fixed": "article_ids"}).to_parquet(
        split / "history.parquet"
    )

    loaded = load_bundle(root, split="train")

    assert "article_id_fixed" in loaded.history
    assert loaded.history.loc[0, "article_id_fixed"].tolist() == [10]
