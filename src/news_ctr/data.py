"""Data contracts and deterministic fixtures for news ranking experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when a dataset cannot support impression-level ranking."""


@dataclass(frozen=True)
class DatasetBundle:
    """Canonical article, behavior, and history tables for one data split."""

    articles: pd.DataFrame
    behaviors: pd.DataFrame
    history: pd.DataFrame
    source_name: str


_REQUIRED = {
    "articles": {
        "article_id",
        "title",
        "subtitle",
        "published_time",
        "category",
        "premium",
    },
    "behaviors": {
        "impression_id",
        "user_id",
        "impression_time",
        "article_ids_inview",
        "article_ids_clicked",
    },
    "history": {"user_id", "article_id_fixed"},
}

_ALIASES = {
    "articles": {
        "published_at": "published_time",
        "category_id": "category",
        "is_premium": "premium",
    },
    "behaviors": {
        "article_id_inview": "article_ids_inview",
        "article_id_clicked": "article_ids_clicked",
        "timestamp": "impression_time",
        "is_subscriber_user": "is_subscriber",
    },
    "history": {
        "article_ids": "article_id_fixed",
        "article_ids_fixed": "article_id_fixed",
        "timestamps": "impression_time_fixed",
    },
}


def _canonicalize(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    aliases = {
        source: target
        for source, target in _ALIASES[table].items()
        if source in frame.columns and target not in frame.columns
    }
    return frame.rename(columns=aliases).copy()


def _as_list(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise DataContractError(f"{field} must contain a list in every row")


def _check_required(frame: pd.DataFrame, table: str) -> None:
    missing = sorted(_REQUIRED[table] - set(frame.columns))
    if missing:
        raise DataContractError(f"{table} is missing required columns: {', '.join(missing)}")


def load_bundle(path: str | Path, *, split: str = "train") -> DatasetBundle:
    """Load an EB-NeRD-style bundle and normalize supported column aliases."""

    root = Path(path)
    split_root = root / split if (root / split).is_dir() else root
    article_path = root / "articles.parquet"
    behavior_path = split_root / "behaviors.parquet"
    history_path = split_root / "history.parquet"
    missing_files = [
        str(item) for item in (article_path, behavior_path, history_path) if not item.is_file()
    ]
    if missing_files:
        raise DataContractError(f"dataset files not found: {', '.join(missing_files)}")

    source = f"ebnerd:{split}"
    marker = root / "dataset.json"
    if marker.is_file():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            source = f"{metadata.get('source', 'ebnerd')}:{split}"
        except (json.JSONDecodeError, OSError) as exc:
            raise DataContractError(f"cannot read dataset metadata: {marker}") from exc

    bundle = DatasetBundle(
        articles=_canonicalize(pd.read_parquet(article_path), "articles"),
        behaviors=_canonicalize(pd.read_parquet(behavior_path), "behaviors"),
        history=_canonicalize(pd.read_parquet(history_path), "history"),
        source_name=source,
    )
    audit_bundle(bundle)
    return bundle


def audit_bundle(bundle: DatasetBundle) -> dict[str, Any]:
    """Validate ranking invariants and return a compact dataset summary."""

    for name in ("articles", "behaviors", "history"):
        _check_required(getattr(bundle, name), name)

    if bundle.behaviors.empty:
        raise DataContractError("behaviors must contain at least one impression")
    if bundle.behaviors["impression_id"].duplicated().any():
        raise DataContractError("behaviors contains duplicate impression_id values")
    if bundle.articles["article_id"].duplicated().any():
        raise DataContractError("articles contains duplicate article_id values")

    try:
        pd.to_datetime(bundle.behaviors["impression_time"], errors="raise")
        pd.to_datetime(bundle.articles["published_time"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            "impression_time and published_time must be valid timestamps"
        ) from exc

    article_ids = set(bundle.articles["article_id"].tolist())
    published_by_article = dict(
        zip(
            bundle.articles["article_id"],
            pd.to_datetime(bundle.articles["published_time"], errors="raise"),
            strict=True,
        )
    )
    candidate_total = 0
    click_total = 0
    for row in bundle.behaviors.itertuples(index=False):
        inview = _as_list(row.article_ids_inview, field="article_ids_inview")
        clicked = _as_list(row.article_ids_clicked, field="article_ids_clicked")
        if not inview:
            raise DataContractError(
                f"impression {row.impression_id} has an empty article_ids_inview list"
            )
        if len(inview) != len(set(inview)):
            raise DataContractError(
                f"impression {row.impression_id} contains duplicate in-view articles"
            )
        unexpected = set(clicked) - set(inview)
        if unexpected:
            raise DataContractError(
                f"clicked articles {sorted(unexpected)} are not present in article_ids_inview "
                f"for impression {row.impression_id}"
            )
        unknown = set(inview) - article_ids
        if unknown:
            raise DataContractError(
                f"in-view articles {sorted(unknown)} are missing from articles for impression "
                f"{row.impression_id}"
            )
        impression_time = pd.Timestamp(row.impression_time)
        future_articles = sorted(
            article_id
            for article_id in inview
            if pd.Timestamp(published_by_article[article_id]) > impression_time
        )
        if future_articles:
            raise DataContractError(
                f"articles {future_articles} were published after impression {row.impression_id}"
            )
        candidate_total += len(inview)
        click_total += len(clicked)

    for value in bundle.history["article_id_fixed"]:
        _as_list(value, field="article_id_fixed")

    return {
        "source_name": bundle.source_name,
        "articles": len(bundle.articles),
        "users": int(bundle.behaviors["user_id"].nunique()),
        "impressions": len(bundle.behaviors),
        "candidates": int(candidate_total),
        "clicks": int(click_total),
        "start_time": pd.Timestamp(bundle.behaviors["impression_time"].min()).isoformat(),
        "end_time": pd.Timestamp(bundle.behaviors["impression_time"].max()).isoformat(),
    }


def expand_candidates(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Expand candidate lists without crossing impression boundaries."""

    _check_required(behaviors, "behaviors")
    records: list[dict[str, Any]] = []
    source_columns = [
        column
        for column in behaviors.columns
        if column not in {"article_ids_inview", "article_ids_clicked"}
    ]
    for row in behaviors.to_dict("records"):
        inview = _as_list(row["article_ids_inview"], field="article_ids_inview")
        clicked = set(_as_list(row["article_ids_clicked"], field="article_ids_clicked"))
        base = {column: row[column] for column in source_columns}
        for position, article_id in enumerate(inview):
            records.append(
                {
                    **base,
                    "article_id": article_id,
                    "label": int(article_id in clicked),
                    "candidate_position": position,
                    "candidate_count": len(inview),
                }
            )
    return pd.DataFrame.from_records(records)


def expand_scoring_candidates(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Expand behavior rows for inference without requiring click labels."""

    required = {"impression_id", "user_id", "impression_time", "article_ids_inview"}
    missing = sorted(required - set(behaviors.columns))
    if missing:
        raise DataContractError(
            f"behaviors are missing required scoring columns: {', '.join(missing)}"
        )
    if behaviors.empty:
        raise DataContractError("behaviors for scoring must not be empty")

    records: list[dict[str, Any]] = []
    source_columns = [
        column
        for column in behaviors.columns
        if column not in {"article_ids_inview", "article_ids_clicked"}
    ]
    for row in behaviors.to_dict("records"):
        inview = _as_list(row["article_ids_inview"], field="article_ids_inview")
        if not inview:
            raise DataContractError(
                f"impression {row['impression_id']} has an empty article_ids_inview list"
            )
        if len(inview) != len(set(inview)):
            raise DataContractError(
                f"impression {row['impression_id']} contains duplicate in-view articles"
            )
        base = {column: row[column] for column in source_columns}
        for position, article_id in enumerate(inview):
            records.append(
                {
                    **base,
                    "article_id": article_id,
                    "candidate_position": position,
                    "candidate_count": len(inview),
                }
            )
    return pd.DataFrame.from_records(records)


def temporal_split(
    frame: pd.DataFrame, *, valid_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on whole timestamps so validation is strictly after training."""

    if not 0 < valid_fraction < 1:
        raise DataContractError("valid_fraction must be between 0 and 1")
    if "impression_time" not in frame:
        raise DataContractError("frame is missing required column: impression_time")
    timestamps = pd.Series(pd.to_datetime(frame["impression_time"], errors="raise")).sort_values()
    unique_times = timestamps.drop_duplicates().tolist()
    if len(unique_times) < 2:
        raise DataContractError(
            "temporal split requires at least two distinct impression timestamps"
        )
    valid_size = min(len(unique_times) - 1, max(1, math.ceil(len(unique_times) * valid_fraction)))
    cutoff = unique_times[-valid_size]
    train = frame.loc[pd.to_datetime(frame["impression_time"]) < cutoff].copy()
    valid = frame.loc[pd.to_datetime(frame["impression_time"]) >= cutoff].copy()
    if train.empty or valid.empty:
        raise DataContractError("temporal split produced an empty fold")
    if pd.Timestamp(train["impression_time"].max()) >= pd.Timestamp(valid["impression_time"].min()):
        raise DataContractError("temporal split is not strictly ordered")
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def write_synthetic_bundle(
    path: str | Path,
    *,
    seed: int = 42,
    n_users: int = 30,
    n_articles: int = 80,
    n_impressions: int = 180,
) -> Path:
    """Write a deterministic, learnable EB-NeRD-shaped dataset for smoke tests."""

    if n_users < 2 or n_articles < 8 or n_impressions < 10:
        raise DataContractError(
            "synthetic data requires at least 2 users, 8 articles, and 10 impressions"
        )
    root = Path(path)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "validation").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    topic_words = np.array(["sport", "business", "culture", "science", "travel"])
    category = np.arange(n_articles) % len(topic_words)
    published = pd.Timestamp("2025-12-20") + pd.to_timedelta(np.arange(n_articles) * 4, unit="h")
    articles = pd.DataFrame(
        {
            "article_id": np.arange(1000, 1000 + n_articles, dtype=np.int64),
            "title": [
                f"{topic_words[item]} headline {index}" for index, item in enumerate(category)
            ],
            "subtitle": [f"Latest {topic_words[item]} analysis and context" for item in category],
            "published_time": published,
            "category": category.astype(np.int16),
            "premium": (np.arange(n_articles) % 7 == 0),
            "article_type": np.where(np.arange(n_articles) % 5 == 0, "analysis", "article"),
            "sentiment_score": np.round(rng.uniform(0.1, 0.9, n_articles), 4),
        }
    )

    preferences = rng.integers(0, len(topic_words), size=n_users)
    histories: list[dict[str, Any]] = []
    for user_index in range(n_users):
        matching = articles.loc[articles["category"] == preferences[user_index], "article_id"]
        history_ids = rng.choice(matching.to_numpy(), size=min(4, len(matching)), replace=False)
        history_times = [
            pd.Timestamp("2025-12-30") + pd.Timedelta(hours=i) for i in range(len(history_ids))
        ]
        histories.append(
            {
                "user_id": user_index + 1,
                "article_id_fixed": history_ids.tolist(),
                "impression_time_fixed": history_times,
                "read_time_fixed": rng.uniform(10, 120, len(history_ids)).round(2).tolist(),
            }
        )
    history = pd.DataFrame(histories)

    behavior_records: list[dict[str, Any]] = []
    candidate_size = min(6, n_articles)
    for impression_index in range(n_impressions):
        user_index = impression_index % n_users
        timestamp = pd.Timestamp("2026-01-10") + pd.Timedelta(hours=impression_index)
        candidates = rng.choice(articles["article_id"].to_numpy(), candidate_size, replace=False)
        metadata = articles.set_index("article_id").loc[candidates]
        affinity = (metadata["category"].to_numpy() == preferences[user_index]).astype(float)
        age_days = (
            timestamp - pd.to_datetime(metadata["published_time"])
        ).dt.total_seconds().to_numpy() / 86400
        logits = 1.8 * affinity - 0.18 * np.arange(candidate_size) - 0.015 * age_days
        logits += rng.normal(0, 0.25, candidate_size)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        clicked = int(rng.choice(candidates, p=probabilities))
        behavior_records.append(
            {
                "impression_id": 50000 + impression_index,
                "user_id": user_index + 1,
                "impression_time": timestamp,
                "article_ids_inview": candidates.tolist(),
                "article_ids_clicked": [clicked],
                "device_type": int(1 + impression_index % 3),
                "is_sso_user": bool(user_index % 2),
                "is_subscriber": bool(user_index % 4 == 0),
            }
        )
    behaviors = pd.DataFrame(behavior_records)
    split_at = max(1, int(n_impressions * 0.8))

    articles.to_parquet(root / "articles.parquet", index=False)
    history.to_parquet(root / "train" / "history.parquet", index=False)
    history.to_parquet(root / "validation" / "history.parquet", index=False)
    behaviors.iloc[:split_at].to_parquet(root / "train" / "behaviors.parquet", index=False)
    behaviors.iloc[split_at:].to_parquet(root / "validation" / "behaviors.parquet", index=False)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "source": "synthetic",
                "seed": seed,
                "users": n_users,
                "articles": n_articles,
                "impressions": n_impressions,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root
