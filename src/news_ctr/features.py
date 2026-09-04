"""Leakage-aware structured and text features for candidate news articles."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize


def _list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _key(value: Any) -> str:
    return "<missing>" if pd.isna(value) else str(value)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


def _boolean_number(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return 1.0
        if normalized in {"false", "0", "no"}:
            return 0.0
        return float("nan")
    return float(bool(value))


def _text_length(value: Any) -> float:
    return 0.0 if value is None or pd.isna(value) else float(len(str(value)))


class NewsFeatureBuilder:
    """Fit reusable user-content representations on training-visible data only."""

    feature_names_: ClassVar[list[str]] = [
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "device_type",
        "is_sso_user",
        "is_subscriber",
        "candidate_position",
        "candidate_count",
        "publication_age_hours",
        "freshness_log_hours",
        "title_length",
        "subtitle_length",
        "category_code",
        "premium",
        "article_type_code",
        "sentiment_score",
        "history_length",
        "category_affinity",
        "text_similarity",
        "recent_text_similarity",
    ]

    def __init__(self, *, n_components: int = 32, random_state: int = 42) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.random_state = random_state
        self._fitted = False

    def fit(
        self,
        articles: pd.DataFrame,
        history: pd.DataFrame,
        train_candidates: pd.DataFrame,
    ) -> NewsFeatureBuilder:
        """Fit text vocabulary, category encoders, histories, and imputation values."""

        article_required = {
            "article_id",
            "title",
            "subtitle",
            "published_time",
            "category",
            "premium",
        }
        candidate_required = {
            "user_id",
            "article_id",
            "impression_time",
            "candidate_position",
            "candidate_count",
        }
        history_required = {"user_id", "article_id_fixed"}
        for name, frame, required in (
            ("articles", articles, article_required),
            ("history", history, history_required),
            ("train_candidates", train_candidates, candidate_required),
        ):
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
        if train_candidates.empty:
            raise ValueError("train_candidates must not be empty")

        self._articles = articles.copy()
        self._articles["published_time"] = pd.to_datetime(
            self._articles["published_time"], errors="raise"
        )
        self._articles_by_id = self._articles.set_index("article_id", drop=False)
        cutoff = pd.Timestamp(pd.to_datetime(train_candidates["impression_time"]).max())
        visible = self._articles.loc[self._articles["published_time"] <= cutoff]
        if len(visible) < 2:
            raise ValueError(
                "at least two training-visible articles are required for text features"
            )

        self._category_map = {
            value: index
            for index, value in enumerate(sorted({_key(item) for item in visible["category"]}))
        }
        article_types = (
            visible["article_type"]
            if "article_type" in visible
            else pd.Series("article", index=visible.index)
        )
        self._article_type_map = {
            value: index
            for index, value in enumerate(sorted({_key(item) for item in article_types}))
        }

        self._vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        max_features=12_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        ngram_range=(3, 5),
                        max_features=12_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        visible_text = self._text(visible)
        sparse_visible = self._vectorizer.fit_transform(visible_text)
        effective_components = min(
            self.n_components,
            max(1, sparse_visible.shape[0] - 1),
            max(1, sparse_visible.shape[1] - 1),
        )
        self._svd = TruncatedSVD(
            n_components=effective_components,
            algorithm="randomized",
            random_state=self.random_state,
        )
        self._svd.fit(sparse_visible)
        all_vectors = normalize(
            self._svd.transform(self._vectorizer.transform(self._text(self._articles)))
        )
        self._article_vectors = {
            article_id: all_vectors[index]
            for index, article_id in enumerate(self._articles["article_id"])
        }
        self._vector_size = all_vectors.shape[1]
        self._build_history_maps(history)

        raw = self._raw_transform(train_candidates)
        medians = raw.median(axis=0, skipna=True).fillna(0.0)
        self._medians = medians.reindex(self.feature_names_, fill_value=0.0)
        self._fitted = True
        return self

    @staticmethod
    def _text(articles: pd.DataFrame) -> pd.Series:
        title = articles["title"].fillna("").astype(str)
        subtitle = articles["subtitle"].fillna("").astype(str)
        return (title + " " + subtitle).str.strip().replace("", "missing text")

    def _build_history_maps(self, history: pd.DataFrame) -> None:
        self._history_vectors: dict[Any, np.ndarray] = {}
        self._recent_vectors: dict[Any, np.ndarray] = {}
        self._history_categories: dict[Any, Counter[str]] = {}
        self._history_lengths: dict[Any, int] = {}
        for row in history.to_dict("records"):
            user_id = row["user_id"]
            article_ids = [
                item for item in _list(row["article_id_fixed"]) if item in self._article_vectors
            ]
            vectors = [self._article_vectors[item] for item in article_ids]
            self._history_lengths[user_id] = len(article_ids)
            if vectors:
                self._history_vectors[user_id] = _unit(np.mean(vectors, axis=0))
                recent_id = article_ids[-1]
                timestamps = _list(row.get("impression_time_fixed", []))
                if len(timestamps) == len(article_ids) and timestamps:
                    timestamp_values = pd.to_datetime(timestamps).astype("int64")
                    recent_id = article_ids[int(np.argmax(timestamp_values))]
                self._recent_vectors[user_id] = self._article_vectors[recent_id]
            else:
                self._history_vectors[user_id] = np.zeros(self._vector_size)
                self._recent_vectors[user_id] = np.zeros(self._vector_size)
            categories = [_key(self._articles_by_id.at[item, "category"]) for item in article_ids]
            self._history_categories[user_id] = Counter(categories)

    def transform(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Build a finite numeric feature matrix while preserving candidate order."""

        if not self._fitted:
            raise RuntimeError("NewsFeatureBuilder must be fitted before transform")
        raw = self._raw_transform(candidates)
        return (
            raw.replace([np.inf, -np.inf], np.nan).fillna(self._medians).fillna(0.0).astype(float)
        )

    def _raw_transform(self, candidates: pd.DataFrame) -> pd.DataFrame:
        unknown = sorted(set(candidates["article_id"]) - set(self._articles_by_id.index))
        if unknown:
            shown = ", ".join(str(item) for item in unknown[:10])
            raise ValueError(f"candidate articles are missing from the catalog: {shown}")

        records: list[dict[str, float]] = []
        for row in candidates.to_dict("records"):
            article = self._articles_by_id.loc[row["article_id"]]
            timestamp = pd.Timestamp(row["impression_time"])
            published = pd.Timestamp(article["published_time"])
            age_hours = max(0.0, (timestamp - published).total_seconds() / 3600)
            user_id = row["user_id"]
            candidate_vector = self._article_vectors[row["article_id"]]
            history_vector = self._history_vectors.get(user_id, np.zeros(self._vector_size))
            recent_vector = self._recent_vectors.get(user_id, np.zeros(self._vector_size))
            history_length = self._history_lengths.get(user_id, 0)
            category_key = _key(article["category"])
            category_count = self._history_categories.get(user_id, Counter()).get(category_key, 0)
            hour_angle = 2 * np.pi * timestamp.hour / 24
            weekday_angle = 2 * np.pi * timestamp.dayofweek / 7
            records.append(
                {
                    "hour_sin": float(np.sin(hour_angle)),
                    "hour_cos": float(np.cos(hour_angle)),
                    "weekday_sin": float(np.sin(weekday_angle)),
                    "weekday_cos": float(np.cos(weekday_angle)),
                    "device_type": self._numeric(row.get("device_type")),
                    "is_sso_user": _boolean_number(row.get("is_sso_user", False)),
                    "is_subscriber": _boolean_number(row.get("is_subscriber", False)),
                    "candidate_position": self._numeric(row.get("candidate_position")),
                    "candidate_count": self._numeric(row.get("candidate_count")),
                    "publication_age_hours": age_hours,
                    "freshness_log_hours": float(np.log1p(age_hours)),
                    "title_length": _text_length(article.get("title")),
                    "subtitle_length": _text_length(article.get("subtitle")),
                    "category_code": float(self._category_map.get(category_key, -1)),
                    "premium": _boolean_number(article.get("premium", False)),
                    "article_type_code": float(
                        self._article_type_map.get(_key(article.get("article_type", "article")), -1)
                    ),
                    "sentiment_score": self._numeric(article.get("sentiment_score")),
                    "history_length": float(history_length),
                    "category_affinity": float(category_count / history_length)
                    if history_length
                    else 0.0,
                    "text_similarity": float(np.dot(candidate_vector, history_vector)),
                    "recent_text_similarity": float(np.dot(candidate_vector, recent_vector)),
                }
            )
        return pd.DataFrame.from_records(records, index=candidates.index)[self.feature_names_]

    @staticmethod
    def _numeric(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
