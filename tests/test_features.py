from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news_ctr.features import NewsFeatureBuilder


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    articles = pd.DataFrame(
        {
            "article_id": [10, 11, 12, 13],
            "title": [
                "Football team wins",
                "Central bank changes rates",
                "Football transfer news",
                "New science discovery",
            ],
            "subtitle": ["sport result", "business economy", "sport player", "research lab"],
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
            "article_type": ["article", "analysis", "article", "article"],
            "sentiment_score": [0.7, 0.4, 0.8, np.nan],
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
            "impression_id": [100, 100, 101],
            "user_id": [1, 1, 2],
            "impression_time": pd.to_datetime(
                ["2026-01-02 08:00", "2026-01-02 08:00", "2026-01-02 09:00"]
            ),
            "article_id": [10, 11, 13],
            "label": [1, 0, 1],
            "candidate_position": [0, 1, 0],
            "candidate_count": [2, 2, 1],
            "device_type": [1, 1, 2],
            "is_sso_user": [True, True, False],
            "is_subscriber": [False, False, True],
        }
    )
    return articles, history, candidates


def test_transform_produces_stable_finite_numeric_features_without_ids() -> None:
    """Catches ID leakage, unstable order, and NaN values entering a model."""
    articles, history, candidates = _frames()
    builder = NewsFeatureBuilder(n_components=2, random_state=7)

    features = builder.fit(articles, history, candidates).transform(candidates)

    assert features.columns.tolist() == builder.feature_names_
    assert len(features) == len(candidates)
    assert all(np.issubdtype(dtype, np.number) for dtype in features.dtypes)
    assert np.isfinite(features.to_numpy()).all()
    assert not {"user_id", "article_id", "impression_id"} & set(features.columns)
    repeated = builder.transform(candidates)
    np.testing.assert_allclose(features.to_numpy(), repeated.to_numpy(), atol=0, rtol=0)


def test_category_affinity_comes_from_the_users_history() -> None:
    """Catches a global-category statistic being mistaken for personalized affinity."""
    articles, history, candidates = _frames()
    builder = NewsFeatureBuilder(n_components=2, random_state=7).fit(articles, history, candidates)

    features = builder.transform(candidates)

    assert features.loc[0, "category_affinity"] == pytest.approx(1.0)
    assert features.loc[1, "category_affinity"] == pytest.approx(0.0)


def test_empty_history_has_zero_interest_similarity() -> None:
    """Catches missing-history users inheriting another user's representation."""
    articles, history, candidates = _frames()
    builder = NewsFeatureBuilder(n_components=2, random_state=7).fit(articles, history, candidates)

    features = builder.transform(candidates.iloc[[2]])

    assert features.iloc[0]["history_length"] == 0
    assert features.iloc[0]["text_similarity"] == 0
    assert features.iloc[0]["recent_text_similarity"] == 0


def test_publication_age_uses_impression_time() -> None:
    """Catches non-reproducible age derived from the machine's current clock."""
    articles, history, candidates = _frames()
    builder = NewsFeatureBuilder(n_components=2, random_state=7).fit(articles, history, candidates)

    features = builder.transform(candidates.iloc[[0]])

    assert features.iloc[0]["publication_age_hours"] == pytest.approx(24.0)


def test_transform_rejects_article_missing_from_catalog() -> None:
    """Catches silently assigning misleading defaults to an unknown candidate."""
    articles, history, candidates = _frames()
    builder = NewsFeatureBuilder(n_components=2, random_state=7).fit(articles, history, candidates)
    candidates.loc[0, "article_id"] = 999

    with pytest.raises(ValueError, match="candidate articles are missing from the catalog: 999"):
        builder.transform(candidates)


def test_missing_boolean_and_text_values_are_not_treated_as_true_or_literal_nan() -> None:
    """Catches Python truthiness turning missing metadata into positive feature values."""
    articles, history, candidates = _frames()
    articles.loc[0, "title"] = np.nan
    articles["premium"] = articles["premium"].astype("boolean")
    articles.loc[0, "premium"] = pd.NA
    candidates["is_sso_user"] = candidates["is_sso_user"].astype("boolean")
    candidates.loc[0, "is_sso_user"] = pd.NA

    builder = NewsFeatureBuilder(n_components=2, random_state=7).fit(articles, history, candidates)
    features = builder.transform(candidates.iloc[[0]])

    assert features.iloc[0]["title_length"] == 0
    assert features.iloc[0]["premium"] == pytest.approx(0.5)
    assert features.iloc[0]["is_sso_user"] == pytest.approx(0.5)
