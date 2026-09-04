from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from news_ctr.models import feature_importance, fit_model, predict_scores


def _training_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = pd.DataFrame(
        {
            "affinity": [1.0, 0.0, 0.8, 0.1, 0.9, 0.2, 0.7, 0.3],
            "position": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.array([10, 10, 11, 11, 12, 12, 13, 13])
    return X, y, groups


def test_logistic_model_returns_deterministic_probabilities() -> None:
    """Catches returning class labels, logits, or uncontrolled random scores."""
    X, y, groups = _training_data()

    first = fit_model("logistic", X, y, groups, seed=19)
    second = fit_model("logistic", X, y, groups, seed=19)
    first_scores = predict_scores(first, X)
    second_scores = predict_scores(second, X)

    assert np.isfinite(first_scores).all()
    assert ((first_scores >= 0) & (first_scores <= 1)).all()
    np.testing.assert_allclose(first_scores, second_scores, atol=0, rtol=0)
    assert first_scores[0] > first_scores[1]


def test_feature_importance_covers_every_input_column() -> None:
    """Catches misaligning learned coefficients with feature names."""
    X, y, groups = _training_data()
    model = fit_model("logistic", X, y, groups, seed=19)

    importance = feature_importance(model, X.columns)

    assert importance.columns.tolist() == ["feature", "importance"]
    assert set(importance["feature"]) == set(X.columns)
    assert importance["importance"].ge(0).all()
    assert importance["importance"].is_monotonic_decreasing


def test_unknown_model_kind_is_rejected() -> None:
    """Catches silently falling back to a different estimator."""
    X, y, groups = _training_data()

    with pytest.raises(ValueError, match="model kind must be one of"):
        fit_model("random-forest", X, y, groups)


def test_lightgbm_extra_has_actionable_dependency_behavior() -> None:
    """Catches an optional model failing with an opaque import traceback."""
    X, y, groups = _training_data()

    if importlib.util.find_spec("lightgbm") is None:
        with pytest.raises(RuntimeError, match=r"pip install .*\[ranking\]"):
            fit_model("lightgbm", X, y, groups)
    else:
        try:
            model = fit_model("lightgbm", X, y, groups, seed=19)
        except RuntimeError as exc:
            assert "OpenMP runtime is missing" in str(exc)
        else:
            assert np.isfinite(predict_scores(model, X)).all()
