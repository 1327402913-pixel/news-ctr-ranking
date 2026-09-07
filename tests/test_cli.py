from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from news_ctr.cli import main
from news_ctr.data import expand_candidates, load_bundle
from news_ctr.models import predict_scores
from news_ctr.persistence import load_ranker


def test_cli_generates_audits_and_trains_reproducibly(tmp_path: Path, capsys) -> None:
    """Catches a workflow that works only through private notebook state."""
    data_path = tmp_path / "synthetic"
    first_output = tmp_path / "run-one"
    second_output = tmp_path / "run-two"

    assert (
        main(
            [
                "make-synthetic",
                "--output",
                str(data_path),
                "--seed",
                "23",
                "--users",
                "12",
                "--articles",
                "30",
                "--impressions",
                "60",
            ]
        )
        == 0
    )
    assert main(["audit", "--data", str(data_path), "--split", "train"]) == 0
    audit_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert audit_output["source_name"] == "synthetic:train"
    assert audit_output["impressions"] == 48

    common = [
        "train",
        "--data",
        str(data_path),
        "--model",
        "logistic",
        "--seed",
        "23",
        "--text-components",
        "4",
    ]
    assert main([*common, "--output", str(first_output)]) == 0
    capsys.readouterr()
    assert main([*common, "--output", str(second_output)]) == 0

    expected_artifacts = {
        "metrics.json",
        "predictions.parquet",
        "feature_importance.csv",
        "run_config.json",
        "model_card.md",
        "model.joblib",
        "model_metadata.json",
    }
    assert {item.name for item in first_output.iterdir()} == expected_artifacts
    metrics = json.loads((first_output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"] == "synthetic:validation"
    assert metrics["model"] == "logistic"
    for name in ("group_auc", "mrr", "ndcg@5", "ndcg@10", "log_loss"):
        assert np.isfinite(metrics["metrics"][name])
    assert 0 <= metrics["metrics"]["mrr"] <= 1

    first_predictions = pd.read_parquet(first_output / "predictions.parquet")
    second_predictions = pd.read_parquet(second_output / "predictions.parquet")
    assert first_predictions.columns.tolist() == [
        "impression_id",
        "article_id",
        "label",
        "score",
    ]
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    restored = load_ranker(first_output)
    run_config = json.loads((first_output / "run_config.json").read_text(encoding="utf-8"))
    assert restored.metadata["training_config"] == run_config
    assert restored.metadata["training_config"]["text_components"] == 4
    assert restored.metadata["training_config"]["validation_rows"] == len(first_predictions)
    valid_candidates = expand_candidates(load_bundle(data_path, split="validation").behaviors)
    restored_scores = predict_scores(
        restored.model,
        restored.feature_builder.transform(valid_candidates)[list(restored.feature_names)],
    )
    np.testing.assert_allclose(restored_scores, first_predictions["score"])
    label_free_behaviors = tmp_path / "label-free-behaviors.parquet"
    pd.read_parquet(data_path / "validation" / "behaviors.parquet").drop(
        columns="article_ids_clicked"
    ).to_parquet(label_free_behaviors, index=False)
    ranked_output = tmp_path / "ranked" / "candidates.parquet"
    assert (
        main(
            [
                "rank",
                "--model",
                str(first_output),
                "--candidates",
                str(label_free_behaviors),
                "--output",
                str(ranked_output),
                "--top-k",
                "3",
            ]
        )
        == 0
    )
    ranked = pd.read_parquet(ranked_output)
    assert ranked.groupby("impression_id").size().eq(3).all()
    assert "label" not in ranked

    malformed = tmp_path / "malformed.parquet"
    valid_candidates.drop(columns="impression_time").to_parquet(malformed, index=False)
    assert (
        main(
            [
                "rank",
                "--model",
                str(first_output),
                "--candidates",
                str(malformed),
                "--output",
                str(tmp_path / "must-not-exist.parquet"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "missing required scoring columns: impression_time" in captured.err
    assert "Traceback" not in captured.err
    model_card = (first_output / "model_card.md").read_text(encoding="utf-8")
    assert "engineering smoke test" in model_card.lower()
    assert "not an EB-NeRD benchmark" in model_card


def test_cli_returns_nonzero_with_an_actionable_data_error(tmp_path: Path, capsys) -> None:
    """Catches raw file tracebacks for a normal user input mistake."""
    missing = tmp_path / "does-not-exist"

    code = main(["audit", "--data", str(missing)])

    assert code == 2
    assert "dataset files not found" in capsys.readouterr().err


def test_benchmark_cli_creates_the_declared_report(tmp_path: Path, capsys) -> None:
    data = tmp_path / "data"
    output = tmp_path / "benchmark"
    assert (
        main(
            [
                "make-synthetic",
                "--output",
                str(data),
                "--seed",
                "13",
                "--users",
                "10",
                "--articles",
                "24",
                "--impressions",
                "40",
            ]
        )
        == 0
    )
    capsys.readouterr()

    code = main(
        [
            "benchmark",
            "--data",
            str(data),
            "--output",
            str(output),
            "--models",
            "position, popularity,logistic",
            "--bootstrap-samples",
            "20",
            "--text-components",
            "4",
            "--seed",
            "13",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["output"] == str(output)
    assert payload["models"] == ["position", "popularity", "logistic"]
    assert (output / "benchmark_report.md").is_file()


def test_decision_science_cli_end_to_end(tmp_path: Path, capsys) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("matplotlib")
    dataset = tmp_path / "ranking"
    experiment = tmp_path / "experiment.parquet"
    config = Path(__file__).parents[1] / "configs" / "experiment-v3.json"

    assert main(["make-synthetic", "--output", str(dataset), "--seed", "42"]) == 0
    assert (
        main(
            [
                "analyze",
                "--data",
                str(dataset),
                "--output",
                str(tmp_path / "analytics"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "experiment-design",
                "--baseline-rate",
                "0.12",
                "--relative-mde",
                "0.10",
                "--daily-units",
                "5000",
            ]
        )
        == 0
    )
    design = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert design["absolute_mde"] == 0.012
    assert design["total_units"] == 2 * design["units_per_arm"]
    assert design["estimated_days"] >= 1

    assert (
        main(
            [
                "make-experiment",
                "--output",
                str(experiment),
                "--seed",
                "42",
                "--users",
                "20000",
            ]
        )
        == 0
    )
    study = tmp_path / "study"
    assert (
        main(
            [
                "experiment",
                "--input",
                str(experiment),
                "--output",
                str(study),
                "--config",
                str(config),
            ]
        )
        == 0
    )
    assert (study / "decision_report.md").is_file()
    assert (study / "effects.png").is_file()


def test_causal_impact_cli_generates_and_analyzes_quasi_evidence(tmp_path: Path, capsys) -> None:
    """Catches the causal workflow working only through private Python calls."""

    pytest.importorskip("statsmodels")
    pytest.importorskip("matplotlib")
    panel = tmp_path / "market-week.parquet"
    output = tmp_path / "causal-study"
    config = Path(__file__).parents[1] / "configs" / "causal-impact-v4.json"

    assert (
        main(
            [
                "make-quasi-experiment",
                "--output",
                str(panel),
                "--seed",
                "42",
                "--markets",
                "60",
                "--pre-weeks",
                "20",
                "--post-weeks",
                "12",
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out.strip())
    assert generated == {
        "evidence_tier": "synthetic-quasi-experiment",
        "output": str(panel),
    }
    assert (
        main(
            [
                "causal-impact",
                "--input",
                str(panel),
                "--output",
                str(output),
                "--config",
                str(config),
            ]
        )
        == 0
    )
    analyzed = json.loads(capsys.readouterr().out.strip())
    assert analyzed == {"output": str(output)}
    assert (output / "causal_report.md").is_file()
    assert (output / "event_study.png").is_file()
    assert (output / "run_manifest.json").is_file()

    assert main(["make-quasi-experiment", "--output", str(panel)]) == 2
    assert (
        main(
            [
                "causal-impact",
                "--input",
                str(panel),
                "--output",
                str(output),
                "--config",
                str(config),
            ]
        )
        == 2
    )
    assert "error:" in capsys.readouterr().err


def test_quasi_experiment_cli_rejects_invalid_dimensions(tmp_path: Path, capsys) -> None:
    """Catches invalid causal fixtures escaping as unhandled tracebacks."""

    code = main(
        [
            "make-quasi-experiment",
            "--output",
            str(tmp_path / "invalid.parquet"),
            "--markets",
            "5",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "markets must be even" in captured.err
    assert "Traceback" not in captured.err
