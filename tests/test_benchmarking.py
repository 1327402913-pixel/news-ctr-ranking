from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import news_ctr.benchmarking as benchmarking
from news_ctr.benchmarking import BenchmarkConfig, run_benchmark
from news_ctr.data import write_synthetic_bundle
from news_ctr.persistence import load_ranker
from news_ctr.reporting import render_benchmark_report


def _config(data: Path, output: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        data=data,
        output=output,
        models=("position", "popularity", "logistic"),
        bootstrap_samples=20,
        text_components=4,
        seed=11,
    )


def test_benchmark_writes_reproducible_complete_artifacts(tmp_path: Path) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data",
        seed=11,
        n_users=12,
        n_articles=30,
        n_impressions=60,
    )

    first = run_benchmark(_config(data, tmp_path / "first"))
    second = run_benchmark(_config(data, tmp_path / "second"))

    expected = {
        "benchmark_config.json",
        "leaderboard.csv",
        "confidence_intervals.csv",
        "ablations.csv",
        "segment_metrics.csv",
        "candidate_diagnostics.csv",
        "benchmark_report.md",
        "runs",
    }
    assert {item.name for item in first.iterdir()} == expected
    benchmark_config = json.loads((first / "benchmark_config.json").read_text(encoding="utf-8"))
    assert benchmark_config["synthetic_generation"] == {
        "seed": 11,
        "users": 12,
        "articles": 30,
        "impressions": 60,
    }
    assert {item.name for item in (first / "runs").iterdir()} == {
        "position",
        "popularity",
        "logistic",
    }
    first_leaderboard = pd.read_csv(first / "leaderboard.csv")
    second_leaderboard = pd.read_csv(second / "leaderboard.csv")
    deterministic = ["model", "group_auc", "mrr", "ndcg@5", "ndcg@10"]
    pd.testing.assert_frame_equal(
        first_leaderboard[deterministic],
        second_leaderboard[deterministic],
        check_exact=False,
        rtol=1e-8,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "confidence_intervals.csv"),
        pd.read_csv(second / "confidence_intervals.csv"),
    )
    assert first_leaderboard["fit_seconds"].ge(0).all()
    assert first_leaderboard["score_ms_per_1000"].ge(0).all()
    assert (
        first_leaderboard.loc[first_leaderboard["model"] == "position", "model_bytes"].item() == 0
    )
    assert first_leaderboard.loc[first_leaderboard["model"] == "logistic", "model_bytes"].item() > 0
    segment_table = pd.read_csv(first / "segment_metrics.csv")
    assert segment_table.columns[0] == "model"
    assert set(segment_table["model"]) == {"logistic"}
    saved_logistic = load_ranker(first / "runs" / "logistic")
    assert saved_logistic.metadata["training_config"]["bootstrap_samples"] == 20
    assert saved_logistic.metadata["training_config"]["text_components"] == 4


def test_failed_benchmark_does_not_publish_output(tmp_path: Path, monkeypatch) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data",
        seed=3,
        n_users=8,
        n_articles=20,
        n_impressions=40,
    )
    output = tmp_path / "result"
    config = BenchmarkConfig(
        data=data,
        output=output,
        models=("logistic",),
        bootstrap_samples=10,
        text_components=4,
        seed=3,
    )

    def fail_after_model_artifacts(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(benchmarking, "bootstrap_ranking_intervals", fail_after_model_artifacts)

    with pytest.raises(RuntimeError, match="boom"):
        run_benchmark(config)
    assert not output.exists()


def test_benchmark_rejects_duplicate_models_or_existing_output(tmp_path: Path) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data",
        seed=5,
        n_users=8,
        n_articles=20,
        n_impressions=40,
    )
    with pytest.raises(ValueError, match="duplicate model"):
        run_benchmark(
            BenchmarkConfig(data=data, output=tmp_path / "result", models=("logistic",) * 2)
        )

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run_benchmark(BenchmarkConfig(data=data, output=output, models=("logistic",)))


def test_synthetic_benchmark_report_is_recruiter_readable_and_truthful() -> None:
    leaderboard = pd.DataFrame(
        {
            "model": ["position", "logistic"],
            "group_auc": [0.5, 0.7],
            "mrr": [0.6, 0.8],
            "ndcg@5": [0.7, 0.9],
            "ndcg@10": [0.7, 0.9],
            "fit_seconds": [0.0, 0.1],
            "score_ms_per_1000": [0.1, 0.2],
            "model_bytes": [0, 2048],
        }
    )
    intervals = pd.DataFrame(
        {
            "model": ["logistic"],
            "metric": ["ndcg@10"],
            "mean": [0.9],
            "lower": [0.8],
            "upper": [0.95],
            "valid_samples": [20],
            "requested_samples": [20],
        }
    )
    ablations = pd.DataFrame(
        {
            "ablation": ["no_position", "full"],
            "feature_count": [20, 21],
            "group_auc": [0.65, 0.7],
            "mrr": [0.75, 0.8],
            "ndcg@5": [0.85, 0.9],
            "ndcg@10": [0.85, 0.9],
        }
    )
    segments = pd.DataFrame(
        {
            "model": ["logistic"],
            "segment": ["device_type"],
            "value": ["1"],
            "impressions": [3],
            "candidates": [18],
            "low_support": [True],
            "group_auc": [0.7],
            "mrr": [0.8],
            "ndcg@5": [0.9],
            "ndcg@10": [0.9],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "diagnostic": ["candidate_position"],
            "bucket": ["0"],
            "candidates": [10],
            "clicks": [4],
            "click_rate": [0.4],
        }
    )

    report = render_benchmark_report(
        dataset="synthetic:validation",
        leaderboard=leaderboard,
        intervals=intervals,
        ablations=ablations,
        segments=segments,
        diagnostics=diagnostics,
        config={
            "data": "data/synthetic",
            "output": "artifacts/portfolio-v2",
            "models": ["position", "logistic"],
            "bootstrap_samples": 20,
            "text_components": 4,
            "valid_fraction": 0.2,
            "seed": 13,
            "synthetic_generation": {
                "seed": 13,
                "users": 10,
                "articles": 24,
                "impressions": 40,
            },
        },
    )

    assert "60-second summary" in report
    assert "not an EB-NeRD benchmark" in report
    assert "95%" in report
    assert "Exposure-bias diagnostic" in report
    assert "Analysis model: `logistic`" in report
    assert "| Model |" in report
    assert "news-ctr make-synthetic" in report
    assert "--users 10 --articles 24 --impressions 40" in report
    assert "--output artifacts/portfolio-v2-reproduced" in report
