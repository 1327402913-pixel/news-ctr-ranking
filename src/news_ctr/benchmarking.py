"""End-to-end orchestration for comparable news-ranking benchmarks."""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from news_ctr.baselines import fit_baseline, predict_baseline
from news_ctr.data import (
    DataContractError,
    DatasetBundle,
    audit_bundle,
    expand_candidates,
    load_bundle,
    temporal_split,
)
from news_ctr.evaluation import (
    bootstrap_ranking_intervals,
    candidate_bias_diagnostics,
    run_logistic_ablations,
    segment_ranking_metrics,
)
from news_ctr.features import NewsFeatureBuilder
from news_ctr.metrics import RANKING_METRICS, ranking_metrics
from news_ctr.models import feature_importance, fit_model, predict_scores
from news_ctr.persistence import SAVED_RANKER_SCHEMA_VERSION, SavedRanker, save_ranker
from news_ctr.reporting import (
    _write_json,
    dataset_fingerprint,
    render_benchmark_report,
    write_run_artifacts,
)

SUPPORTED_BENCHMARK_MODELS = ("position", "popularity", "logistic", "lightgbm")


@dataclass(frozen=True)
class BenchmarkConfig:
    data: Path
    output: Path
    models: tuple[str, ...] = ("position", "popularity", "logistic")
    bootstrap_samples: int = 200
    text_components: int = 32
    valid_fraction: float = 0.2
    seed: int = 42


def _validate_config(config: BenchmarkConfig) -> None:
    if not config.models:
        raise ValueError("benchmark requires at least one model")
    if len(config.models) != len(set(config.models)):
        raise ValueError("benchmark model list contains a duplicate model")
    unknown = [name for name in config.models if name not in SUPPORTED_BENCHMARK_MODELS]
    if unknown:
        raise ValueError("benchmark model must be one of: " + ", ".join(SUPPORTED_BENCHMARK_MODELS))
    if config.bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    if config.text_components < 1:
        raise ValueError("text_components must be positive")
    if not 0 < config.valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")
    if Path(config.output).exists():
        raise ValueError(f"benchmark output already exists: {config.output}")


def _load_split(
    config: BenchmarkConfig,
) -> tuple[DatasetBundle, DatasetBundle, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_bundle = load_bundle(config.data, split="train")
    train_summary = audit_bundle(train_bundle)
    all_train_candidates = expand_candidates(train_bundle.behaviors)
    validation_path = Path(config.data) / "validation" / "behaviors.parquet"
    if validation_path.is_file():
        valid_bundle = load_bundle(config.data, split="validation")
        valid_summary = audit_bundle(valid_bundle)
        train_candidates = all_train_candidates
        valid_candidates = expand_candidates(valid_bundle.behaviors)
        if pd.Timestamp(train_candidates["impression_time"].max()) >= pd.Timestamp(
            valid_candidates["impression_time"].min()
        ):
            raise DataContractError(
                "training impressions must end before validation impressions begin"
            )
    else:
        train_candidates, valid_candidates = temporal_split(
            all_train_candidates, valid_fraction=config.valid_fraction
        )
        valid_bundle = DatasetBundle(
            articles=train_bundle.articles,
            behaviors=train_bundle.behaviors.loc[
                train_bundle.behaviors["impression_id"].isin(valid_candidates["impression_id"])
            ],
            history=train_bundle.history,
            source_name=f"{train_bundle.source_name}:temporal-validation",
        )
        valid_summary = {
            **train_summary,
            "source_name": valid_bundle.source_name,
            "impressions": int(valid_candidates["impression_id"].nunique()),
            "candidates": len(valid_candidates),
        }
    return train_bundle, valid_bundle, train_candidates, valid_candidates, valid_summary


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "scikit-learn": version("scikit-learn"),
    }
    with suppress(PackageNotFoundError):
        versions["lightgbm"] = version("lightgbm")
    return versions


def _synthetic_generation_config(data: Path) -> dict[str, int] | None:
    marker = data / "dataset.json"
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    fields = ("seed", "users", "articles", "impressions")
    if payload.get("source") != "synthetic" or any(field not in payload for field in fields):
        return None
    return {field: int(payload[field]) for field in fields}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10f")


def _execute_benchmark(config: BenchmarkConfig, output: Path) -> None:
    train_bundle, valid_bundle, train_candidates, valid_candidates, valid_summary = _load_split(
        config
    )
    fingerprint = dataset_fingerprint(config.data)
    builder = NewsFeatureBuilder(n_components=config.text_components, random_state=config.seed).fit(
        train_bundle.articles, train_bundle.history, train_candidates
    )
    X_train = builder.transform(train_candidates)
    X_valid = builder.transform(valid_candidates)
    y_train = train_candidates["label"].to_numpy()
    train_groups = train_candidates["impression_id"].to_numpy()
    y_valid = valid_candidates["label"].to_numpy()
    valid_groups = valid_candidates["impression_id"].to_numpy()

    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True)
    scores_by_model: dict[str, Any] = {}
    leaderboard_rows: list[dict[str, Any]] = []
    versions = _library_versions()

    for model_name in config.models:
        fit_started = time.perf_counter()
        if model_name in {"position", "popularity"}:
            fitted = fit_baseline(model_name, train_candidates)
        else:
            fitted = fit_model(model_name, X_train, y_train, train_groups, seed=config.seed)
        fit_seconds = time.perf_counter() - fit_started

        score_started = time.perf_counter()
        if model_name in {"position", "popularity"}:
            scores = predict_baseline(fitted, valid_candidates)
        else:
            scores = predict_scores(fitted, X_valid)
        score_seconds = time.perf_counter() - score_started
        scores_by_model[model_name] = scores
        metrics = ranking_metrics(y_valid, scores, valid_groups)

        predictions = valid_candidates[["impression_id", "article_id", "label"]].copy()
        predictions["score"] = scores
        importance = (
            pd.DataFrame(columns=["feature", "importance"])
            if model_name in {"position", "popularity"}
            else feature_importance(fitted, builder.feature_names_)
        )
        run_dir = runs_dir / model_name
        run_config = {
            "data": str(config.data),
            "output": str(Path(config.output) / "runs" / model_name),
            "model": model_name,
            "seed": config.seed,
            "text_components": config.text_components,
            "valid_fraction": config.valid_fraction,
            "dataset_fingerprint": fingerprint,
            "train_source": train_bundle.source_name,
            "train_rows": len(train_candidates),
            "validation_rows": len(valid_candidates),
            "features": builder.feature_names_,
            "benchmark_models": list(config.models),
            "bootstrap_samples": config.bootstrap_samples,
        }
        write_run_artifacts(
            run_dir,
            dataset=valid_bundle.source_name,
            dataset_summary=valid_summary,
            model_kind=model_name,
            metrics=metrics,
            predictions=predictions,
            importance=importance,
            run_config=run_config,
        )
        model_bytes = 0
        if model_name not in {"position", "popularity"}:
            model_path, _ = save_ranker(
                run_dir,
                SavedRanker(
                    schema_version=SAVED_RANKER_SCHEMA_VERSION,
                    model=fitted,
                    feature_builder=builder,
                    feature_names=tuple(builder.feature_names_),
                    metadata={
                        "dataset": valid_bundle.source_name,
                        "dataset_fingerprint": fingerprint,
                        "model": model_name,
                        "seed": config.seed,
                        "library_versions": versions,
                        "training_config": run_config,
                    },
                ),
            )
            model_bytes = model_path.stat().st_size
        leaderboard_rows.append(
            {
                "model": model_name,
                **{metric: metrics[metric] for metric in RANKING_METRICS},
                "fit_seconds": fit_seconds,
                "score_ms_per_1000": score_seconds * 1_000_000 / len(valid_candidates),
                "model_bytes": model_bytes,
            }
        )

    leaderboard = pd.DataFrame.from_records(leaderboard_rows)
    intervals = bootstrap_ranking_intervals(
        y_valid,
        scores_by_model,
        valid_groups,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    ablations = run_logistic_ablations(
        X_train,
        y_train,
        train_groups,
        X_valid,
        y_valid,
        valid_groups,
        seed=config.seed,
    )
    logistic_scores = scores_by_model.get("logistic")
    if logistic_scores is None:
        diagnostic_model = fit_model("logistic", X_train, y_train, train_groups, seed=config.seed)
        logistic_scores = predict_scores(diagnostic_model, X_valid)
    segments = segment_ranking_metrics(valid_candidates, logistic_scores, X_valid)
    segments.insert(0, "model", "logistic")
    diagnostics = candidate_bias_diagnostics(valid_candidates, X_valid)

    benchmark_config = {
        "data": str(config.data),
        "output": str(config.output),
        "models": config.models,
        "bootstrap_samples": config.bootstrap_samples,
        "text_components": config.text_components,
        "valid_fraction": config.valid_fraction,
        "seed": config.seed,
        "dataset": valid_bundle.source_name,
        "dataset_fingerprint": fingerprint,
        "train_rows": len(train_candidates),
        "validation_rows": len(valid_candidates),
    }
    synthetic_generation = _synthetic_generation_config(Path(config.data))
    if synthetic_generation is not None:
        benchmark_config["synthetic_generation"] = synthetic_generation
    _write_json(output / "benchmark_config.json", benchmark_config)
    _write_csv(leaderboard, output / "leaderboard.csv")
    _write_csv(intervals, output / "confidence_intervals.csv")
    _write_csv(ablations, output / "ablations.csv")
    _write_csv(segments, output / "segment_metrics.csv")
    _write_csv(diagnostics, output / "candidate_diagnostics.csv")
    (output / "benchmark_report.md").write_text(
        render_benchmark_report(
            dataset=valid_bundle.source_name,
            leaderboard=leaderboard,
            intervals=intervals,
            ablations=ablations,
            segments=segments,
            diagnostics=diagnostics,
            config=benchmark_config,
        ),
        encoding="utf-8",
    )


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Run a complete benchmark and publish it only after every artifact succeeds."""

    _validate_config(config)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.staging-"
    ) as temporary_directory:
        staging = Path(temporary_directory)
        _execute_benchmark(config, staging)
        os.replace(staging, output)
    return output
