"""Command-line entry points for auditing and running news ranking experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from news_ctr.benchmarking import BenchmarkConfig, run_benchmark
from news_ctr.data import (
    DataContractError,
    DatasetBundle,
    audit_bundle,
    expand_candidates,
    expand_scoring_candidates,
    load_bundle,
    temporal_split,
    write_synthetic_bundle,
)
from news_ctr.features import NewsFeatureBuilder
from news_ctr.metrics import ranking_metrics
from news_ctr.models import feature_importance, fit_model, predict_scores
from news_ctr.persistence import (
    SAVED_RANKER_SCHEMA_VERSION,
    SavedRanker,
    load_ranker,
    rank_candidates,
    save_ranker,
)
from news_ctr.reporting import dataset_fingerprint, write_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-ctr",
        description="Leakage-safe news click ranking experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "make-synthetic", help="create a deterministic EB-NeRD-shaped smoke dataset"
    )
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--users", type=int, default=30)
    synthetic.add_argument("--articles", type=int, default=80)
    synthetic.add_argument("--impressions", type=int, default=180)
    synthetic.set_defaults(handler=_make_synthetic)

    audit = subparsers.add_parser("audit", help="validate a dataset bundle")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--split", choices=("train", "validation"), default="train")
    audit.set_defaults(handler=_audit)

    train = subparsers.add_parser("train", help="train and evaluate a ranking model")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", choices=("logistic", "lightgbm"), default="logistic")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--text-components", type=int, default=32)
    train.add_argument("--valid-fraction", type=float, default=0.2)
    train.set_defaults(handler=_train)

    rank = subparsers.add_parser("rank", help="score candidates with a trusted saved ranker")
    rank.add_argument("--model", type=Path, required=True)
    rank.add_argument("--candidates", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--top-k", type=int)
    rank.set_defaults(handler=_rank)

    benchmark = subparsers.add_parser("benchmark", help="compare ranking baselines and models")
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--models", default="position,popularity,logistic")
    benchmark.add_argument("--bootstrap-samples", type=int, default=200)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--text-components", type=int, default=32)
    benchmark.add_argument("--valid-fraction", type=float, default=0.2)
    benchmark.set_defaults(handler=_benchmark)

    analyze = subparsers.add_parser("analyze", help="run reproducible SQL product analytics")
    analyze.add_argument("--data", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--split", choices=("train", "validation"), default="validation")
    analyze.add_argument("--min-cell-count", type=int, default=5)
    analyze.set_defaults(handler=_analyze)

    experiment_design = subparsers.add_parser(
        "experiment-design", help="size a two-arm binary-outcome experiment"
    )
    experiment_design.add_argument("--baseline-rate", type=float, required=True)
    experiment_design.add_argument("--relative-mde", type=float, required=True)
    experiment_design.add_argument("--alpha", type=float, default=0.05)
    experiment_design.add_argument("--power", type=float, default=0.80)
    experiment_design.add_argument("--daily-units", type=int)
    experiment_design.set_defaults(handler=_experiment_design)

    make_experiment = subparsers.add_parser(
        "make-experiment", help="create a deterministic randomized teaching fixture"
    )
    make_experiment.add_argument("--output", type=Path, required=True)
    make_experiment.add_argument("--seed", type=int, default=42)
    make_experiment.add_argument("--users", type=int, default=20_000)
    make_experiment.add_argument("--corrupt-allocation", action="store_true")
    make_experiment.set_defaults(handler=_make_experiment)

    experiment = subparsers.add_parser(
        "experiment", help="analyze randomized evidence and make a launch decision"
    )
    experiment.add_argument("--input", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--config", type=Path, required=True)
    experiment.set_defaults(handler=_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and convert expected user errors to concise exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (DataContractError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _make_synthetic(args: argparse.Namespace) -> int:
    path = write_synthetic_bundle(
        args.output,
        seed=args.seed,
        n_users=args.users,
        n_articles=args.articles,
        n_impressions=args.impressions,
    )
    print(json.dumps({"dataset": str(path), "source": "synthetic"}, sort_keys=True))
    return 0


def _audit(args: argparse.Namespace) -> int:
    summary = audit_bundle(load_bundle(args.data, split=args.split))
    print(json.dumps(summary, sort_keys=True))
    return 0


def _rank(args: argparse.Namespace) -> int:
    ranker = load_ranker(args.model)
    candidates = pd.read_parquet(args.candidates)
    if "article_ids_inview" in candidates:
        candidates = expand_scoring_candidates(candidates)
    ranked = rank_candidates(ranker, candidates, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(args.output, index=False)
    print(
        json.dumps(
            {
                "candidates": len(ranked),
                "impressions": int(ranked["impression_id"].nunique(dropna=False)),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    raw_models = args.models.split(",")
    models = tuple(name.strip() for name in raw_models)
    if not models or any(not name for name in models):
        raise ValueError("models must be a comma-separated list without blank names")
    if len(models) != len(set(models)):
        raise ValueError("benchmark model list contains a duplicate model")
    output = run_benchmark(
        BenchmarkConfig(
            data=args.data,
            output=args.output,
            models=models,
            bootstrap_samples=args.bootstrap_samples,
            text_components=args.text_components,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        )
    )
    print(json.dumps({"models": list(models), "output": str(output)}, sort_keys=True))
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from news_ctr.analytics import AnalyticsConfig, run_analytics

    output = run_analytics(
        AnalyticsConfig(
            data=args.data,
            output=args.output,
            split=args.split,
            min_cell_count=args.min_cell_count,
        )
    )
    print(json.dumps({"output": str(output), "split": args.split}, sort_keys=True))
    return 0


def _experiment_design(args: argparse.Namespace) -> int:
    from news_ctr.experiments import ExperimentDesign, required_sample_size

    if args.relative_mde <= 0:
        raise ValueError("relative_mde must be positive")
    if args.daily_units is not None and args.daily_units <= 0:
        raise ValueError("daily_units must be positive")
    absolute_mde = args.baseline_rate * args.relative_mde
    units_per_arm = required_sample_size(
        ExperimentDesign(
            baseline_rate=args.baseline_rate,
            absolute_mde=absolute_mde,
            alpha=args.alpha,
            power=args.power,
        )
    )
    payload: dict[str, Any] = {
        "absolute_mde": absolute_mde,
        "alpha": args.alpha,
        "baseline_rate": args.baseline_rate,
        "power": args.power,
        "relative_mde": args.relative_mde,
        "total_units": 2 * units_per_arm,
        "units_per_arm": units_per_arm,
    }
    if args.daily_units is not None:
        payload["daily_units"] = args.daily_units
        payload["estimated_days"] = int(np.ceil(payload["total_units"] / args.daily_units))
    print(json.dumps(payload, sort_keys=True))
    return 0


def _make_experiment(args: argparse.Namespace) -> int:
    from news_ctr.experiment_data import write_synthetic_experiment

    output = write_synthetic_experiment(
        args.output,
        seed=args.seed,
        users=args.users,
        corrupt_allocation=args.corrupt_allocation,
    )
    print(json.dumps({"evidence_tier": "synthetic-rct", "output": str(output)}, sort_keys=True))
    return 0


def _experiment(args: argparse.Namespace) -> int:
    from news_ctr.experiments import run_experiment

    output = run_experiment(args.input, args.output, args.config)
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


def _train(args: argparse.Namespace) -> int:
    train_bundle = load_bundle(args.data, split="train")
    train_summary = audit_bundle(train_bundle)
    train_candidates = expand_candidates(train_bundle.behaviors)

    validation_path = args.data / "validation" / "behaviors.parquet"
    if validation_path.is_file():
        valid_bundle = load_bundle(args.data, split="validation")
        valid_summary = audit_bundle(valid_bundle)
        valid_candidates = expand_candidates(valid_bundle.behaviors)
        if pd.Timestamp(train_candidates["impression_time"].max()) >= pd.Timestamp(
            valid_candidates["impression_time"].min()
        ):
            raise DataContractError(
                "training impressions must end before validation impressions begin"
            )
    else:
        train_candidates, valid_candidates = temporal_split(
            train_candidates, valid_fraction=args.valid_fraction
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

    feature_builder = NewsFeatureBuilder(
        n_components=args.text_components, random_state=args.seed
    ).fit(train_bundle.articles, train_bundle.history, train_candidates)
    X_train = feature_builder.transform(train_candidates)
    X_valid = feature_builder.transform(valid_candidates)
    model = fit_model(
        args.model,
        X_train,
        train_candidates["label"].to_numpy(),
        train_candidates["impression_id"].to_numpy(),
        seed=args.seed,
    )
    scores = predict_scores(model, X_valid)
    metrics: dict[str, Any] = ranking_metrics(
        valid_candidates["label"].to_numpy(),
        scores,
        valid_candidates["impression_id"].to_numpy(),
    )
    if args.model == "logistic":
        metrics["log_loss"] = float(
            log_loss(valid_candidates["label"], np.clip(scores, 1e-12, 1 - 1e-12))
        )

    predictions = valid_candidates[["impression_id", "article_id", "label"]].copy()
    predictions["score"] = scores
    importance = feature_importance(model, feature_builder.feature_names_)
    config = {
        "data": str(args.data),
        "output": str(args.output),
        "model": args.model,
        "seed": args.seed,
        "text_components": args.text_components,
        "valid_fraction": args.valid_fraction,
        "dataset_fingerprint": dataset_fingerprint(args.data),
        "train_source": train_bundle.source_name,
        "train_rows": len(train_candidates),
        "validation_rows": len(valid_candidates),
        "features": feature_builder.feature_names_,
    }
    write_run_artifacts(
        args.output,
        dataset=valid_bundle.source_name,
        dataset_summary=valid_summary,
        model_kind=args.model,
        metrics=metrics,
        predictions=predictions,
        importance=importance,
        run_config=config,
    )
    library_versions = {
        "python": platform.python_version(),
        "scikit-learn": version("scikit-learn"),
    }
    with suppress(PackageNotFoundError):
        library_versions["lightgbm"] = version("lightgbm")
    save_ranker(
        args.output,
        SavedRanker(
            schema_version=SAVED_RANKER_SCHEMA_VERSION,
            model=model,
            feature_builder=feature_builder,
            feature_names=tuple(feature_builder.feature_names_),
            metadata={
                "dataset": valid_bundle.source_name,
                "dataset_fingerprint": config["dataset_fingerprint"],
                "model": args.model,
                "seed": args.seed,
                "library_versions": library_versions,
                "training_config": config,
            },
        ),
    )
    print(
        json.dumps(
            {
                "dataset": valid_bundle.source_name,
                "metrics": metrics,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
