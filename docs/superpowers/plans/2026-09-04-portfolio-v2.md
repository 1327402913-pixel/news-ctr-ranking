# News CTR Ranking Portfolio V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible model benchmarking, statistical uncertainty, ablations, segment analysis, persisted rankers, and batch inference so the repository provides stronger evidence for data-analysis and machine-learning roles.

**Architecture:** One audited temporal split feeds training-only baselines and a shared feature builder. Focused evaluation modules assemble point metrics, paired impression-level bootstrap intervals, ablations, and segment diagnostics; persistence stores a trusted fitted ranker for batch scoring. The CLI remains thin and delegates complete workflows to orchestration functions that publish output directories atomically.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, PyArrow, joblib, optional LightGBM, pytest, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-04-portfolio-v2-design.md`

## Global Constraints

- Preserve the existing `audit`, `make-synthetic`, and `train` command interfaces.
- Raw EB-NeRD files and row-level real-data artifacts must never be committed or redistributed.
- Validation labels, validation clicks, and future article aggregates must not affect fitted state.
- Synthetic outputs must say they are engineering evidence and not an EB-NeRD benchmark.
- Core installation must work without LightGBM; `pip install ".[ranking]"` enables it.
- Tests and core CI must support Python 3.10, 3.11, and 3.12.
- Public output directories appear only after every requested benchmark stage succeeds.
- Joblib artifacts are loaded only from trusted sources and carry an explicit schema version.

---

## File Structure

- Create `src/news_ctr/baselines.py`: position and smoothed-popularity estimators.
- Create `src/news_ctr/evaluation.py`: bootstrap intervals, ablations, segments, and candidate diagnostics.
- Create `src/news_ctr/persistence.py`: saved-ranker schema and trusted artifact save/load.
- Create `src/news_ctr/benchmarking.py`: one-split benchmark orchestration and atomic publication.
- Modify `src/news_ctr/features.py`: feature-group registry and validated subset selection.
- Modify `src/news_ctr/metrics.py`: expose metric names used by bootstrap reporting.
- Modify `src/news_ctr/reporting.py`: benchmark tables and Markdown report rendering.
- Modify `src/news_ctr/cli.py`: `benchmark` and `rank` commands; persisted artifacts from `train`.
- Create `tests/test_baselines.py`: baseline behavior and leakage boundaries.
- Create `tests/test_evaluation.py`: bootstrap, ablations, segments, and diagnostics.
- Create `tests/test_persistence.py`: round trips and schema rejection.
- Create `tests/test_benchmarking.py`: benchmark output and atomic failure behavior.
- Modify `tests/test_features.py`: complete feature-group coverage.
- Modify `tests/test_cli.py`: end-to-end benchmark and rank workflows.
- Modify `.github/workflows/publish.yml`: exercise the four-model benchmark on Linux.
- Modify `README.md`, `docs/experiment-protocol.md`, and `Makefile`: recruiter tour and reproducible commands.
- Create `artifacts/portfolio-v2/`: deterministic synthetic summary artifacts only.

---

### Task 1: Feature Group Registry and Ablation Selection

**Files:**
- Modify: `src/news_ctr/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `NewsFeatureBuilder.feature_names_`.
- Produces: `FEATURE_GROUPS: dict[str, tuple[str, ...]]` and `select_feature_names(*, include_groups: Sequence[str] | None = None, exclude_groups: Sequence[str] = ()) -> list[str]`.

- [ ] **Step 1: Write failing feature-group tests**

```python
from news_ctr.features import FEATURE_GROUPS, NewsFeatureBuilder, select_feature_names


def test_feature_groups_cover_every_feature_exactly_once() -> None:
    grouped = [name for names in FEATURE_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(NewsFeatureBuilder.feature_names_)
    assert len(grouped) == len(set(grouped))


def test_feature_selection_supports_ablation_and_rejects_empty_sets() -> None:
    selected = select_feature_names(exclude_groups=("semantic",))
    assert "text_similarity" not in selected
    assert "candidate_position" in selected
    with pytest.raises(ValueError, match="unknown feature groups"):
        select_feature_names(exclude_groups=("missing",))
    with pytest.raises(ValueError, match="must not be empty"):
        select_feature_names(exclude_groups=tuple(FEATURE_GROUPS))
```

- [ ] **Step 2: Verify the tests fail for missing interfaces**

Run: `pytest tests/test_features.py -q`

Expected: collection fails because `FEATURE_GROUPS` and `select_feature_names` do not exist.

- [ ] **Step 3: Add the exact feature groups and validated selector**

```python
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "context": (
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "device_type",
        "is_sso_user",
        "is_subscriber",
        "candidate_position",
        "candidate_count",
    ),
    "article": (
        "publication_age_hours",
        "freshness_log_hours",
        "title_length",
        "subtitle_length",
        "category_code",
        "premium",
        "article_type_code",
        "sentiment_score",
    ),
    "personalization": ("history_length", "category_affinity"),
    "semantic": ("text_similarity", "recent_text_similarity"),
}


def select_feature_names(*, include_groups=None, exclude_groups=()) -> list[str]:
    requested = set(FEATURE_GROUPS if include_groups is None else include_groups)
    excluded = set(exclude_groups)
    unknown = sorted((requested | excluded) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown feature groups: {', '.join(unknown)}")
    selected_groups = requested - excluded
    names = [
        name
        for name in NewsFeatureBuilder.feature_names_
        if any(name in FEATURE_GROUPS[group] for group in selected_groups)
    ]
    if not names:
        raise ValueError("feature selection must not be empty")
    return names
```

Keep this registry in the same order as `NewsFeatureBuilder.feature_names_`; no feature may be inferred by prefix.

- [ ] **Step 4: Run the focused and existing feature tests**

Run: `pytest tests/test_features.py -q`

Expected: all feature tests pass.

- [ ] **Step 5: Commit the feature registry**

```bash
git add src/news_ctr/features.py tests/test_features.py
git commit -m "feat: define auditable feature groups"
```

---

### Task 2: Training-Only Ranking Baselines

**Files:**
- Create: `src/news_ctr/baselines.py`
- Create: `tests/test_baselines.py`

**Interfaces:**
- Consumes candidate rows with `article_id`, `candidate_position`, and `label`.
- Produces `BaselineScorer`, `fit_baseline(kind, train_candidates, *, alpha=20.0) -> BaselineScorer`, and `predict_baseline(model, candidates) -> np.ndarray`.

- [ ] **Step 1: Write failing deterministic baseline tests**

```python
def test_position_baseline_prefers_earlier_candidates() -> None:
    train = pd.DataFrame({"article_id": [1, 2], "candidate_position": [0, 1], "label": [0, 1]})
    model = fit_baseline("position", train)
    np.testing.assert_array_equal(predict_baseline(model, train), np.array([0.0, -1.0]))


def test_popularity_uses_smoothed_training_ctr_and_global_fallback() -> None:
    train = pd.DataFrame(
        {"article_id": [1, 1, 2, 2], "candidate_position": [0, 1, 0, 1], "label": [1, 1, 0, 0]}
    )
    valid = pd.DataFrame({"article_id": [1, 2, 999], "candidate_position": [0, 1, 2]})
    model = fit_baseline("popularity", train, alpha=2.0)
    scores = predict_baseline(model, valid)
    assert scores[0] > scores[2] > scores[1]
    assert scores[2] == pytest.approx(0.5)
```

- [ ] **Step 2: Run the new tests and observe the missing module failure**

Run: `pytest tests/test_baselines.py -q`

Expected: FAIL because `news_ctr.baselines` does not exist.

- [ ] **Step 3: Implement immutable fitted baselines**

```python
@dataclass(frozen=True)
class BaselineScorer:
    kind: str
    global_rate: float | None = None
    article_rates: Mapping[Any, float] = field(default_factory=dict)


def fit_baseline(kind, train_candidates, *, alpha=20.0):
    if kind == "position":
        return BaselineScorer(kind=kind)
    if kind != "popularity":
        raise ValueError("baseline kind must be one of: popularity, position")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    global_rate = float(train_candidates["label"].mean())
    grouped = train_candidates.groupby("article_id")["label"].agg(["sum", "count"])
    rates = (grouped["sum"] + alpha * global_rate) / (grouped["count"] + alpha)
    return BaselineScorer(kind=kind, global_rate=global_rate, article_rates=rates.to_dict())
```

`predict_baseline` returns negative zero-based position for `position`, and mapped smoothed rates with `global_rate` fallback for `popularity`. Validate required columns, binary labels, finite alpha, and non-empty training data.

- [ ] **Step 4: Run baseline tests and the complete suite**

Run: `pytest tests/test_baselines.py -q && pytest -q`

Expected: baseline tests and all existing tests pass.

- [ ] **Step 5: Commit the baselines**

```bash
git add src/news_ctr/baselines.py tests/test_baselines.py
git commit -m "feat: add ranking diagnostic baselines"
```

---

### Task 3: Paired Impression Bootstrap

**Files:**
- Create: `src/news_ctr/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `src/news_ctr/metrics.py`

**Interfaces:**
- Consumes aligned labels, impression IDs, and `Mapping[str, Sequence[float]]` scores.
- Produces `RANKING_METRICS`, `bootstrap_ranking_intervals(labels, scores_by_model, groups, *, samples=200, seed=42, confidence=0.95) -> pd.DataFrame`.

- [ ] **Step 1: Write failing bootstrap tests**

```python
def test_bootstrap_is_deterministic_and_paired() -> None:
    labels = np.array([1, 0, 0, 1, 1, 0])
    groups = np.array([10, 10, 20, 20, 30, 30])
    scores = {
        "good": np.array([0.9, 0.1, 0.1, 0.9, 0.8, 0.2]),
        "bad": np.array([0.1, 0.9, 0.9, 0.1, 0.2, 0.8]),
    }
    first = bootstrap_ranking_intervals(labels, scores, groups, samples=25, seed=7)
    second = bootstrap_ranking_intervals(labels, scores, groups, samples=25, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["model"]) == {"good", "bad"}
    assert (first["lower"] <= first["mean"]).all()
    assert (first["mean"] <= first["upper"]).all()


def test_bootstrap_relabels_duplicate_group_draws() -> None:
    class FixedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 3, 3)
            return np.array([0, 0, 2])

    rows, remapped = _bootstrap_indices(np.array([10, 10, 20, 20, 30, 30]), FixedRng())
    np.testing.assert_array_equal(rows, np.array([0, 1, 0, 1, 4, 5]))
    np.testing.assert_array_equal(remapped, np.array([0, 0, 1, 1, 2, 2]))
```

- [ ] **Step 2: Run the focused tests and verify missing functions fail**

Run: `pytest tests/test_evaluation.py -q`

Expected: FAIL because the evaluation interfaces do not exist.

- [ ] **Step 3: Implement shared group draws with fresh occurrence IDs**

```python
def _bootstrap_indices(groups: np.ndarray, rng: np.random.Generator):
    unique = pd.unique(groups)
    drawn = rng.integers(0, len(unique), size=len(unique))
    row_parts, group_parts = [], []
    for occurrence, group_index in enumerate(drawn):
        rows = np.flatnonzero(groups == unique[group_index])
        row_parts.append(rows)
        group_parts.append(np.full(len(rows), occurrence))
    return np.concatenate(row_parts), np.concatenate(group_parts)
```

Validate samples >= 2, `0 < confidence < 1`, binary labels, finite scores, unique model names, and aligned lengths. Reuse every generated `(row_indices, bootstrap_groups)` pair for all models. Return long-form columns `model`, `metric`, `mean`, `lower`, `upper`, `valid_samples`, and `requested_samples`, sorted by model and metric.

- [ ] **Step 4: Run evaluation and full regression tests**

Run: `pytest tests/test_evaluation.py -q && pytest -q`

Expected: deterministic bootstrap tests and all regressions pass.

- [ ] **Step 5: Commit bootstrap evaluation**

```bash
git add src/news_ctr/evaluation.py src/news_ctr/metrics.py tests/test_evaluation.py
git commit -m "feat: add paired impression bootstrap intervals"
```

---

### Task 4: Ablations, Segments, and Bias Diagnostics

**Files:**
- Modify: `src/news_ctr/evaluation.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes training/validation features and candidates from one split.
- Produces `run_logistic_ablations(X_train, y_train, train_groups, X_valid, y_valid, valid_groups, *, seed=42) -> pd.DataFrame`.
- Produces `segment_ranking_metrics(candidates, scores, features, *, min_impressions=5) -> pd.DataFrame`.
- Produces `candidate_bias_diagnostics(candidates, features) -> pd.DataFrame`.

- [ ] **Step 1: Add failing tests for every declared analysis table**

```python
def test_ablations_use_declared_feature_subsets() -> None:
    table = run_logistic_ablations(
        X_train, y_train, train_groups, X_valid, y_valid, valid_groups, seed=3
    )
    assert table["ablation"].tolist() == [
        "context_only",
        "no_position",
        "no_semantic",
        "no_personalization",
        "full",
    ]
    assert (
        table.loc[table["ablation"] == "no_position", "feature_count"].item()
        == X_train.shape[1] - 1
    )


def test_segment_metrics_flag_low_support_without_splitting_impressions() -> None:
    result = segment_ranking_metrics(candidates, scores, features, min_impressions=5)
    assert {"device_type", "candidate_count", "history_length"} <= set(result["segment"])
    assert result["low_support"].dtype == bool


def test_candidate_diagnostics_are_click_rates_not_ranking_metrics() -> None:
    result = candidate_bias_diagnostics(candidates, features)
    assert set(result["diagnostic"]) == {"candidate_position", "article_freshness"}
    assert "ndcg@5" not in result.columns
```

- [ ] **Step 2: Run the focused tests and confirm interface failures**

Run: `pytest tests/test_evaluation.py -q`

Expected: FAIL for the three missing analysis functions.

- [ ] **Step 3: Implement the fixed ablation matrix**

Use these exact selections:

```python
ABLATIONS = {
    "context_only": {"include_groups": ("context",)},
    "no_position": {"exclude_features": ("candidate_position",)},
    "no_semantic": {"exclude_groups": ("semantic",)},
    "no_personalization": {"exclude_groups": ("personalization",)},
    "full": {},
}
```

Each variant calls `fit_model("logistic", X_train[selected], y_train, train_groups, seed=seed)`, predicts with `X_valid[selected]`, records `feature_count`, and adds point ranking metrics. Input frames must have identical ordered columns.

- [ ] **Step 4: Implement impression-level segments and candidate diagnostics**

Bucket candidate count as `small` (<=5), `medium` (6-10), or `large` (>10); history length as `cold` (0), `short` (1-5), or `long` (>5); freshness as `<24h`, `1-3d`, `3-7d`, or `7d+`. Ranking slices use only impression-constant device, candidate-count, and history buckets. Diagnostics aggregate candidate rows by position or freshness and return `candidates`, `clicks`, and `click_rate`.

- [ ] **Step 5: Run focused and full tests**

Run: `pytest tests/test_evaluation.py -q && pytest -q`

Expected: all evaluation and regression tests pass.

- [ ] **Step 6: Commit analysis functions**

```bash
git add src/news_ctr/evaluation.py tests/test_evaluation.py
git commit -m "feat: add ablation and segment analysis"
```

---

### Task 5: Trusted Saved-Ranker Artifacts

**Files:**
- Create: `src/news_ctr/persistence.py`
- Create: `tests/test_persistence.py`
- Modify: `src/news_ctr/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes `NewsFeatureBuilder`, `TrainedModel`, feature names, and run metadata.
- Produces `SavedRanker`, `save_ranker(path, ranker) -> tuple[Path, Path]`, `load_ranker(path) -> SavedRanker`, and new `train` artifacts `model.joblib` plus `model_metadata.json`.

- [ ] **Step 1: Write failing persistence tests**

```python
def test_saved_ranker_round_trip_preserves_scores(tmp_path) -> None:
    saved = SavedRanker(
        schema_version=1,
        model=model,
        feature_builder=builder,
        feature_names=tuple(builder.feature_names_),
        metadata={"dataset_fingerprint": "abc", "model": "logistic"},
    )
    model_path, metadata_path = save_ranker(tmp_path, saved)
    restored = load_ranker(model_path)
    np.testing.assert_allclose(
        predict_scores(saved.model, saved.feature_builder.transform(valid)),
        predict_scores(restored.model, restored.feature_builder.transform(valid)),
    )
    assert json.loads(metadata_path.read_text())["schema_version"] == 1


def test_load_rejects_incompatible_schema(tmp_path) -> None:
    joblib.dump({"schema_version": 99}, tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="unsupported saved-ranker schema"):
        load_ranker(tmp_path / "model.joblib")
```

- [ ] **Step 2: Run persistence tests and verify the missing module failure**

Run: `pytest tests/test_persistence.py -q`

Expected: FAIL because `news_ctr.persistence` does not exist.

- [ ] **Step 3: Implement schema-versioned save/load**

```python
SAVED_RANKER_SCHEMA_VERSION = 1


@dataclass
class SavedRanker:
    schema_version: int
    model: TrainedModel
    feature_builder: NewsFeatureBuilder
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
```

`save_ranker` writes joblib to a temporary sibling and replaces `model.joblib`, then writes JSON metadata through the existing JSON-safe conversion path. `load_ranker` accepts a file or run directory, verifies the object type and exact schema version, and raises a concise `ValueError` for corrupt/incompatible artifacts without hiding the trust warning in documentation.

- [ ] **Step 4: Make `train` emit both model artifacts**

Construct `SavedRanker` after fitting, recording Python, scikit-learn, and LightGBM versions when present. Extend `tests/test_cli.py` expected artifacts and load the emitted model to verify score reproducibility.

- [ ] **Step 5: Run persistence, CLI, and full tests**

Run: `pytest tests/test_persistence.py tests/test_cli.py -q && pytest -q`

Expected: new artifact tests and all existing tests pass.

- [ ] **Step 6: Commit persistence**

```bash
git add src/news_ctr/persistence.py src/news_ctr/cli.py tests/test_persistence.py tests/test_cli.py
git commit -m "feat: persist reproducible ranker artifacts"
```

---

### Task 6: Batch Ranking API and CLI

**Files:**
- Modify: `src/news_ctr/persistence.py`
- Modify: `src/news_ctr/cli.py`
- Modify: `tests/test_persistence.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes a trusted `SavedRanker` and either behavior rows or expanded candidates.
- Produces `rank_candidates(ranker, candidates, *, top_k: int | None = None) -> pd.DataFrame` and `news-ctr rank`.

- [ ] **Step 1: Write failing ranking tests**

```python
def test_rank_candidates_orders_each_impression_and_applies_top_k() -> None:
    ranked = rank_candidates(saved_ranker, expanded_candidates, top_k=2)
    assert ranked.columns.tolist() == ["impression_id", "article_id", "score", "rank"]
    assert ranked.groupby("impression_id").size().eq(2).all()
    rank_lists = ranked.groupby("impression_id")["rank"].apply(list)
    assert rank_lists.map(lambda values: values == [1, 2]).all()
    assert ranked.groupby("impression_id")["score"].apply(lambda s: s.is_monotonic_decreasing).all()


def test_rank_cli_expands_behavior_parquet(tmp_path) -> None:
    code = main(
        [
            "rank",
            "--model",
            str(model_path),
            "--candidates",
            str(behaviors_path),
            "--output",
            str(output),
            "--top-k",
            "3",
        ]
    )
    assert code == 0
    ranked = pd.read_parquet(output)
    assert ranked.groupby("impression_id").size().eq(3).all()
    assert "label" not in ranked
```

- [ ] **Step 2: Run focused tests and verify missing rank behavior**

Run: `pytest tests/test_persistence.py tests/test_cli.py -q`

Expected: FAIL because `rank_candidates` and the `rank` parser are missing.

- [ ] **Step 3: Implement deterministic within-impression scoring**

Validate `top_k` is positive, transform with the stored builder, select the stored feature names, score, stably sort by `impression_id`, descending `score`, and original candidate order, then assign ranks using `ranked.groupby("impression_id").cumcount() + 1`.

- [ ] **Step 4: Add the CLI parser and behavior expansion**

```python
rank = subparsers.add_parser("rank", help="score candidates with a trusted saved ranker")
rank.add_argument("--model", type=Path, required=True)
rank.add_argument("--candidates", type=Path, required=True)
rank.add_argument("--output", type=Path, required=True)
rank.add_argument("--top-k", type=int)
```

Read Parquet; call `expand_candidates` when list-valued `article_ids_inview` is present, otherwise require expanded candidate columns. Create the output parent only after scoring succeeds.

- [ ] **Step 5: Run focused and complete tests**

Run: `pytest tests/test_persistence.py tests/test_cli.py -q && pytest -q`

Expected: all batch ranking and regression tests pass.

- [ ] **Step 6: Commit batch inference**

```bash
git add src/news_ctr/persistence.py src/news_ctr/cli.py tests/test_persistence.py tests/test_cli.py
git commit -m "feat: add batch candidate ranking"
```

---

### Task 7: Benchmark Orchestration and Atomic Output

**Files:**
- Create: `src/news_ctr/benchmarking.py`
- Create: `tests/test_benchmarking.py`
- Modify: `src/news_ctr/reporting.py`

**Interfaces:**
- Consumes a dataset path, output path, ordered model names, bootstrap count, feature components, split fraction, and seed.
- Produces `BenchmarkConfig`, `run_benchmark(config) -> Path`, and complete benchmark artifacts from the approved spec.

- [ ] **Step 1: Write a failing end-to-end orchestration test**

```python
def test_benchmark_writes_reproducible_complete_artifacts(tmp_path) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data", seed=11, n_users=12, n_articles=30, n_impressions=60
    )
    first = run_benchmark(
        BenchmarkConfig(
            data=data,
            output=tmp_path / "first",
            models=("position", "popularity", "logistic"),
            bootstrap_samples=20,
            text_components=4,
            seed=11,
        )
    )
    second = run_benchmark(
        BenchmarkConfig(
            data=data,
            output=tmp_path / "second",
            models=("position", "popularity", "logistic"),
            bootstrap_samples=20,
            text_components=4,
            seed=11,
        )
    )
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
    pd.testing.assert_frame_equal(
        pd.read_csv(first / "leaderboard.csv"),
        pd.read_csv(second / "leaderboard.csv"),
        check_exact=False,
        rtol=1e-8,
    )
```

- [ ] **Step 2: Write a failing atomic-publication test**

```python
def test_failed_benchmark_does_not_publish_output(tmp_path, monkeypatch) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data", seed=3, n_users=8, n_articles=20, n_impressions=40
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
    monkeypatch.setattr(benchmarking, "fit_model", Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        run_benchmark(config)
    assert not output.exists()
```

- [ ] **Step 3: Run tests and verify orchestration is absent**

Run: `pytest tests/test_benchmarking.py -q`

Expected: FAIL because `news_ctr.benchmarking` does not exist.

- [ ] **Step 4: Implement one shared split and model loop**

`BenchmarkConfig` is a frozen dataclass. Validate ordered unique models from `position`, `popularity`, `logistic`, and `lightgbm`; require an output path that does not already exist. Load/audit data once, split once, fit the feature builder once, and preserve aligned validation candidates for every score vector.

For baseline runs call `fit_baseline`; for fitted ML runs call `fit_model`, save predictions/metrics/importance, and persist the ranker. Record `fit_seconds`, `score_ms_per_1000`, and `model_bytes` in the leaderboard; tests compare deterministic statistical columns but only require non-negative timing columns.

- [ ] **Step 5: Assemble intervals, ablations, and segments**

Call the Task 3 and Task 4 interfaces with aligned labels/groups. Use logistic scores for segment ranking tables and candidate diagnostics. Store all tables with stable row order and fixed float serialization.

- [ ] **Step 6: Publish atomically**

Create a temporary sibling with `tempfile.TemporaryDirectory(dir=output.parent)`, write every run/table/report there, then rename the completed directory to `output`. Any exception removes the temporary directory and leaves `output` absent.

- [ ] **Step 7: Run orchestration and full tests**

Run: `pytest tests/test_benchmarking.py -q && pytest -q`

Expected: benchmark artifacts are reproducible and failure leaves no published output.

- [ ] **Step 8: Commit orchestration**

```bash
git add src/news_ctr/benchmarking.py src/news_ctr/reporting.py tests/test_benchmarking.py
git commit -m "feat: orchestrate reproducible ranking benchmarks"
```

---

### Task 8: Benchmark Report and CLI

**Files:**
- Modify: `src/news_ctr/reporting.py`
- Modify: `src/news_ctr/cli.py`
- Modify: `tests/test_benchmarking.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes benchmark tables and configuration.
- Produces `render_benchmark_report(*, dataset, leaderboard, intervals, ablations, segments, diagnostics, config) -> str` and `news-ctr benchmark`.

- [ ] **Step 1: Write failing report truthfulness tests**

```python
def test_synthetic_benchmark_report_is_recruiter_readable_and_truthful() -> None:
    report = render_benchmark_report(
        dataset="synthetic:validation",
        leaderboard=leaderboard,
        intervals=intervals,
        ablations=ablations,
        segments=segments,
        diagnostics=diagnostics,
        config=config,
    )
    assert "60-second summary" in report
    assert "not an EB-NeRD benchmark" in report
    assert "95%" in report
    assert "Exposure-bias diagnostic" in report
    assert "| Model |" in report
```

- [ ] **Step 2: Write a failing CLI artifact test**

```python
def test_benchmark_cli_creates_the_declared_report(tmp_path) -> None:
    code = main(
        [
            "benchmark",
            "--data",
            str(data),
            "--output",
            str(output),
            "--models",
            "position,popularity,logistic",
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
```

- [ ] **Step 3: Run focused tests and observe missing report/CLI behavior**

Run: `pytest tests/test_benchmarking.py tests/test_cli.py -q`

Expected: FAIL because report rendering and parser behavior are missing.

- [ ] **Step 4: Render a self-contained Markdown report**

Include the required synthetic/real-data banner, ranked leaderboard, 95% intervals, ablation deltas against `full`, low-support segment flags, exposure-bias diagnostics, latency/model-size table, leakage safeguards, limitations, and exact reproduction command. Format values through one helper so `None` becomes `N/A` and finite floats use six decimals.

- [ ] **Step 5: Add the benchmark parser**

```python
benchmark = subparsers.add_parser("benchmark", help="compare ranking baselines and models")
benchmark.add_argument("--data", type=Path, required=True)
benchmark.add_argument("--output", type=Path, required=True)
benchmark.add_argument("--models", default="position,popularity,logistic")
benchmark.add_argument("--bootstrap-samples", type=int, default=200)
benchmark.add_argument("--seed", type=int, default=42)
benchmark.add_argument("--text-components", type=int, default=32)
benchmark.add_argument("--valid-fraction", type=float, default=0.2)
```

Parse the comma-separated list by trimming whitespace, rejecting blanks/duplicates, and preserving order. Delegate to `run_benchmark` and print one sorted JSON summary.

- [ ] **Step 6: Run CLI, report, and full tests**

Run: `pytest tests/test_benchmarking.py tests/test_cli.py -q && pytest -q`

Expected: all report and CLI tests pass with no regressions.

- [ ] **Step 7: Commit report and CLI**

```bash
git add src/news_ctr/reporting.py src/news_ctr/cli.py tests/test_benchmarking.py tests/test_cli.py
git commit -m "feat: expose portfolio benchmark workflow"
```

---

### Task 9: Generate Portfolio V2 Evidence

**Files:**
- Create: `artifacts/portfolio-v2/benchmark_config.json`
- Create: `artifacts/portfolio-v2/leaderboard.csv`
- Create: `artifacts/portfolio-v2/confidence_intervals.csv`
- Create: `artifacts/portfolio-v2/ablations.csv`
- Create: `artifacts/portfolio-v2/segment_metrics.csv`
- Create: `artifacts/portfolio-v2/candidate_diagnostics.csv`
- Create: `artifacts/portfolio-v2/benchmark_report.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes the public CLI and deterministic synthetic fixture.
- Produces recruiter-visible summary artifacts while keeping row-level predictions and serialized models untracked.

- [ ] **Step 1: Extend ignore rules before generating artifacts**

Ignore `artifacts/portfolio-v2/runs/`, `model.joblib`, `model_metadata.json`, and benchmark temporary directories. Keep only aggregate tables/config/report under `artifacts/portfolio-v2/`.

- [ ] **Step 2: Run the exact deterministic benchmark**

```bash
news-ctr make-synthetic --output data/portfolio-v2 --seed 42 --users 30 --articles 80 --impressions 180
news-ctr benchmark --data data/portfolio-v2 --output artifacts/portfolio-v2 --models position,popularity,logistic,lightgbm --bootstrap-samples 200 --seed 42
```

Expected: command exits 0 and the report banner says the result is synthetic engineering evidence.

- [ ] **Step 3: Verify tracked artifact boundaries**

Run: `git status --short --ignored artifacts/portfolio-v2 data/portfolio-v2`

Expected: aggregate CSV/JSON/Markdown files are untracked for addition; `data/portfolio-v2`, `runs/`, predictions, and model files are ignored.

- [ ] **Step 4: Commit generated aggregate evidence**

```bash
git add .gitignore artifacts/portfolio-v2
git commit -m "docs: add reproducible portfolio benchmark evidence"
```

---

### Task 10: Recruiter-Facing Documentation and CI

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment-protocol.md`
- Modify: `Makefile`
- Modify: `.github/workflows/publish.yml`

**Interfaces:**
- Consumes the tested public CLI and committed V2 aggregate artifacts.
- Produces a 60-second recruiter tour, truthful resume bullets, reproduction commands, and Linux verification of the full model path.

- [ ] **Step 1: Update README from generated evidence**

Add, near the top, the business question, latest synthetic leaderboard, winning-model delta with interval context, three engineering decisions, a prominent synthetic-data limitation, and links to the full report. Add two resume bullets that describe what was built without claiming production lift or a real EB-NeRD score.

- [ ] **Step 2: Document benchmark and trusted inference commands**

Add exact commands for core synthetic benchmark, LightGBM benchmark, licensed EB-NeRD benchmark, persisted `train`, and `rank`. State that joblib is pickle-based and must not load untrusted files.

- [ ] **Step 3: Update experiment protocol and Make targets**

Document paired impression bootstrap, ablation definitions, segment interpretation, and the difference between candidate diagnostics and ranking metrics. Add `make benchmark` for the core model list and `make benchmark-ranking` for all four models.

- [ ] **Step 4: Strengthen CI**

Keep Python 3.10-3.12 core jobs. In `ranking-extra`, replace the single LightGBM train smoke with a small four-model benchmark using 20 bootstrap samples, then assert `leaderboard.csv` contains `lightgbm` and the report contains the synthetic warning.

- [ ] **Step 5: Run documentation commands and CI-equivalent checks**

```bash
ruff check .
ruff format --check .
pytest --cov=news_ctr --cov-report=term-missing
python -m build
news-ctr --help
news-ctr benchmark --help
news-ctr rank --help
```

Expected: lint and formatting pass, all tests pass with at least 85% coverage, package builds, and all three help commands exit 0.

- [ ] **Step 6: Commit documentation and CI**

```bash
git add README.md docs/experiment-protocol.md Makefile .github/workflows/publish.yml
git commit -m "docs: present portfolio v2 case study"
```

---

### Task 11: Final Verification, Review, Merge, and Publish

**Files:**
- Verify all tracked project files.
- Modify no source files unless verification exposes a defect, in which case return to the failing task's TDD cycle.

**Interfaces:**
- Consumes the complete feature branch.
- Produces a reviewed main branch and a public GitHub revision with green CI.

- [ ] **Step 1: Run the complete local verification suite from a clean shell**

```bash
python -m pytest --cov=news_ctr --cov-report=term-missing
ruff check .
ruff format --check .
python -m build
git diff --check main...HEAD
git status --short
```

Expected: zero failures, at least 85% coverage, successful sdist/wheel build, no whitespace errors, and only intended ignored build outputs.

- [ ] **Step 2: Run an independent requirements review**

Compare every acceptance criterion in `docs/superpowers/specs/2026-09-04-portfolio-v2-design.md` to implementation evidence. Record any mismatch as a concrete finding; fix findings through the corresponding test-first task before continuing.

- [ ] **Step 3: Merge the isolated feature branch into local `main`**

Use the `superpowers:finishing-a-development-branch` workflow. Re-run the complete suite on merged `main`; do not delete the feature worktree until merged verification succeeds.

- [ ] **Step 4: Publish the exact merged tree to GitHub**

Update `https://github.com/1327402913-pixel/news-ctr-ranking` without exposing credentials or committing raw data/model binaries. Browser actions that create public commits require action-time confirmation under the browser policy.

- [ ] **Step 5: Verify remote content and CI**

Fetch public `origin/main`, compare its tree hash with local `HEAD^{tree}`, and verify the latest GitHub Actions run has successful Python 3.10, 3.11, 3.12, and `ranking-extra` jobs.

- [ ] **Step 6: Report the finished portfolio evidence**

Provide the public repository link, full benchmark report link, local README link, test/coverage counts, and the remaining licensed-data limitation. Do not describe synthetic metrics as real recommendation performance.
