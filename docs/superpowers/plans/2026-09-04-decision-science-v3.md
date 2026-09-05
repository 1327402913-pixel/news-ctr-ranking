# Decision Science V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible SQL analytics and randomized-experiment decision workflow that turns the existing ranking benchmark into a portfolio-grade decision-science case study.

**Architecture:** The existing EB-NeRD-shaped loader remains the source of truth for ranking data. A packaged DuckDB SQL layer produces aggregate product analytics, while a separate user-randomized experiment pipeline validates assignment, estimates ITT and CUPED effects, applies a predeclared decision policy, and publishes reports atomically. Both flows expose thin CLI adapters and clearly label synthetic, observational, and licensed evidence.

**Tech Stack:** Python 3.10+, pandas, NumPy, SciPy, DuckDB, Matplotlib, PyArrow, pytest, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-04-decision-science-v3-design.md`

## Global Constraints

- Keep the existing ranking, persistence, and benchmark interfaces backward compatible.
- Use user-level randomization and one analysis row per randomization unit.
- Never claim online lift or causal impact from observational EB-NeRD clicks.
- Keep licensed raw and row-level data under ignored `data/`; publish aggregates only.
- Label outputs exactly as `synthetic-rct`, `synthetic-observational`, or `licensed-ebnerd`.
- Suppress or flag segment cells below the configured minimum support.
- Publish output directories atomically and refuse to overwrite completed evidence.
- Keep decision-science dependencies in the optional `decision` extra.
- Maintain at least 90% package coverage and Python 3.10–3.12 compatibility.
- Do not add deep neural recommenders, HTTP serving, Docker, Kubernetes, or streaming.

---

### Task 1: Decision-science dependencies and evidence metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/news_ctr/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: dataset source names such as `synthetic:validation` and experiment metadata.
- Produces: `EvidenceTier`, `classify_dataset_evidence(source_name: str) -> EvidenceTier`, and `evidence_banner(tier: EvidenceTier) -> str`.

- [ ] **Step 1: Write failing evidence-classification tests**

```python
from news_ctr.evidence import EvidenceTier, classify_dataset_evidence, evidence_banner


def test_dataset_evidence_is_explicit_and_truthful() -> None:
    assert classify_dataset_evidence("synthetic:validation") is EvidenceTier.SYNTHETIC_OBSERVATIONAL
    assert classify_dataset_evidence("ebnerd:validation") is EvidenceTier.LICENSED_EBNERD
    assert "not production lift" in evidence_banner(EvidenceTier.SYNTHETIC_RCT).lower()


def test_unknown_dataset_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown evidence source"):
        classify_dataset_evidence("mystery:validation")
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -q`

Expected: FAIL because `news_ctr.evidence` does not exist.

- [ ] **Step 3: Implement the evidence model and optional dependency group**

```python
class EvidenceTier(str, Enum):
    SYNTHETIC_RCT = "synthetic-rct"
    SYNTHETIC_OBSERVATIONAL = "synthetic-observational"
    LICENSED_EBNERD = "licensed-ebnerd"


def classify_dataset_evidence(source_name: str) -> EvidenceTier:
    if source_name.startswith("synthetic:"):
        return EvidenceTier.SYNTHETIC_OBSERVATIONAL
    if source_name.startswith("ebnerd:"):
        return EvidenceTier.LICENSED_EBNERD
    raise ValueError(f"unknown evidence source: {source_name}")
```

Add `decision = ["duckdb>=1.1", "matplotlib>=3.9", "scipy>=1.14"]` and package data for `sql/*.sql`. Ignore generated `data/*.parquet`, `artifacts/analytics-*`, and `artifacts/experiment-*` without ignoring committed portfolio evidence directories.

- [ ] **Step 4: Run focused tests and metadata validation**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -q && .venv/bin/python -m build`

Expected: PASS and both sdist and wheel contain the package.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/news_ctr/evidence.py tests/test_evidence.py
git commit -m "feat: classify decision science evidence"
```

---

### Task 2: DuckDB SQL analytics engine

**Files:**
- Create: `src/news_ctr/sql/data_quality.sql`
- Create: `src/news_ctr/sql/kpi_summary.sql`
- Create: `src/news_ctr/sql/exposure_funnel.sql`
- Create: `src/news_ctr/sql/segment_kpis.sql`
- Create: `src/news_ctr/sql/user_cohorts.sql`
- Create: `src/news_ctr/analytics.py`
- Create: `tests/test_analytics.py`

**Interfaces:**
- Consumes: `load_bundle(path: str | Path, *, split: str = "train")`, `expand_candidates(behaviors: pd.DataFrame)`, and `dataset_fingerprint(path: str | Path)`.
- Produces: `AnalyticsConfig(data: Path, output: Path, split: str = "validation", min_cell_count: int = 5)` and `run_analytics(config: AnalyticsConfig) -> Path`.

- [ ] **Step 1: Write failing tests for hand-computed SQL outputs**

```python
def test_run_analytics_computes_kpis_and_segments(tmp_path: Path) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data", seed=7, n_users=8, n_articles=20, n_impressions=40
    )
    output = run_analytics(
        AnalyticsConfig(data=data, output=tmp_path / "analytics", min_cell_count=2)
    )
    summary = pd.read_csv(output / "kpi_summary.csv").iloc[0]
    assert summary["impressions"] == 8
    assert summary["clicks"] == 8
    assert summary["ctr"] == pytest.approx(1 / 6)
    assert set(pd.read_csv(output / "segment_kpis.csv")["segment"]) >= {
        "device_type",
        "candidate_position",
    }


def test_small_analytics_cells_are_flagged(tmp_path: Path) -> None:
    data = write_synthetic_bundle(
        tmp_path / "data", seed=8, n_users=8, n_articles=20, n_impressions=40
    )
    output = run_analytics(
        AnalyticsConfig(data=data, output=tmp_path / "analytics", min_cell_count=10)
    )
    segments = pd.read_csv(output / "segment_kpis.csv")
    assert segments.loc[segments["units"] < 10, "low_support"].all()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_analytics.py -q`

Expected: FAIL because `news_ctr.analytics` and packaged SQL files do not exist.

- [ ] **Step 3: Add SQL that exposes denominators and stable schemas**

Each query selects named columns with deterministic ordering. For example, `kpi_summary.sql` is:

```sql
SELECT
    COUNT(DISTINCT impression_id) AS impressions,
    COUNT(*) AS candidates,
    SUM(label) AS clicks,
    SUM(label)::DOUBLE / COUNT(*) AS ctr,
    COUNT(DISTINCT user_id) AS active_users,
    AVG(candidate_count) AS mean_candidates
FROM candidates;
```

`segment_kpis.sql` uses `UNION ALL` for device, position, candidate-count, history-length, and freshness buckets, returning `segment`, `value`, `units`, `candidates`, `clicks`, and `ctr`. `user_cohorts.sql` computes first-observed week and active-week offset with window/date functions; it returns an empty typed table when fewer than two distinct weeks exist.

- [ ] **Step 4: Implement analytics orchestration with atomic publication**

```python
@dataclass(frozen=True)
class AnalyticsConfig:
    data: Path
    output: Path
    split: str = "validation"
    min_cell_count: int = 5


def run_analytics(config: AnalyticsConfig) -> Path:
    _validate_analytics_config(config)
    bundle = load_bundle(config.data, split=config.split)
    candidates = _prepare_analytics_frame(bundle)
    with _atomic_output(config.output) as staging:
        _execute_sql_outputs(candidates, staging, config)
        _write_analytics_report(staging, bundle, config)
    return config.output
```

Use `importlib.resources.files("news_ctr").joinpath("sql", name)` so installed wheels execute the same reviewed SQL. Join article metadata and history length before registering the DataFrame as DuckDB table `candidates`.

- [ ] **Step 5: Test SQL, wheel resources, and failure cleanup**

Run: `.venv/bin/python -m pytest tests/test_analytics.py -q && .venv/bin/python -m build`

Expected: PASS; the wheel lists all five `.sql` files; an injected SQL failure leaves no output directory.

- [ ] **Step 6: Commit**

```bash
git add src/news_ctr/sql src/news_ctr/analytics.py tests/test_analytics.py pyproject.toml
git commit -m "feat: add reproducible SQL product analytics"
```

---

### Task 3: Experiment design and deterministic randomized fixture

**Files:**
- Create: `src/news_ctr/experiment_data.py`
- Create: `src/news_ctr/experiments.py`
- Create: `configs/experiment-v3.json`
- Create: `tests/test_experiment_data.py`
- Create: `tests/test_experiments.py`

**Interfaces:**
- Produces: `ExperimentDesign`, `required_sample_size(design: ExperimentDesign) -> int`, `write_synthetic_experiment(path: Path, *, seed: int, users: int, corrupt_allocation: bool = False) -> Path`, `MetricSpec`, and `ExperimentConfig.from_json(path: Path)`.

- [ ] **Step 1: Write failing power and fixture tests**

```python
def test_required_sample_size_is_monotonic() -> None:
    easy = ExperimentDesign(0.12, 0.02, alpha=0.05, power=0.80)
    hard = ExperimentDesign(0.12, 0.01, alpha=0.05, power=0.80)
    assert required_sample_size(hard) > required_sample_size(easy) > 0


def test_synthetic_experiment_is_deterministic_and_user_randomized(tmp_path: Path) -> None:
    first = write_synthetic_experiment(tmp_path / "a.parquet", seed=42, users=2_000)
    second = write_synthetic_experiment(tmp_path / "b.parquet", seed=42, users=2_000)
    pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))
    assert pd.read_parquet(first)["user_id"].is_unique
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_experiment_data.py tests/test_experiments.py -q`

Expected: FAIL because experiment modules do not exist.

- [ ] **Step 3: Implement validated sample-size calculation**

```python
@dataclass(frozen=True)
class ExperimentDesign:
    baseline_rate: float
    absolute_mde: float
    alpha: float = 0.05
    power: float = 0.80


def required_sample_size(design: ExperimentDesign) -> int:
    _validate_design(design)
    p1 = design.baseline_rate
    p2 = p1 + design.absolute_mde
    pooled = (p1 + p2) / 2
    numerator = (
        norm.ppf(1 - design.alpha / 2) * sqrt(2 * pooled * (1 - pooled))
        + norm.ppf(design.power) * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return ceil(numerator / design.absolute_mde**2)
```

Validate `0 < baseline_rate < 1`, `0 < baseline_rate + absolute_mde < 1`, `0 < alpha < 1`, and `0 < power < 1`.

- [ ] **Step 4: Implement the randomized teaching fixture and config**

Generate exactly one row per user with columns `user_id`, `variant`, `pre_ctr`, `click`, `dwell_seconds`, `latency_ms`, `device_type`, and `history_segment`. Assignment uses a seeded 50/50 Bernoulli draw. Outcomes share a latent pre-period signal; treatment adds a declared positive click effect, preserves dwell time within its margin, and increases latency by less than its guardrail margin. Write adjacent JSON metadata containing `evidence_tier`, seed, unit count, allocation, and declared data-generating effects.

The committed config declares the control/treatment labels, expected allocation, primary binary metric, continuous guardrails, CUPED covariate, segment columns, alpha, practical threshold, non-inferiority margins, and minimum segment units.

- [ ] **Step 5: Run focused tests and validate deterministic files**

Run: `.venv/bin/python -m pytest tests/test_experiment_data.py tests/test_experiments.py -q`

Expected: PASS; identical seeds produce identical Parquet content and metadata.

- [ ] **Step 6: Commit**

```bash
git add src/news_ctr/experiment_data.py src/news_ctr/experiments.py configs/experiment-v3.json tests/test_experiment_data.py tests/test_experiments.py
git commit -m "feat: add experiment design and randomized fixture"
```

---

### Task 4: Experiment validation, SRM, and ITT effects

**Files:**
- Modify: `src/news_ctr/experiments.py`
- Modify: `tests/test_experiments.py`

**Interfaces:**
- Produces: `validate_experiment(frame: pd.DataFrame, config: ExperimentConfig) -> None`, `sample_ratio_mismatch(frame: pd.DataFrame, config: ExperimentConfig) -> SRMResult`, `estimate_metric_itt(frame: pd.DataFrame, metric: MetricSpec, config: ExperimentConfig) -> EffectEstimate`, `estimate_itt(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame`, and `analyze_experiment_frame(frame: pd.DataFrame, config: ExperimentConfig) -> ExperimentAnalysis`.

- [ ] **Step 1: Add failing validation and SRM tests**

```python
def test_duplicate_randomization_units_are_rejected(frame, config) -> None:
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate user_id"):
        validate_experiment(duplicated, config)


def test_srm_detects_corrupted_assignment(tmp_path: Path, config) -> None:
    path = write_synthetic_experiment(
        tmp_path / "bad.parquet", seed=42, users=10_000, corrupt_allocation=True
    )
    result = sample_ratio_mismatch(pd.read_parquet(path), config)
    assert result.p_value < config.alpha
    assert result.passed is False
```

- [ ] **Step 2: Add failing hand-calculated ITT tests**

```python
def test_binary_itt_matches_difference_in_means() -> None:
    frame = pd.DataFrame(
        {"variant": ["control"] * 4 + ["treatment"] * 4, "click": [0, 0, 1, 0, 0, 1, 1, 1]}
    )
    config = minimal_experiment_config(primary=MetricSpec("click", "binary", "increase"))
    row = estimate_metric_itt(frame, metric=config.primary, config=config)
    assert row.effect == pytest.approx(0.50)
    assert row.control_mean == pytest.approx(0.25)
    assert row.treatment_mean == pytest.approx(0.75)
```

- [ ] **Step 3: Run focused tests and verify red state**

Run: `.venv/bin/python -m pytest tests/test_experiments.py -q`

Expected: FAIL on the new validation, SRM, and estimation interfaces.

- [ ] **Step 4: Implement strict validation and SRM**

Require all configured columns, exactly two configured arms, unique and non-null units, finite numeric metrics, binary values in `{0, 1}`, and positive arm counts. Compute Pearson chi-square against configured allocation and return observed counts, expected counts, statistic, p-value, and `passed = p_value >= alpha`.

- [ ] **Step 5: Implement binary and continuous ITT estimates**

Use difference in arm means, Welch standard error, normal confidence intervals, two-sided p-values, and relative effect only when the control mean is nonzero. Define immutable `EffectEstimate` fields `metric`, `kind`, `control_units`, `treatment_units`, `control_mean`, `treatment_mean`, `effect`, `relative_effect`, `standard_error`, `ci_lower`, `ci_upper`, and `p_value`. `ExperimentAnalysis` contains `srm`, raw effect rows, optional CUPED effect rows, and segment effect rows.

- [ ] **Step 6: Run focused and regression tests**

Run: `.venv/bin/python -m pytest tests/test_experiments.py tests/test_evaluation.py -q`

Expected: PASS with no ranking-evaluation regressions.

- [ ] **Step 7: Commit**

```bash
git add src/news_ctr/experiments.py tests/test_experiments.py
git commit -m "feat: validate experiments and estimate intent to treat"
```

---

### Task 5: CUPED and heterogeneous effects

**Files:**
- Modify: `src/news_ctr/experiments.py`
- Modify: `tests/test_experiments.py`

**Interfaces:**
- Produces: `estimate_cuped(frame: pd.DataFrame, metric: MetricSpec, covariate: str, config: ExperimentConfig) -> EffectEstimate` and `estimate_segment_effects(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame`.

- [ ] **Step 1: Write failing CUPED tests**

```python
def test_cuped_reduces_standard_error_for_correlated_pre_metric(synthetic_rct, config) -> None:
    raw = estimate_metric_itt(synthetic_rct, config.primary, config)
    adjusted = estimate_cuped(synthetic_rct, config.primary, config.cuped_covariate, config)
    assert adjusted.standard_error < raw.standard_error
    assert adjusted.variance_reduction > 0


def test_post_treatment_covariate_is_rejected(config) -> None:
    bad = replace(config, cuped_covariate="latency_ms")
    with pytest.raises(ValueError, match="pre-treatment"):
        validate_experiment_config(bad)
```

- [ ] **Step 2: Write failing segment-support tests**

```python
def test_segment_effects_are_labeled_exploratory_and_low_support(frame, config) -> None:
    result = estimate_segment_effects(frame, config)
    assert set(result["analysis_role"]) == {"exploratory"}
    assert result.loc[result["units"] < config.min_segment_units, "low_support"].all()
```

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_experiments.py -q`

Expected: FAIL because CUPED and segment estimators are missing.

- [ ] **Step 4: Implement CUPED without treatment leakage**

Estimate `theta = cov(outcome, covariate) / var(covariate)` on the pooled sample, then analyze `outcome - theta * (covariate - mean(covariate))` with the same ITT estimator. Record theta, raw and adjusted standard errors, and `1 - adjusted_variance / raw_variance`. If covariate variance is zero, return theta and variance reduction as zero with the unchanged estimate.

- [ ] **Step 5: Implement predeclared segment effects**

Estimate the primary effect for configured segment values, retain both arm counts, mark support, and label undeclared-confirmatory segments as exploratory. Apply Holm-adjusted alpha across confirmatory segments; do not hide estimates for low-support cells.

- [ ] **Step 6: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_experiments.py -q`

Expected: PASS for fixed correlated, zero-variance, post-treatment, segment, and multiplicity fixtures.

- [ ] **Step 7: Commit**

```bash
git add src/news_ctr/experiments.py tests/test_experiments.py
git commit -m "feat: add CUPED and heterogeneous treatment effects"
```

---

### Task 6: Decision policy, reports, and deterministic charts

**Files:**
- Create: `src/news_ctr/decisioning.py`
- Create: `src/news_ctr/visualization.py`
- Create: `tests/test_decisioning.py`
- Create: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `ExperimentAnalysis`, `ExperimentConfig`, and evidence metadata.
- Produces: `DecisionStatus`, `evaluate_decision(*, srm_passed: bool, primary_lower: float, primary_upper: float, practical_threshold: float, guardrail_states: Sequence[str]) -> DecisionStatus`, `make_decision(analysis: ExperimentAnalysis, config: ExperimentConfig) -> DecisionResult`, `render_decision_report(result: DecisionResult, config: ExperimentConfig, evidence_tier: EvidenceTier) -> str`, and `write_effect_plot(effects: pd.DataFrame, path: Path) -> Path`.

- [ ] **Step 1: Write failing tests for all decision states**

```python
@pytest.mark.parametrize(
    ("srm_passed", "primary_interval", "guardrail_state", "expected"),
    [
        (False, (0.01, 0.03), "pass", DecisionStatus.INVALID_EXPERIMENT),
        (True, (0.01, 0.03), "pass", DecisionStatus.LAUNCH),
        (True, (-0.01, 0.02), "pass", DecisionStatus.CONTINUE_EXPERIMENT),
        (True, (-0.03, -0.01), "pass", DecisionStatus.DO_NOT_LAUNCH),
        (True, (0.01, 0.03), "fail", DecisionStatus.DO_NOT_LAUNCH),
    ],
)
def test_decision_policy(srm_passed, primary_interval, guardrail_state, expected):
    assert (
        evaluate_decision(
            srm_passed=srm_passed,
            primary_lower=primary_interval[0],
            primary_upper=primary_interval[1],
            practical_threshold=0.005,
            guardrail_states=[guardrail_state],
        )
        is expected
    )
```

- [ ] **Step 2: Write failing report and chart tests**

```python
def test_decision_report_names_evidence_and_rule(result, config) -> None:
    report = render_decision_report(result, config, evidence_tier=EvidenceTier.SYNTHETIC_RCT)
    assert "Synthetic randomized teaching evidence" in report
    assert "not production lift" in report.lower()
    assert "Guardrails" in report


def test_effect_plot_is_reproducible(tmp_path: Path, effects) -> None:
    first = write_effect_plot(effects, tmp_path / "a.png")
    second = write_effect_plot(effects, tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()
```

- [ ] **Step 3: Run focused tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_decisioning.py tests/test_visualization.py -q`

Expected: FAIL because decision and visualization modules do not exist.

- [ ] **Step 4: Implement transparent decision states**

Primary launch requires its lower interval to exceed the practical threshold. A direction-`increase` guardrail passes when its lower interval is above the negative non-inferiority margin and fails only when its upper interval is below that margin; a direction-`decrease` guardrail mirrors those inequalities. Intervals crossing a guardrail margin produce `continue_experiment`. SRM failure always produces `invalid_experiment` before effect interpretation.

- [ ] **Step 5: Implement Markdown report and deterministic forest plot**

The report lists evidence tier, validity, sample sizes, raw and CUPED primary estimates, guardrail intervals, segments, decision rule, recommendation, caveats, and exact reproduction commands. Matplotlib uses the `Agg` backend, fixed figure dimensions, fixed fonts/colors, sorted rows, and PNG metadata without timestamps.

- [ ] **Step 6: Run focused tests and inspect the rendered image**

Run: `.venv/bin/python -m pytest tests/test_decisioning.py tests/test_visualization.py -q`

Expected: PASS and the PNG is byte-identical across two writes.

- [ ] **Step 7: Commit**

```bash
git add src/news_ctr/decisioning.py src/news_ctr/visualization.py tests/test_decisioning.py tests/test_visualization.py
git commit -m "feat: turn experiment evidence into launch decisions"
```

---

### Task 7: CLI orchestration and atomic experiment publication

**Files:**
- Modify: `src/news_ctr/experiments.py`
- Modify: `src/news_ctr/cli.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_experiment_workflow.py`

**Interfaces:**
- Produces CLI commands `analyze`, `experiment-design`, `make-experiment`, and `experiment`, plus `run_experiment(input_path: Path, output: Path, config_path: Path) -> Path`.

- [ ] **Step 1: Write failing parser and end-to-end CLI tests**

```python
def test_decision_science_cli_end_to_end(tmp_path: Path) -> None:
    dataset = tmp_path / "ranking"
    experiment = tmp_path / "experiment.parquet"
    assert main(["make-synthetic", "--output", str(dataset), "--seed", "42"]) == 0
    assert main(["analyze", "--data", str(dataset), "--output", str(tmp_path / "analytics")]) == 0
    assert (
        main(["make-experiment", "--output", str(experiment), "--seed", "42", "--users", "20000"])
        == 0
    )
    assert (
        main(
            [
                "experiment",
                "--input",
                str(experiment),
                "--output",
                str(tmp_path / "study"),
                "--config",
                "configs/experiment-v3.json",
            ]
        )
        == 0
    )
    assert (tmp_path / "study" / "decision_report.md").is_file()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_experiment_workflow.py -q`

Expected: FAIL because the new subcommands are unregistered.

- [ ] **Step 3: Implement thin CLI adapters**

Register exact arguments from the design. `experiment-design` accepts `--baseline-rate`, `--relative-mde`, `--alpha`, `--power`, and optional `--daily-units`; it prints sorted JSON. Each handler delegates to its domain module and returns `0`; expected contract errors are converted to exit code `2` by existing `main` behavior.

- [ ] **Step 4: Implement atomic experiment output**

`run_experiment` loads config and Parquet, validates before interpretation, writes all CSV/JSON/report/chart files to a sibling temporary directory, and uses `os.replace` only after every file succeeds. It refuses existing output directories and cleans staging after injected failures.

- [ ] **Step 5: Run CLI, atomicity, and full regression tests**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_experiment_workflow.py -q && .venv/bin/python -m pytest -q`

Expected: PASS; all original commands retain their interfaces.

- [ ] **Step 6: Commit**

```bash
git add src/news_ctr/cli.py src/news_ctr/experiments.py tests/test_cli.py tests/test_experiment_workflow.py
git commit -m "feat: add decision science command workflows"
```

---

### Task 8: Recruiter evidence, documentation, and CI release gate

**Files:**
- Modify: `README.md`
- Modify: `Makefile`
- Modify: `.github/workflows/publish.yml`
- Modify: `docs/experiment-protocol.md`
- Create: `docs/launch-decision.md`
- Create: `artifacts/portfolio-v3/*`
- Modify: `tests/test_experiment_workflow.py`

**Interfaces:**
- Consumes all public CLIs and deterministic fixtures.
- Produces committed aggregate V3 evidence and a recruiter-facing three-minute tour.

- [ ] **Step 1: Add failing recruiter-evidence assertions**

```python
def test_committed_v3_evidence_is_truthful_and_reproducible() -> None:
    report = Path("artifacts/portfolio-v3/decision_report.md").read_text(encoding="utf-8")
    assert "synthetic-rct" in report
    assert "not production lift" in report.lower()
    assert "Decision" in report
    assert Path("artifacts/portfolio-v3/effects.png").is_file()
```

- [ ] **Step 2: Run the assertion and verify failure**

Run: `.venv/bin/python -m pytest tests/test_experiment_workflow.py::test_committed_v3_evidence_is_truthful_and_reproducible -q`

Expected: FAIL because V3 artifacts have not been generated.

- [ ] **Step 3: Add Make targets and generate clean evidence**

Add `analytics`, `experiment`, and `portfolio-v3` targets. Generate a fresh synthetic ranking bundle and 20,000-user randomized fixture, run analytics and experiment workflows, and copy only aggregate reports, CSV/JSON summaries, and charts into `artifacts/portfolio-v3/`. Do not commit generated Parquet inputs.

- [ ] **Step 4: Update recruiter-facing documentation**

README additions must include the decision question, SQL metric layer, power/SRM/ITT/CUPED definitions, one KPI table, the effect chart, the launch decision with its predeclared rule, the evidence boundary, a capability map tied to mathematics/economics/statistics/ML, exact commands, and truthful resume bullets. `docs/launch-decision.md` is a concise product memo; `docs/experiment-protocol.md` separates randomized, observational, and licensed evidence.

- [ ] **Step 5: Extend CI with decision-extra**

Install `.[dev,decision]`, run focused experiment/analytics tests, generate a small synthetic dataset and randomized experiment, run both CLIs, assert all declared output files, and grep the report for `synthetic-rct` and `not production lift`. Keep Python 3.10–3.12 core and `ranking-extra` unchanged.

- [ ] **Step 6: Run the full release gate**

Run:

```bash
.venv/bin/python -m pytest --cov=news_ctr --cov-report=term-missing
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m build
git diff --check
```

Expected: all tests pass, package coverage is at least 90%, Ruff passes, formatting is clean, sdist and wheel build, and no whitespace errors exist.

- [ ] **Step 7: Reproduce committed evidence from a clean checkout**

Generate inputs into ignored paths, run the public Make/CLI flow, and compare every deterministic aggregate CSV, JSON, and Markdown file with `artifacts/portfolio-v3/`. Normalize only explicitly documented runtime fields. Verify rendered PNGs structurally across platforms and byte-for-byte within the same runtime because font rasterization can vary with the operating system and FreeType build.

- [ ] **Step 8: Commit**

```bash
git add README.md Makefile .github/workflows/publish.yml docs/experiment-protocol.md docs/launch-decision.md artifacts/portfolio-v3 tests/test_experiment_workflow.py
git commit -m "docs: publish decision science portfolio v3"
```

- [ ] **Step 9: Request independent code review and fix findings test-first**

Review the complete V3 diff against the design spec. Any critical or important finding receives a failing regression test before the fix. Repeat the full release gate after the final correction.

- [ ] **Step 10: Publish and verify GitHub**

Publish the exact verified tree without rewriting remote history. Fetch the remote and compare `HEAD^{tree}` with `origin/main^{tree}`. Verify the final GitHub Actions run reports success for Python 3.10, 3.11, 3.12, `ranking-extra`, and `decision-extra` before reporting completion.
