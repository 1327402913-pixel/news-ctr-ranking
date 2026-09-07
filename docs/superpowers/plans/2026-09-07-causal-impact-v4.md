# Causal Impact V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, portfolio-grade difference-in-differences case study with clustered inference, event-study diagnostics, a placebo test, business translation, candid reporting, and reproducible GitHub evidence.

**Architecture:** Keep synthetic panel generation, statistical estimation, decision/report rendering, plotting, and filesystem publication in separate modules. A thin CLI calls an atomic workflow that validates the Parquet sidecar, fits exposure-weighted market and week fixed-effect models with market-clustered covariance, writes ten declared artifacts, and refuses partial or overwritten output.

**Tech Stack:** Python 3.10+, pandas, NumPy, PyArrow, SciPy, statsmodels, Matplotlib, Pillow, pytest, Ruff, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-09-05-causal-impact-v4-design.md`

## Global Constraints

- The committed evidence tier is exactly `synthetic-quasi-experiment`.
- A common week-zero adoption with never-treated controls is supported; staggered adoption is not.
- Primary and event-study models use candidate-exposure weights and covariance clustered by `market_id`.
- The event-study reference week is `-1`; the published window is `[-12, 8]`.
- Structural validation failures publish no output; failed pre-trend or placebo diagnostics produce `invalid_design`.
- No production-lift, licensed-data, currency, revenue, or automatic real-world launch claim is allowed.
- Scientific CSV, JSON, and Markdown outputs are deterministic; PNG bytes are compared only within one runtime.
- The package keeps Python 3.10-3.12 CI support and at least 90% statement coverage.

---

## File Structure

- Create `src/news_ctr/quasi_data.py`: deterministic market-panel fixture and metadata.
- Create `src/news_ctr/causal.py`: immutable config, structural validation, fixed-effect estimators, diagnostics, and structured analysis.
- Create `src/news_ctr/causal_reporting.py`: interpretation enum, decision rule, and Markdown report.
- Create `src/news_ctr/causal_visualization.py`: deterministic event-study PNG.
- Create `src/news_ctr/causal_workflow.py`: metadata/provenance validation and atomic ten-file publication.
- Create `configs/causal-impact-v4.json`: reviewed analysis contract.
- Create `tests/test_quasi_data.py`, `tests/test_causal.py`, `tests/test_causal_reporting.py`, `tests/test_causal_visualization.py`, and `tests/test_causal_workflow.py`.
- Modify `src/news_ctr/evidence.py`, `src/news_ctr/cli.py`, `tests/test_evidence.py`, `tests/test_cli.py`, `tests/test_ci_contract.py`, `pyproject.toml`, `Makefile`, `.github/workflows/publish.yml`, `README.md`, and `docs/experiment-protocol.md`.
- Create the ten committed files under `artifacts/causal-impact-v4/` only after the implementation and tests pass.

---

### Task 1: Quasi-Experiment Evidence Tier and Deterministic Panel

**Files:**
- Modify: `src/news_ctr/evidence.py`
- Create: `src/news_ctr/quasi_data.py`
- Modify: `tests/test_evidence.py`
- Create: `tests/test_quasi_data.py`

**Interfaces:**
- Produces: `EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT`
- Produces: `write_synthetic_market_panel(path: Path, *, seed: int, markets: int, pre_weeks: int, post_weeks: int) -> Path`
- Produces sidecar: `path.with_suffix(".metadata.json")`

- [ ] **Step 1: Write failing evidence-tier tests**

Add assertions that fail until the new enum and banner exist:

```python
def test_quasi_experiment_evidence_is_explicitly_limited() -> None:
    tier = EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT
    assert tier.value == "synthetic-quasi-experiment"
    banner = evidence_banner(tier)
    assert "quasi-experimental" in banner.lower()
    assert "not production lift" in banner.lower()
```

- [ ] **Step 2: Run the evidence test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_evidence.py::test_quasi_experiment_evidence_is_explicitly_limited -q`

Expected: FAIL because `SYNTHETIC_QUASI_EXPERIMENT` does not exist.

- [ ] **Step 3: Add the tier and mandatory banner**

Add the enum value and this banner text:

```python
EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT: (
    "Synthetic quasi-experimental teaching evidence. It validates the "
    "identification, diagnostic, and reporting workflow; it is not production lift."
)
```

Extend `classify_dataset_evidence()` to recognize the exact tier value without
changing the existing observational, randomized, or licensed mappings.

- [ ] **Step 4: Run the evidence tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing deterministic-generator tests**

Create tests that call the planned API twice and assert:

```python
first = write_synthetic_market_panel(
    tmp_path / "a.parquet", seed=42, markets=60, pre_weeks=20, post_weeks=12
)
second = write_synthetic_market_panel(
    tmp_path / "b.parquet", seed=42, markets=60, pre_weeks=20, post_weeks=12
)
pd.testing.assert_frame_equal(pd.read_parquet(first), pd.read_parquet(second))
frame = pd.read_parquet(first)
assert frame.columns.tolist() == [
    "market_id", "relative_week", "treated_market", "policy_active",
    "candidate_exposures", "clicks", "ctr", "market_size_index",
]
assert len(frame) == 60 * 32
assert frame[["market_id", "relative_week"]].duplicated().sum() == 0
assert frame.groupby("market_id")["treated_market"].nunique().eq(1).all()
assert (frame["policy_active"] == (
    frame["treated_market"] * (frame["relative_week"] >= 0)
)).all()
```

Also assert the two metadata JSON documents are identical, the tier is
`synthetic-quasi-experiment`, `known_effects["ctr_absolute"] == 0.006`, the row count
is 1,920, the SHA-256 matches the adjacent Parquet file, invalid dimensions raise
`ValueError`, and either an existing Parquet or sidecar prevents overwrite.

- [ ] **Step 6: Run the generator tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_quasi_data.py -q`

Expected: collection FAIL because `news_ctr.quasi_data` does not exist.

- [ ] **Step 7: Implement the deterministic panel generator**

Implement argument validation and a stable row ordering. Use `np.random.default_rng(seed)`,
market-level size and fixed effects, a common seasonal term, a common linear trend,
and this outcome contract:

```python
policy_active = treated_market * (relative_week >= 0)
click_probability = np.clip(
    0.12
    + market_effect
    + 0.006 * np.sin(2 * np.pi * relative_week / 13)
    + 0.00015 * relative_week
    + 0.006 * policy_active,
    0.02,
    0.40,
)
clicks = rng.binomial(candidate_exposures, click_probability)
ctr = clicks / candidate_exposures
```

Assign exactly half the markets to treatment using their observed
`market_size_index` ranking so that levels differ but untreated trends do not.
Write Parquet first, hash its bytes, then write sorted/indented JSON plus a trailing
newline. If metadata writing fails, remove the newly written Parquet before raising.

- [ ] **Step 8: Run generator and evidence tests**

Run: `.venv/bin/python -m pytest tests/test_evidence.py tests/test_quasi_data.py -q`

Expected: PASS.

- [ ] **Step 9: Commit the fixture layer**

```bash
git add src/news_ctr/evidence.py src/news_ctr/quasi_data.py \
  tests/test_evidence.py tests/test_quasi_data.py
git commit -m "feat: add deterministic quasi-experiment panel"
```

---

### Task 2: Immutable Causal Configuration and Structural Validation

**Files:**
- Create: `src/news_ctr/causal.py`
- Create: `tests/test_causal.py`
- Create: `configs/causal-impact-v4.json`

**Interfaces:**
- Produces: `CausalConfig.from_json(path: str | Path) -> CausalConfig`
- Produces: `validate_causal_config(config: CausalConfig) -> None`
- Produces: `validate_market_panel(frame: pd.DataFrame, config: CausalConfig) -> dict[str, object]`
- Consumes the eight-column frame from `write_synthetic_market_panel()`.

- [ ] **Step 1: Write failing config-loading tests**

Define the intended immutable dataclass fields in tests:

```python
config = CausalConfig.from_json(Path("configs/causal-impact-v4.json"))
assert config.unit_column == "market_id"
assert config.time_column == "relative_week"
assert config.group_column == "treated_market"
assert config.active_column == "policy_active"
assert config.exposure_column == "candidate_exposures"
assert config.click_column == "clicks"
assert config.outcome_column == "ctr"
assert config.balance_columns == ("market_size_index", "candidate_exposures", "ctr")
assert config.rollout_week == 0
assert config.reference_week == -1
assert config.event_window == (-12, 8)
assert config.placebo_week == -8
assert config.alpha == 0.05
assert config.practical_threshold == 0.002
assert config.business_exposure_scale == 1_000_000
```

Parameterize invalid payloads for identical rollout/reference week, a window that
does not contain the reference or rollout, non-positive alpha/threshold/scale,
placebo week outside the pre-period, cluster minima below two, and negative
missingness.

- [ ] **Step 2: Run config tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k config -q`

Expected: FAIL because `CausalConfig` does not exist.

- [ ] **Step 3: Implement the immutable config and reviewed JSON**

Use a frozen dataclass with explicit scalar and tuple fields. Parse JSON into exact
types and reject extra keys by comparing the payload key set with the declared schema.
Write `configs/causal-impact-v4.json` with the values fixed in the specification.

- [ ] **Step 4: Run config tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k config -q`

Expected: PASS.

- [ ] **Step 5: Write failing structural-validation tests**

Use the deterministic panel and assert the successful result contains:

```python
diagnostics = validate_market_panel(frame, config)
assert diagnostics["rows"] == 1_920
assert diagnostics["markets"] == 60
assert diagnostics["treated_markets"] == 30
assert diagnostics["control_markets"] == 30
assert diagnostics["missing_market_week_share"] == 0.0
assert diagnostics["week_min"] == -20
assert diagnostics["week_max"] == 11
assert diagnostics["integrity_passed"] is True
```

Add one mutation test for each failure class in the spec: missing columns, empty
frame, duplicate key, null ID/group, changing treatment membership, incorrect active
flag, non-integer week, non-positive exposure, invalid clicks, inconsistent CTR,
non-finite values, insufficient arm clusters, insufficient pre/post weeks, and a
missing market-week row above the zero threshold.

- [ ] **Step 6: Run validation tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k validation -q`

Expected: FAIL because `validate_market_panel` does not exist.

- [ ] **Step 7: Implement validation and diagnostic summary**

Validate schema before metadata or estimation, compute the complete cartesian
market-week expectation, and compare stored CTR with `np.isclose(..., atol=1e-12,
rtol=0)`. Return only JSON-serializable Python scalars and sorted arm counts.

- [ ] **Step 8: Run all config/validation tests**

Run: `.venv/bin/python -m pytest tests/test_causal.py -q`

Expected: PASS for the implemented test subset.

- [ ] **Step 9: Commit config and validation**

```bash
git add src/news_ctr/causal.py tests/test_causal.py configs/causal-impact-v4.json
git commit -m "feat: validate causal impact analysis contracts"
```

---

### Task 3: Exposure-Weighted DiD With Market-Clustered Inference

**Files:**
- Modify: `src/news_ctr/causal.py`
- Modify: `tests/test_causal.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `fit_difference_in_differences(frame: pd.DataFrame, config: CausalConfig) -> pd.DataFrame`
- Internal: `_fit_fixed_effect_model(frame, config, treatment_columns) -> RegressionResults`
- Adds optional dependency group: `causal = ["matplotlib>=3.9", "pillow>=10", "scipy>=1.14", "statsmodels>=0.14"]`

- [ ] **Step 1: Write the failing primary-estimator test**

```python
estimate = fit_difference_in_differences(frame, config)
assert estimate["term"].tolist() == ["policy_active"]
row = estimate.iloc[0]
assert row["ci_lower"] < 0.006 < row["ci_upper"]
assert row["effect"] > config.practical_threshold
assert row["standard_error"] > 0
assert 0 <= row["p_value"] <= 1
assert row["clusters"] == 60
assert row["observations"] == len(frame)
assert row["total_exposure"] == frame["candidate_exposures"].sum()
assert row["covariance"] == "cluster:market_id"
```

Add a deliberately null-effect fixture whose interval contains zero. Add a small
constructed panel where exposure weighting gives a known coefficient different from
the unweighted coefficient; assert the function returns the weighted value and names
the covariance `cluster:market_id`.

- [ ] **Step 2: Run the estimator test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k difference_in_differences -q`

Expected: FAIL because the estimator is missing.

- [ ] **Step 3: Add the causal dependency and install it**

Add the `causal` optional group without moving causal-only libraries into core
dependencies. Run:

`.venv/bin/python -m pip install -e ".[dev,causal]"`

Expected: installation succeeds and `python -c "import statsmodels"` exits zero.

- [ ] **Step 4: Implement a stable fixed-effect design matrix**

Build a float matrix with a constant, sorted `drop_first=True` market dummies, sorted
week dummies, and the requested treatment columns:

```python
model = sm.WLS(
    frame[config.outcome_column].astype(float),
    design.astype(float),
    weights=frame[config.exposure_column].astype(float),
)
result = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": frame[config.unit_column], "use_correction": True},
)
```

Reject a rank-deficient or non-finite fit with a concise `ValueError`. Format the one
row output through fixed column order and native numeric types; do not round in memory.

- [ ] **Step 5: Run estimator tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k difference_in_differences -q`

Expected: PASS.

- [ ] **Step 6: Run the existing suite without causal extras**

Run: `.venv/bin/python -m pytest -q`

Expected: all existing and new non-optional tests pass; causal tests use
`pytest.importorskip("statsmodels")` when the extra is unavailable.

- [ ] **Step 7: Commit clustered DiD**

```bash
git add src/news_ctr/causal.py tests/test_causal.py pyproject.toml
git commit -m "feat: estimate clustered difference in differences"
```

---

### Task 4: Event Study, Pre-Trend Test, Placebo, Balance, and Business Impact

**Files:**
- Modify: `src/news_ctr/causal.py`
- Modify: `tests/test_causal.py`

**Interfaces:**
- Produces: `estimate_event_study(frame, config) -> tuple[pd.DataFrame, dict[str, object]]`
- Produces: `estimate_placebo(frame, config) -> tuple[pd.DataFrame, dict[str, object]]`
- Produces: `summarize_preperiod_balance(frame, config) -> pd.DataFrame`
- Produces: `translate_business_impact(did_estimate, config) -> pd.DataFrame`
- Produces dataclass: `CausalAnalysis(did_estimate, event_study, placebo_estimate, balance, business_impact, diagnostics)`
- Produces: `analyze_causal_frame(frame, config) -> CausalAnalysis`

- [ ] **Step 1: Write failing event-study and joint-test tests**

Assert exact ordered weeks `[-12, ..., -2, 0, ..., 8]`, omission of `-1`, required
columns `relative_week/effect/standard_error/p_value/ci_lower/ci_upper`, and finite
values. Assert:

```python
event, parallel = estimate_event_study(frame, config)
assert parallel["lead_terms"] == 11
assert 0 <= parallel["p_value"] <= 1
assert parallel["passed"] is True
assert event.loc[event["relative_week"] >= 0, "effect"].mean() > 0.004
```

Create a copied frame with `ctr` and `clicks` modified to add a +0.004 treated-market
pre-trend while preserving the CTR identity; assert the joint test fails.

- [ ] **Step 2: Run event-study tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k "event_study or parallel" -q`

Expected: FAIL because the event-study functions are missing.

- [ ] **Step 3: Implement event-study estimation**

Filter to the configured window. Create one treated-week interaction per included week
except reference `-1`, use the shared fixed-effect helper, and call a joint Wald test
over all negative-week coefficients. Store `statistic`, `degrees_of_freedom`,
`p_value`, `lead_terms`, `alpha`, and `passed` using native scalars.

- [ ] **Step 4: Run event-study tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing placebo, balance, and business tests**

Assert the placebo uses only `relative_week < 0`, names its term
`placebo_policy_active`, and passes only when its interval contains zero. Assert
balance output has three rows in configured order and reports finite standardized
mean differences. Assert business translation is exact:

```python
impact = translate_business_impact(did, config).iloc[0]
assert impact["scale_exposures"] == 1_000_000
assert impact["incremental_clicks"] == pytest.approx(did.iloc[0]["effect"] * 1_000_000)
assert impact["ci_lower_clicks"] == pytest.approx(did.iloc[0]["ci_lower"] * 1_000_000)
assert impact["ci_upper_clicks"] == pytest.approx(did.iloc[0]["ci_upper"] * 1_000_000)
```

- [ ] **Step 6: Run the new diagnostic tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal.py -k "placebo or balance or business" -q`

Expected: FAIL because the functions are missing.

- [ ] **Step 7: Implement placebo, balance, translation, and orchestration**

For placebo, restrict to true pre-period rows and define the fake active flag as
`treated_market * (relative_week >= placebo_week)`. For balance, aggregate each
market over true pre-period rows first, then compute the standardized mean difference
with pooled arm variance. `analyze_causal_frame()` first calls structural validation,
then returns all five frames plus nested `integrity`, `parallel_trends`, and `placebo`
diagnostics.

- [ ] **Step 8: Run all causal tests**

Run: `.venv/bin/python -m pytest tests/test_causal.py -q`

Expected: PASS.

- [ ] **Step 9: Commit complete statistical diagnostics**

```bash
git add src/news_ctr/causal.py tests/test_causal.py
git commit -m "feat: add causal design diagnostics"
```

---

### Task 5: Interpretation Policy and Recruiter-Readable Report

**Files:**
- Create: `src/news_ctr/causal_reporting.py`
- Create: `tests/test_causal_reporting.py`

**Interfaces:**
- Produces enum: `CausalDecisionStatus`
- Produces dataclass: `CausalDecision(status, diagnostics, did_estimate, placebo_estimate, business_impact)`
- Produces: `evaluate_causal_decision(analysis: CausalAnalysis, config: CausalConfig) -> CausalDecision`
- Produces: `render_causal_report(decision, analysis, config, *, evidence_tier, provenance) -> str`

- [ ] **Step 1: Write failing four-state policy tests**

Build minimal `CausalAnalysis` fixtures and parameterize these expectations:

```python
cases = [
    ({"parallel": False, "placebo": True, "lower": 0.004, "upper": 0.008}, "invalid_design"),
    ({"parallel": True, "placebo": True, "lower": 0.003, "upper": 0.008}, "supports_incremental_impact"),
    ({"parallel": True, "placebo": True, "lower": -0.008, "upper": 0.0}, "evidence_of_no_benefit"),
    ({"parallel": True, "placebo": True, "lower": -0.001, "upper": 0.006}, "inconclusive"),
]
```

Also verify a failed placebo returns `invalid_design` regardless of the primary
estimate.

- [ ] **Step 2: Run policy tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal_reporting.py -k decision -q`

Expected: FAIL because the reporting module is missing.

- [ ] **Step 3: Implement the deterministic policy**

Apply design validity first, then no-benefit, then practical-positive evidence, then
inconclusive. Reject missing or non-finite diagnostic/interval inputs rather than
silently choosing a state.

- [ ] **Step 4: Run policy tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write the failing report-content test**

Assert the report contains all of the following:

```python
required = [
    "synthetic-quasi-experiment",
    "not production lift",
    "Difference-in-Differences",
    "market fixed effects",
    "week fixed effects",
    "clustered by `market_id`",
    "Parallel trends",
    "Placebo rollout",
    "per 1,000,000 candidate exposures",
    "supports_incremental_impact",
    "unobserved time-varying confounding",
    "make causal-impact-v4",
]
assert all(item.lower() in report.lower() for item in required)
assert provenance["input_sha256"] in report
```

- [ ] **Step 6: Run report test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal_reporting.py -k report -q`

Expected: FAIL because `render_causal_report` is missing.

- [ ] **Step 7: Implement the Markdown report**

Render fixed sections: evidence banner, decision question, identification strategy,
integrity summary, primary estimate, event-study/pre-trend result, placebo result,
pre-period balance, per-million translation, interpretation, assumptions,
limitations, provenance hashes, and reproduction. Format estimates to four decimals
and click translations to whole clicks without changing stored CSV precision.

- [ ] **Step 8: Run reporting tests**

Run: `.venv/bin/python -m pytest tests/test_causal_reporting.py -q`

Expected: PASS.

- [ ] **Step 9: Commit interpretation and reporting**

```bash
git add src/news_ctr/causal_reporting.py tests/test_causal_reporting.py
git commit -m "feat: report quasi-experimental impact decisions"
```

---

### Task 6: Event-Study Visualization and Atomic Publication Workflow

**Files:**
- Create: `src/news_ctr/causal_visualization.py`
- Create: `src/news_ctr/causal_workflow.py`
- Create: `tests/test_causal_visualization.py`
- Create: `tests/test_causal_workflow.py`

**Interfaces:**
- Produces: `write_event_study_plot(event_study: pd.DataFrame, path: Path, *, confidence_level: float = 0.95) -> Path`
- Produces constant: `CAUSAL_ARTIFACTS: tuple[str, ...]`
- Produces: `run_causal_impact(input_path: Path, output: Path, config_path: Path) -> Path`

- [ ] **Step 1: Write failing visualization tests**

Use a three-row frame to assert two writes are byte-identical within the runtime,
file size exceeds 5,000 bytes, the image description equals
`Synthetic quasi-experiment event study with 95% confidence intervals`, and missing
columns, empty data, invalid confidence, or an existing destination raise
`ValueError`.

- [ ] **Step 2: Run visualization tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal_visualization.py -q`

Expected: collection FAIL because the module is missing.

- [ ] **Step 3: Implement deterministic plotting**

Use the `Agg` backend, sorted weeks, fixed 9-by-5-inch dimensions, DejaVu Sans,
fixed colors, a horizontal zero line, a vertical week-zero rollout line, capped 95%
intervals, fixed labels, white background, and metadata without timestamps. Close the
figure in all successful paths.

- [ ] **Step 4: Run visualization tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing workflow tests**

Generate a panel and run the planned workflow. Assert the output directory contains
exactly:

```python
CAUSAL_ARTIFACTS = (
    "balance.csv", "business_impact.csv", "causal_report.md",
    "config_snapshot.json", "diagnostics.json", "did_estimate.csv",
    "event_study.csv", "event_study.png", "placebo_estimate.csv",
    "run_manifest.json",
)
```

Verify config bytes, input/metadata/config SHA-256 values, the evidence tier, exact
artifact names, and report boundary. Add tests for missing input, missing sidecar,
incorrect sidecar hash/tier/rows, existing output, and injected writer failure leaving
no output or staging directory.

- [ ] **Step 6: Run workflow tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal_workflow.py -q`

Expected: collection FAIL because `causal_workflow` is missing.

- [ ] **Step 7: Implement provenance validation and atomic writer**

Load config and frame, perform structural validation before sidecar checks, require
the exact quasi-experiment tier, and verify metadata row count, market count, week
parameters, known effect, and Parquet SHA. Build a schema-versioned manifest with
analysis contract and package versions. In a `TemporaryDirectory` under
`output.parent`, write CSVs with `float_format="%.10f"`, sorted/indented JSON with a
trailing newline, config bytes unchanged, the report, and PNG; verify
`set(CAUSAL_ARTIFACTS)` before `os.replace(staging, output)`.

- [ ] **Step 8: Run workflow, visualization, and causal tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_causal.py tests/test_causal_reporting.py \
  tests/test_causal_visualization.py tests/test_causal_workflow.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit visualization and atomic workflow**

```bash
git add src/news_ctr/causal_visualization.py src/news_ctr/causal_workflow.py \
  tests/test_causal_visualization.py tests/test_causal_workflow.py
git commit -m "feat: publish causal impact evidence atomically"
```

---

### Task 7: CLI, Make Target, and Public Configuration

**Files:**
- Modify: `src/news_ctr/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `Makefile`
- Modify: `pyproject.toml`
- Use: `configs/causal-impact-v4.json`

**Interfaces:**
- CLI: `news-ctr make-quasi-experiment --output PATH [--seed 42 --markets 60 --pre-weeks 20 --post-weeks 12]`
- CLI: `news-ctr causal-impact --input PATH --output DIR --config PATH`
- Make: `make causal-impact-v4`

- [ ] **Step 1: Write failing parser and end-to-end CLI tests**

Extend the existing CLI integration test to invoke both commands, parse the generator
JSON, require `evidence_tier == "synthetic-quasi-experiment"`, run the analyzer, and
assert `causal_report.md`, `event_study.png`, and `run_manifest.json` exist. Add error
tests for invalid market/week counts and output overwrite returning exit code 2 with a
concise `error:` message.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k "quasi or causal" -q`

Expected: FAIL because the subcommands are unknown.

- [ ] **Step 3: Add lazy CLI handlers and parser arguments**

Follow the existing decision-extra pattern so core imports do not require
statsmodels/Matplotlib:

```python
def _make_quasi_experiment(args: argparse.Namespace) -> int:
    from news_ctr.quasi_data import write_synthetic_market_panel
    output = write_synthetic_market_panel(
        args.output, seed=args.seed, markets=args.markets,
        pre_weeks=args.pre_weeks, post_weeks=args.post_weeks,
    )
    print(json.dumps({
        "evidence_tier": "synthetic-quasi-experiment", "output": str(output)
    }, sort_keys=True))
    return 0

def _causal_impact(args: argparse.Namespace) -> int:
    from news_ctr.causal_workflow import run_causal_impact
    output = run_causal_impact(args.input, args.output, args.config)
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0
```

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Add Make setup and V4 targets**

Add `setup-causal` and `causal-impact-v4` to `.PHONY`. The workflow checks that its
input and output do not exist, then runs the exact two public CLI commands. Define:

```make
CAUSAL_INPUT ?= data/synthetic-market-panel-v4.parquet
CAUSAL_OUTPUT ?= artifacts/causal-impact-v4-regenerated
```

- [ ] **Step 6: Exercise the Make target in unique temporary paths**

Run with explicit paths below so committed artifacts are untouched:

```bash
verification_dir=$(mktemp -d /tmp/news-ctr-causal.XXXXXX)
make causal-impact-v4 \
  CAUSAL_INPUT="$verification_dir/panel.parquet" \
  CAUSAL_OUTPUT="$verification_dir/output"
```

Expected: ten artifacts are created and the report says `not production lift`.

- [ ] **Step 7: Run core-only import and full CLI regression tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_cli.py -q
.venv/bin/python -c "from news_ctr.cli import build_parser; build_parser()"
```

Expected: PASS.

- [ ] **Step 8: Commit public entry points**

```bash
git add src/news_ctr/cli.py tests/test_cli.py Makefile pyproject.toml \
  configs/causal-impact-v4.json
git commit -m "feat: add causal impact command line workflow"
```

---

### Task 8: Committed Evidence and Recruiter Documentation

**Files:**
- Create: `artifacts/causal-impact-v4/*` (exactly ten files)
- Modify: `README.md`
- Modify: `docs/experiment-protocol.md`
- Create: `docs/causal-impact-brief.md`
- Modify: `tests/test_causal_workflow.py`

**Interfaces:**
- Consumes: `make causal-impact-v4`
- Produces: committed deterministic V4 evidence and a two-minute interview brief.

- [ ] **Step 1: Write a failing committed-evidence contract test**

Assert the committed directory contains exactly `CAUSAL_ARTIFACTS`, the report contains
the exact evidence tier, method, identifying assumptions, decision state, and
limitation, the estimate interval covers `0.006`, diagnostics show both pre-trend and
placebo passing, and the manifest hashes the copied config.

- [ ] **Step 2: Run the committed-evidence test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_causal_workflow.py -k committed -q`

Expected: FAIL because `artifacts/causal-impact-v4` does not exist.

- [ ] **Step 3: Generate the committed evidence once**

Run:

```bash
.venv/bin/news-ctr make-quasi-experiment \
  --output data/synthetic-market-panel-v4.parquet \
  --seed 42 --markets 60 --pre-weeks 20 --post-weeks 12
.venv/bin/news-ctr causal-impact \
  --input data/synthetic-market-panel-v4.parquet \
  --output artifacts/causal-impact-v4 \
  --config configs/causal-impact-v4.json
```

Expected: exactly ten files; no row-level panel is tracked.

- [ ] **Step 4: Run the contract test and inspect every aggregate**

Run the test from Step 2, then inspect all CSV/JSON/Markdown files and the PNG. Verify
the synthetic fixture returns `supports_incremental_impact`, the interval covers the
known effect, and the report does not use production or launch language as a factual
claim.

- [ ] **Step 5: Update the recruiter-facing README**

Add a `Causal Impact V4` section before the ranking evidence with:

- the observational business question;
- the DiD point estimate and market-clustered 95% interval from committed artifacts;
- event-study image and links to report/diagnostics;
- parallel-trend and placebo results;
- incremental clicks per million exposures;
- a contrast between RCT and quasi-experimental identification;
- architecture nodes for market panel, fixed effects, event study, placebo, and report;
- a truthful resume bullet beginning `Built a market-level quasi-experimental impact workflow...`.

Do not hand-copy numbers before reading the generated artifacts.

- [ ] **Step 6: Extend the protocol and write the interview brief**

Document when DiD is appropriate, the exact estimand, the parallel-trends assumption,
why clustered standard errors are used, how to interpret pre-trend/placebo checks, and
the explicit non-goals. In `docs/causal-impact-brief.md`, answer: business context,
why not A/B, estimator, assumptions, diagnostics, result, business translation,
limitations, and next real-data step.

- [ ] **Step 7: Reproduce all ten aggregates in a separate output**

Run the generator/analyzer into a new temporary directory. Compare every non-PNG
artifact byte-for-byte and validate PNG size, mode, description, and non-empty color
content with Pillow. Also confirm a second same-runtime PNG is byte-identical.

- [ ] **Step 8: Run documentation and workflow tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_causal_workflow.py tests/test_causal_reporting.py -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Expected: PASS.

- [ ] **Step 9: Commit public V4 evidence**

```bash
git add artifacts/causal-impact-v4 README.md docs/experiment-protocol.md \
  docs/causal-impact-brief.md tests/test_causal_workflow.py
git commit -m "docs: publish causal impact portfolio evidence"
```

---

### Task 9: CI Release Gate and Final Verification

**Files:**
- Modify: `.github/workflows/publish.yml`
- Modify: `tests/test_ci_contract.py`

**Interfaces:**
- Produces GitHub job: `causal-extra`
- Consumes: both public V4 CLI commands, `CAUSAL_ARTIFACTS`, and committed V4 evidence.

- [ ] **Step 1: Write a failing CI contract test**

Assert the workflow contains `causal-extra`, installs `.[dev,causal]`, runs focused
causal tests, invokes both V4 commands, checks exact artifact names, compares
non-PNG bytes, and structurally validates PNG metadata. Require both
`synthetic-quasi-experiment` and `not production lift` report checks.

- [ ] **Step 2: Run the CI contract test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_ci_contract.py -q`

Expected: FAIL because `causal-extra` is absent.

- [ ] **Step 3: Add the Ubuntu/Python 3.12 causal job**

Add a separate job that:

1. checks out the repository;
2. installs Python 3.12 and `.[dev,causal]`;
3. runs the five causal-focused test modules;
4. generates `data/ci-market-panel.parquet`;
5. publishes `artifacts/causal-ci`;
6. asserts its filenames equal `CAUSAL_ARTIFACTS`;
7. compares CSV/JSON/Markdown bytes against `artifacts/causal-impact-v4`;
8. validates both PNGs with Pillow using the V3 cross-platform contract;
9. greps the report for the evidence tier and limitation.

- [ ] **Step 4: Run CI contract and parse workflow YAML**

Run:

```bash
.venv/bin/python -m pytest tests/test_ci_contract.py -q
ruby -e 'require "yaml"; YAML.load_file(".github/workflows/publish.yml"); puts "workflow YAML parsed"'
```

Expected: PASS and `workflow YAML parsed`.

- [ ] **Step 5: Run the complete verification gate**

Run:

```bash
.venv/bin/python -m pytest --cov=news_ctr --cov-report=term-missing --cov-fail-under=90
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m build
git diff --check
```

Expected: all tests pass, coverage is at least 90%, formatting/lint pass, source and
wheel builds succeed, and no whitespace errors are reported.

- [ ] **Step 6: Commit the release gate**

```bash
git add .github/workflows/publish.yml tests/test_ci_contract.py
git commit -m "ci: verify causal impact portfolio"
```

- [ ] **Step 7: Verify the committed branch from a clean Git archive**

Create a temporary archive from `HEAD`. Install
`.[dev,decision,ranking,causal]`, rerun the complete tests and
`make causal-impact-v4` in unique paths, and compare all ten artifacts using the
documented reproducibility contract.

- [ ] **Step 8: Request independent code review**

Have the reviewer inspect statistical correctness, data leakage/identification
claims, atomic output, CLI compatibility, tests, documentation, and CI. Resolve every
Critical or Important finding with a new failing test before changing production
code. Re-run Steps 5 and 7 after any fix, then commit the fix.

- [ ] **Step 9: Publish through a pull request**

Push `feat/causal-impact-v4`, create a PR targeting `main`, include the evidence
boundary and verification results, wait for every job to pass, merge, and verify the
post-merge `main` run. Do not delete the remote branch until the merged main commit and
README are confirmed publicly accessible.
