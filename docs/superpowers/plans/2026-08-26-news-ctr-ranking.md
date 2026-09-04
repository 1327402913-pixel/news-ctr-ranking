# News CTR Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a reproducible news click-through-rate ranking portfolio project based on the EB-NeRD data contract.

**Architecture:** Parquet inputs are normalized and audited before impression candidates are expanded. A fitted feature builder combines context, article metadata, and TF-IDF/SVD user-content similarity; pointwise logistic regression and optional LightGBM LambdaRank share group-aware evaluation and artifact reporting.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, PyArrow, optional LightGBM, pytest, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-26-news-ctr-ranking-design.md`

## Global Constraints

- Raw EB-NeRD files are never committed, repackaged, or uploaded.
- The default random seed is `42`.
- The core package must run without LightGBM; LightGBM is an optional `ranking` extra.
- Candidate negatives remain inside their source impression.
- The maximum training timestamp is strictly earlier than the minimum validation timestamp.
- Generated predictions retain `impression_id`, `article_id`, `label`, and `score`.
- Synthetic results are labeled as engineering verification, not EB-NeRD benchmark results.

---

### Task 1: Package skeleton and validated data contract

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/news_ctr/__init__.py`
- Create: `src/news_ctr/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces: `DatasetBundle`, `load_bundle(path)`, `audit_bundle(bundle)`, `expand_candidates(behaviors)`, `temporal_split(frame, valid_fraction)`, and `write_synthetic_bundle(path, seed)`.
- Consumes: EB-NeRD Parquet files or the equivalent synthetic schema.

- [ ] **Step 1: Write data-contract tests**

  Add tests that assert deterministic synthetic generation, required columns, one expanded row per candidate, labels only in `{0, 1}`, clicked candidates belonging to their original impression, actionable missing-column errors, and strict temporal separation.

- [ ] **Step 2: Run tests and confirm the package is absent**

  Run `python -m pytest tests/test_data.py -q` and expect import or symbol failures.

- [ ] **Step 3: Implement the minimal validated data layer**

  Define a frozen `DatasetBundle` dataclass containing articles, behaviors, history, and source name. Normalize supported EB-NeRD aliases, validate list columns and timestamps, expand candidates with stable zero-based positions, split on ordered unique timestamps, and generate a small dataset whose click probability depends on category affinity, freshness, and position.

- [ ] **Step 4: Run data tests**

  Run `python -m pytest tests/test_data.py -q` and expect all data tests to pass.

- [ ] **Step 5: Commit the data layer**

  Run `git add pyproject.toml .gitignore src/news_ctr tests/test_data.py && git commit -m "feat: add validated news impression data layer"`.

### Task 2: Leakage-safe feature builder

**Files:**
- Create: `src/news_ctr/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: canonical articles, history, and expanded candidate frames from `news_ctr.data`.
- Produces: `NewsFeatureBuilder.fit(articles, history, train_candidates) -> NewsFeatureBuilder`, `transform(candidates) -> pandas.DataFrame`, and `feature_names_`.

- [ ] **Step 1: Write feature tests**

  Verify the output preserves row order, contains no IDs, contains only finite numeric values, exposes stable feature names, gives higher category affinity to matching history, returns zero similarity for empty history, and calculates publication age from the impression timestamp rather than the current clock.

- [ ] **Step 2: Run feature tests and confirm failure**

  Run `python -m pytest tests/test_features.py -q` and expect `news_ctr.features` to be missing.

- [ ] **Step 3: Implement article and user representations**

  Fit a `FeatureUnion` of word and character TF-IDF vectorizers on training-visible title and subtitle text, reduce it with Truncated SVD, normalize vectors, and map user histories to mean, recent, and category-count representations. Encode bounded categorical fields with deterministic training-time maps and reserve `-1` for unknown values.

- [ ] **Step 4: Implement candidate transforms**

  Produce context, article, history, category-affinity, cosine-similarity, freshness, and text-length features. Median-fill missing numeric inputs from fitted training statistics and replace remaining non-finite values with zero.

- [ ] **Step 5: Run feature tests**

  Run `python -m pytest tests/test_features.py -q` and expect all feature tests to pass.

- [ ] **Step 6: Commit the feature builder**

  Run `git add src/news_ctr/features.py tests/test_features.py && git commit -m "feat: build leakage-safe user and article features"`.

### Task 3: Group-aware metrics and models

**Files:**
- Create: `src/news_ctr/metrics.py`
- Create: `src/news_ctr/models.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: numeric feature frames, binary labels, and impression IDs.
- Produces: `ranking_metrics(labels, scores, groups, ks=(5, 10))`, `fit_model(kind, X, y, groups, seed)`, `predict_scores(model, X)`, and `feature_importance(model, names)`.

- [ ] **Step 1: Write exact metric tests**

  Use hand-calculated two-impression examples to verify MRR, NDCG@5, group AUC, skipped AUC groups, row-order invariance within each group, and clear failures for length mismatch or non-binary labels.

- [ ] **Step 2: Run metric tests and confirm failure**

  Run `python -m pytest tests/test_metrics.py -q` and expect the metric module to be missing.

- [ ] **Step 3: Implement ranking metrics**

  Group rows by impression, use stable descending score order, calculate reciprocal rank from the first relevant candidate, calculate DCG/IDCG for each requested cutoff, and average valid per-group AUC values.

- [ ] **Step 4: Write model tests**

  Verify logistic fitting returns finite probabilities in `[0, 1]`, fixed seeds are deterministic, feature importance includes every column, and requesting LightGBM without the extra raises an installation instruction.

- [ ] **Step 5: Implement model adapters**

  Implement logistic regression as a median-imputer/standard-scaler/classifier pipeline. Lazily import `LGBMRanker`, stably sort by impression before fitting with explicit group sizes, and expose coefficients or gain importance through one normalized table.

- [ ] **Step 6: Run model and metric tests**

  Run `python -m pytest tests/test_metrics.py tests/test_models.py -q` and expect all tests to pass.

- [ ] **Step 7: Commit metrics and models**

  Run `git add src/news_ctr/metrics.py src/news_ctr/models.py tests/test_metrics.py tests/test_models.py && git commit -m "feat: add ranking metrics and model baselines"`.

### Task 4: Training workflow, artifacts, and CLI

**Files:**
- Create: `src/news_ctr/reporting.py`
- Create: `src/news_ctr/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: a bundle path, output path, model kind, validation fraction, SVD dimension, and seed.
- Produces: console commands `news-ctr audit`, `news-ctr make-synthetic`, and `news-ctr train`; writes `metrics.json`, `predictions.parquet`, `feature_importance.csv`, `run_config.json`, and `model_card.md`.

- [ ] **Step 1: Write an end-to-end CLI smoke test**

  Generate a temporary bundle, audit it, train logistic regression, and assert every declared artifact exists, the report says `synthetic`, prediction columns match the spec, metrics are finite, and a second run with the same seed reproduces scores.

- [ ] **Step 2: Run the CLI test and confirm failure**

  Run `python -m pytest tests/test_cli.py -q` and expect the CLI module or commands to be missing.

- [ ] **Step 3: Implement reporting**

  Serialize JSON with sorted keys and native scalar conversion, write predictions as Parquet, sort importance descending, compute a SHA-256 fingerprint from input file metadata, and render a model card that distinguishes synthetic from EB-NeRD results.

- [ ] **Step 4: Implement commands and training orchestration**

  Use `argparse` subcommands. For `train`, load and audit the bundle, expand candidates, use an explicit validation bundle when present or a temporal split otherwise, fit the feature builder on training data only, train the selected model, calculate ranking metrics, and write artifacts. Convert data-contract exceptions to concise non-zero CLI exits.

- [ ] **Step 5: Run all tests**

  Run `python -m pytest -q` and expect the complete suite to pass.

- [ ] **Step 6: Commit the executable workflow**

  Run `git add src/news_ctr/reporting.py src/news_ctr/cli.py tests/test_cli.py && git commit -m "feat: add reproducible training CLI and artifacts"`.

### Task 5: Portfolio documentation, automation, and release verification

**Files:**
- Create: `README.md`
- Create: `docs/data.md`
- Create: `docs/experiment-protocol.md`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `Makefile`
- Create: `.github/workflows/ci.yml`
- Create: `artifacts/synthetic-smoke/metrics.json`
- Create: `artifacts/synthetic-smoke/model_card.md`
- Create: `artifacts/synthetic-smoke/feature_importance.csv`

**Interfaces:**
- Consumes: the tested CLI and synthetic generator.
- Produces: a recruiter-readable repository, CI workflow, and reproducible smoke-run evidence.

- [ ] **Step 1: Run the synthetic experiment**

  Run `news-ctr make-synthetic --output data/synthetic --seed 42` and `news-ctr train --data data/synthetic --output artifacts/synthetic-smoke --model logistic --seed 42`. Remove generated predictions from the tracked artifact directory because row-level examples are reproducible and unnecessarily large.

- [ ] **Step 2: Write project documentation**

  Document the business framing, impression-group data model, architecture, quick start, EB-NeRD license boundary, experiment protocol, leakage safeguards, metric interpretation, synthetic smoke results, limitations, repository map, and concrete interview discussion prompts. Do not claim synthetic metrics measure real recommendation quality.

- [ ] **Step 3: Add repository metadata and CI**

  Add an MIT license for original code, citation metadata referencing EB-NeRD, Make targets for setup/test/lint/smoke, and GitHub Actions that install the development extra and run Ruff plus pytest on Python 3.10, 3.11, and 3.12.

- [ ] **Step 4: Verify formatting, tests, packaging, and repository hygiene**

  Run `ruff check .`, `python -m pytest -q`, `python -m build`, `news-ctr --help`, `git status --short`, and a tracked-file scan that rejects data files, secrets, `.venv`, caches, and prediction outputs.

- [ ] **Step 5: Commit the release candidate**

  Run `git add README.md docs LICENSE CITATION.cff Makefile .github artifacts/synthetic-smoke && git commit -m "docs: prepare portfolio release"`.

- [ ] **Step 6: Publish the public repository**

  Create a public GitHub repository named `news-ctr-ranking`, set the description to `Leakage-safe news click ranking with EB-NeRD, temporal validation, and group-aware evaluation`, push the default `main` branch, add topics `machine-learning`, `recommender-system`, `learning-to-rank`, `ctr-prediction`, `ebnerd`, and `portfolio-project`, then verify the public repository page and CI status.
