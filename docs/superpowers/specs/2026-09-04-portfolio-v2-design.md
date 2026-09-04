# News CTR Ranking Portfolio V2 Design

**Date:** 2026-09-04

**Status:** Proposed for implementation

## Objective

Upgrade the repository from a well-engineered training pipeline into a stronger
data-analysis and machine-learning portfolio case study. The V2 release must make
model quality, uncertainty, failure modes, and reproducible inference visible to a
recruiter without overstating synthetic results.

The project will remain focused on offline news ranking. It will not become a generic
MLOps demo or add a neural model solely for keyword coverage.

## Why This Scope

The current project already demonstrates data contracts, strict temporal validation,
leakage-aware feature engineering, group-aware ranking metrics, a logistic baseline,
LambdaRank, tests, packaging, and CI. Its main weakness is evidence: there is no unified
baseline comparison, uncertainty estimate, segment analysis, feature ablation, or
reusable scoring artifact.

V2 therefore prioritizes experimental rigor and an end-to-end model lifecycle. A thin
batch inference interface is included because it proves that training output is usable.
HTTP serving, Docker, and deep neural recommenders are deliberately deferred.

## User-Facing Workflows

### Benchmark

```bash
news-ctr benchmark \
  --data data/synthetic \
  --output artifacts/portfolio-v2 \
  --models position,popularity,logistic,lightgbm \
  --bootstrap-samples 200 \
  --seed 42
```

The command will load and audit the same EB-NeRD-shaped bundle used by `train`, apply
one shared temporal split, fit every requested approach on training-visible data only,
and evaluate every score on the identical validation impressions.

Supported approaches:

1. `position`: score earlier in-view positions higher. This is an exposure-bias
   diagnostic, not a deployable recommender.
2. `popularity`: rank by smoothed article click-through rate estimated only from the
   training candidate rows, with a global-prior fallback for unseen articles.
3. `logistic`: the existing pointwise, calibrated probability baseline.
4. `lightgbm`: the existing LambdaRank implementation with impression group sizes.

The default model list will be `position,popularity,logistic`; LightGBM remains behind
the existing `ranking` extra. Requesting an unavailable optional model must fail with
the current actionable installation guidance.

### Train and Save

The existing `train` command will additionally write `model.joblib` and
`model_metadata.json`. The saved object contains the fitted feature builder, fitted
estimator, feature names, model kind, dataset fingerprint, and training configuration.
Metadata will also record the Python, scikit-learn, and optional LightGBM versions.

Because joblib uses pickle semantics, documentation will state that users must load
only artifacts they created or otherwise trust.

### Batch Rank

```bash
news-ctr rank \
  --model artifacts/portfolio-v2/runs/logistic/model.joblib \
  --candidates data/synthetic/validation/behaviors.parquet \
  --output artifacts/scored-validation.parquet \
  --top-k 5
```

`rank` will expand EB-NeRD behavior rows when necessary, use the saved feature builder,
score candidates, sort within each impression, assign one-based ranks, and optionally
retain only the top K. The output columns will be `impression_id`, `article_id`,
`score`, and `rank`, with original labels omitted from the serving-shaped result.

## Experiment Outputs

`benchmark` will create:

```text
artifacts/portfolio-v2/
├── benchmark_config.json
├── leaderboard.csv
├── confidence_intervals.csv
├── ablations.csv
├── segment_metrics.csv
├── benchmark_report.md
└── runs/
    ├── position/
    ├── popularity/
    ├── logistic/
    └── lightgbm/
```

Each model run retains its predictions and metrics. `leaderboard.csv` contains one row
per model with group AUC, MRR, NDCG@5, NDCG@10, fit time, scoring time per 1,000
candidates, and serialized model size when applicable.

The report must state at the top whether the dataset is synthetic or EB-NeRD. Synthetic
results will always be labeled as engineering evidence, never as a real recommendation
benchmark.

## Statistical Evaluation

Confidence intervals will use impression-level bootstrap resampling. Entire impression
groups are sampled with replacement so candidate dependence is preserved, and every
sampled occurrence receives a fresh bootstrap group identifier so duplicate draws do
not collapse into one group. The same seed and sampled group indices will be reused
across models, enabling paired model comparisons. The report will show the mean and 95%
percentile interval for group AUC, MRR, NDCG@5, and NDCG@10.

The implementation must reject fewer than two bootstrap samples, non-binary labels,
non-finite scores, and length mismatches. Samples in which group AUC is undefined are
excluded only from that metric and counted in the output.

## Feature Ablations

A feature-group registry will assign every engineered feature to exactly one group:

- `context`: time, device, subscriber state, candidate position, and candidate count.
- `article`: publication age, freshness, text lengths, category, premium state,
  article type, and sentiment.
- `personalization`: history length and category affinity.
- `semantic`: long-term and recent text similarity.

The benchmark will run logistic regression with the following deterministic variants:

- `context_only`
- `no_position`
- `no_semantic`
- `no_personalization`
- `full`

Every feature must belong to one group, and an unknown or empty selection must raise a
clear error. Ablations use the same split and validation impressions as the main
leaderboard.

## Segment and Bias Analysis

Validation ranking metrics will be sliced by impression-level signals already present
in the data contract:

- device type;
- candidate-set size bucket;
- user-history length bucket.

Article-freshness buckets and candidate positions will be reported separately as
candidate-level empirical click-rate diagnostics. They will not be presented as ranking
metrics because filtering candidates would change the original impression task.

Small slices with fewer than five impressions will be retained but marked
`low_support=true`. The report will never present slice differences as causal effects.
Position click rates are explicitly described as exposure-bias evidence.

## Architecture

New modules will keep responsibilities separate:

- `baselines.py`: fitted position and smoothed-popularity scorers.
- `evaluation.py`: paired group bootstrap, feature ablations, segment metrics, timing,
  and leaderboard assembly.
- `persistence.py`: trusted-artifact save/load and saved-ranker schema.
- `benchmarking.py`: orchestration over a single audited split.

Existing modules remain responsible for their current domains:

- `data.py`: loading, auditing, expansion, and temporal split.
- `features.py`: fitted leakage-aware feature transformations and feature groups.
- `models.py`: logistic and LambdaRank estimator adapters.
- `metrics.py`: deterministic point estimates.
- `reporting.py`: machine-readable artifacts and Markdown rendering.
- `cli.py`: argument parsing and concise user-facing errors.

The CLI will call orchestration functions rather than contain experiment logic.

## Data Flow

```text
audited bundle
    -> one temporal train/validation split
    -> training-only baseline statistics + fitted feature builder
    -> shared validation candidates
    -> baseline/model scores
    -> point metrics + paired bootstrap intervals
    -> feature ablations + segment diagnostics
    -> machine-readable tables + recruiter-facing report
    -> trusted saved model -> batch rank command
```

No validation labels, validation clicks, or future article aggregates may influence
training features, popularity statistics, model fitting, or imputation values.

## Error Handling

Expected data, configuration, dependency, and filesystem errors will continue to return
CLI exit code 2 without a traceback. New actionable failures include unknown model
names, duplicate model names, invalid top-K values, incompatible saved-artifact schema,
missing columns for batch scoring, and corrupt artifacts. The saved format will include
an explicit schema version; trust remains a documented user responsibility because a
pickle-based artifact cannot establish its own provenance.

Partial benchmark results will be written to a temporary directory and atomically moved
into place only after all requested models and reports succeed. A failed run must not
look complete.

## Testing Strategy

Development will remain test-driven.

Unit tests will cover:

- smoothed popularity using training labels only and unseen-article fallback;
- deterministic position scores;
- impression-level bootstrap determinism and interval bounds;
- paired sampling across models;
- complete feature-group membership and ablation selection;
- segment bucketing and low-support flags;
- saved-artifact round trips and incompatible-schema errors.

Integration tests will cover:

- a deterministic synthetic `benchmark` run and its complete artifact tree;
- reproducible leaderboard and confidence intervals with the same seed;
- `train` followed by `rank`, including correct within-impression ordering and top K;
- a deliberate failed benchmark leaving no complete output directory.

CI will run the core suite on Python 3.10, 3.11, and 3.12. The Linux ranking job will
run the full four-model synthetic benchmark so LambdaRank and native LightGBM loading
remain tested.

## Portfolio Presentation

The README will gain a recruiter-oriented opening section containing:

- a 60-second project tour;
- a compact leaderboard and ablation table;
- three explicit engineering decisions and their business implications;
- a truthful limitations box;
- two ready-to-adapt resume bullets;
- one-command synthetic reproduction and one-command licensed EB-NeRD reproduction.

The committed V2 report will use deterministic synthetic data and carry the required
warning. The repository will be ready to regenerate the same report on EB-NeRD demo,
small, or large after the user independently accepts the dataset license and places the
files under `data/`.

## Out of Scope

- Accepting the EB-NeRD license or redistributing its raw or derived row-level data.
- Claiming production lift, causal impact, or leaderboard standing.
- Neural recommenders such as NRMS, LSTUR, NPA, or transformer fine-tuning.
- HTTP APIs, Docker, Kubernetes, cloud deployment, feature stores, or streaming.
- Online A/B testing and inverse-propensity correction.

These are potential follow-ups only after a real-data benchmark establishes a credible
experimental baseline.

## Acceptance Criteria

1. All existing behavior remains backward compatible.
2. `benchmark` produces every declared table and a self-contained Markdown report.
3. Repeated runs with the same data and seed produce identical metric tables.
4. Bootstrap resampling preserves impression groups and produces paired intervals.
5. Ablations cover every registered feature group without leakage.
6. `train` produces a loadable model artifact and `rank` produces correctly ordered
   top-K candidates.
7. Synthetic and real-data claims remain visibly separated in every report.
8. The core suite passes on Python 3.10-3.12 and the LightGBM benchmark passes on Linux.
9. The committed synthetic report is generated by the public CLI, not hand edited.
10. README commands work from a clean installation and the package builds as both sdist
    and wheel.
