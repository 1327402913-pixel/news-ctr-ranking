# News CTR Ranking Design

## Purpose

Build a portfolio-grade, reproducible machine-learning repository for ranking news articles within an impression. The project demonstrates problem framing, leakage-safe feature engineering, temporal validation, pointwise and learning-to-rank models, group-aware evaluation, and research communication.

## Prediction task

For every impression, the system receives a user, a timestamp, context, a click history, and a list of candidate news articles. It produces one score per candidate and ranks the candidates by estimated click propensity. A candidate clicked in that impression has label `1`; other exposed candidates have label `0`.

The impression is the atomic evaluation group. The implementation must never create evaluation negatives by mixing candidates from unrelated impressions.

## Dataset and licensing

The primary dataset is EB-NeRD. Users obtain the demo, small, or large bundle from the official dataset site after accepting its license. Raw EB-NeRD files are never committed, repackaged, or uploaded by this project.

The expected layout is:

```text
data/ebnerd_demo/
├── articles.parquet
├── train/
│   ├── behaviors.parquet
│   └── history.parquet
└── validation/
    ├── behaviors.parquet
    └── history.parquet
```

The repository includes a deterministic synthetic dataset with the same essential schema for tests and smoke runs. Synthetic results are labeled as engineering verification and never presented as EB-NeRD benchmark results.

## Internal data contract

The loader normalizes source columns to these canonical names:

- Behaviors: `impression_id`, `user_id`, `impression_time`, `article_ids_inview`, `article_ids_clicked`, and optional `device_type`, `is_sso_user`, `is_subscriber`.
- History: `user_id`, `article_id_fixed`, and optional `impression_time_fixed`, `read_time_fixed`, `scroll_percentage_fixed`.
- Articles: `article_id`, `title`, `subtitle`, `published_time`, `category`, `premium`, and optional `article_type`, `sentiment_score`.

Validation fails with actionable messages when required columns are absent, candidate lists are empty, clicked articles are not in view, labels are invalid, or timestamps cannot be parsed.

## Architecture

```text
Parquet bundle
    -> schema validation and canonical normalization
    -> impression-safe candidate expansion
    -> temporal train/validation split
    -> structured and text feature builder
    -> pointwise logistic baseline or LightGBM LambdaRank
    -> group-aware ranking metrics and artifact report
```

The Python package is split into focused modules:

- `data.py`: loading, normalization, auditing, candidate expansion, temporal split, and synthetic fixtures.
- `features.py`: article text representation, user-interest aggregation, and leakage-safe numeric features.
- `models.py`: model fitting and scoring behind a small common interface.
- `metrics.py`: impression-grouped AUC, MRR, and NDCG.
- `reporting.py`: JSON, Markdown, predictions, and feature-importance artifacts.
- `cli.py`: `audit`, `make-synthetic`, and `train` commands.

## Feature design

The core feature matrix contains only numeric values and excludes raw user IDs and article IDs.

Context features:

- hour of day and day of week;
- device, SSO, and subscription status when present;
- candidate position and candidate count.

Article features:

- publication age in hours using only the impression timestamp;
- title and subtitle lengths;
- category code, premium flag, article type code, and sentiment score.

User-history features:

- click-history length;
- category affinity between the candidate and historical clicks;
- cosine similarity between a candidate text vector and the mean vector of historical clicked articles;
- similarity to the most recent historical article when history timestamps exist.

Text vectors use word and character TF-IDF followed by Truncated SVD. The default model stays CPU-friendly. The transformer embeddings distributed separately by EB-NeRD are outside the required scope.

Features such as `total_inviews`, `total_pageviews`, and `total_read_time` are excluded because their seven-day aggregation window may include information unavailable at the impression timestamp.

## Models

Two model families use the same features and temporal evaluation protocol:

1. Logistic regression is the mandatory interpretable pointwise baseline. Numeric features are median-imputed and standardized inside a fitted pipeline.
2. LightGBM LambdaRank is an optional ranking model. Rows are sorted by impression, group sizes are passed explicitly, and the objective is `lambdarank`.

The CLI defaults to logistic regression so the core project installs without platform-specific native dependencies. Installing the `ranking` extra enables LightGBM.

## Evaluation

Metrics are calculated per impression and averaged across valid groups:

- Group AUC, excluding groups that do not contain both labels;
- MRR, using the rank of the highest-ranked clicked article;
- NDCG@5 and NDCG@10;
- log loss over candidate probabilities for the pointwise baseline.

Temporal splitting uses ordered unique impression timestamps. All impressions at a timestamp remain in one split. The maximum training timestamp must be strictly earlier than the minimum validation timestamp.

The report records dataset fingerprint, row/group counts, feature names, model configuration, random seed, metrics, and skipped-group counts.

## Reliability and error handling

- Randomness is controlled by a single seed, default `42`.
- Every CLI failure exits non-zero and explains the violated data contract.
- Output paths are created safely; raw input files are never modified.
- Predictions retain `impression_id`, `article_id`, `label`, and `score` for independent metric reproduction.
- Tests cover schema failures, expansion, temporal splitting, feature finiteness, metrics, model training, CLI smoke flow, and deterministic output.

## Repository presentation

The public repository includes an English-first README with a concise Chinese summary, architecture diagram, reproducible quick start, dataset-license instructions, methodology, leakage decisions, example synthetic smoke results, limitations, and interview talking points. It also includes an MIT license for the code, citation metadata, a model card template generated with every run, and GitHub Actions for Python 3.10 through 3.12.

## Success criteria

- A fresh environment can install the core package with `pip install -e ".[dev]"`.
- `news-ctr make-synthetic` followed by `news-ctr train` completes on CPU in under one minute on ordinary hardware.
- The complete test suite and Ruff checks pass.
- The generated report contains finite ranking metrics and clearly identifies the synthetic dataset.
- The repository is committed with no raw data, credentials, virtual environment, or generated cache files.
