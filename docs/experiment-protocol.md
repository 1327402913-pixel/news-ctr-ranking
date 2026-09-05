# Experiment protocol

## Evidence tiers and permitted claims

This repository keeps three evidence types separate:

| Tier | Source | Supports | Does not support |
| --- | --- | --- | --- |
| `synthetic-observational` | Deterministic EB-NeRD-shaped ranking logs | Data contracts, SQL correctness, reproducibility, ranking workflow | Real ranking quality, causal lift |
| `synthetic-rct` | Deterministic user-level randomized fixture | Power/SRM/ITT/CUPED/decision-code behavior | Production lift or external validity |
| `licensed-ebnerd` | User-provided licensed public dataset | Aggregate offline ranking quality on the named split | Online causal impact |

Reports display the tier and a claim boundary. Synthetic numbers must never be
presented as EB-NeRD or production results.

## Randomized decision protocol

The committed [`experiment-v3.json`](../configs/experiment-v3.json) is the analysis
contract. Randomization occurs once per `user_id`, with one outcome row per user and
expected 50/50 allocation. The primary outcome is click; dwell time and latency are
continuous non-inferiority guardrails. `pre_ctr` is explicitly recorded before
treatment and is the only permitted CUPED covariate.

Before estimating effects, the analyzer requires:

- unique, non-null randomization units;
- exactly the configured control and treatment arms;
- finite configured outcomes and covariates;
- binary values restricted to zero or one;
- positive arm counts and no missing segment values;
- an SRM chi-square test against the declared allocation.

The primary estimand is the user-level intent-to-treat difference in means. The raw
estimate uses a two-sided normal confidence interval with a Welch standard error.
CUPED estimates one pooled coefficient from the pre-treatment covariate, transforms
the outcome, and applies the same ITT calculation. It does not condition on treatment
outcomes or post-treatment variables.

Device and history-segment estimates are exploratory by default. Low-support cells
remain visible rather than being silently dropped. If a segment column is declared
confirmatory, Holm step-down adjusted p-values and rejection decisions control the
family-wise error rate for zero-effect tests. Bonferroni simultaneous intervals are
reported separately; exploratory cells retain pointwise intervals.

The deterministic decision policy has four states:

1. `invalid_experiment` when SRM or integrity checks fail;
2. `launch` when the CUPED primary lower bound clears the practical threshold and all
   guardrails pass;
3. `do_not_launch` for a credibly harmful primary result or a failed guardrail;
4. `continue_experiment` for intervals that cross a decision boundary.

See the concise [launch memo](launch-decision.md) for the fixed synthetic run.

Only `synthetic-rct` metadata is accepted by the current experiment-decision command.
Observational evidence tiers are rejected before estimation. The sidecar unit,
allocation, and row count must match the input frame and config. Each persisted run
copies the exact predeclared config and records SHA-256 hashes for the Parquet input,
metadata sidecar, and config.

## Observational SQL protocol

The DuckDB layer operates on observed recommendation exposures. Candidate CTR is
defined as clicked candidate rows divided by candidate exposures. Segment outputs
show `units` (distinct impressions), `candidates` (the exposure denominator), and
`clicks` (the numerator), and flag small cells. These metrics are descriptive; no SQL
slice is interpreted as a treatment effect.

## Research question

Can leakage-safe context, article, history, and semantic features improve
within-impression news ranking beyond exposure-sensitive position and training-only
popularity baselines?

The unit of evaluation is an impression: one user request with its observed candidate
set. Candidate rows from different impressions must never be mixed into one ranking
group.

## Shared comparison

Every benchmark uses one audited split and evaluates every requested approach on the
same ordered validation rows:

1. `position`: negative zero-based in-view position; an exposure-bias diagnostic, not
   a deployable recommender;
2. `popularity`: article click rate smoothed toward the global click rate, using only
   training candidate labels and a global fallback for unseen validation articles;
3. `logistic`: balanced pointwise logistic regression over the full feature matrix;
4. `lightgbm`: LambdaRank with impression group sizes and the same features.

Fit time, scoring time per 1,000 candidates, and serialized model size accompany the
quality metrics. Timing is machine-dependent and must not be compared across unrelated
environments.

## Split and fitting policy

Use EB-NeRD's supplied train and validation bundles when both are present. The maximum
training impression timestamp must be strictly earlier than the minimum validation
timestamp.

If only one behavior bundle exists, split ordered unique timestamps. Every impression
at the cutoff timestamp remains in one fold. Never split random candidate rows.

- Fit text vocabulary, SVD, category encoders, and imputation values on
  training-visible data.
- Derive user interests from history supplied before the evaluated behavior period.
- Calculate article age from `impression_time - published_time`.
- Exclude raw user, article, and impression IDs from the feature matrix.
- Exclude aggregates whose windows can extend past the impression timestamp.
- Estimate popularity from training labels only; validation clicks are evaluation-only.

## Ranking metrics

Report group AUC, MRR, NDCG@5, and NDCG@10. Metrics are calculated independently per
impression and then averaged.

- Group AUC excludes impressions containing only one label class and records how many
  were skipped.
- MRR is the reciprocal rank of the first clicked candidate, or zero when no candidate
  is relevant.
- NDCG discounts relevant candidates lower in the predicted ranking.
- Log loss is reported only for models that emit calibrated click probabilities.

Preserve predictions with impression IDs in per-run artifacts so point estimates can
be independently reproduced. These row-level artifacts are ignored in the committed
synthetic portfolio output.

## Paired impression bootstrap

Confidence intervals resample complete impressions with replacement. If the same
impression is drawn twice, each occurrence receives a new bootstrap group ID; otherwise
duplicate draws would collapse and understate their weight.

The random draw for each bootstrap replicate is shared by all models. This paired
design preserves candidate dependence within impressions and makes model comparisons
less noisy than unrelated resamples. The report uses the 2.5th and 97.5th percentiles
for a 95% interval. A replicate where group AUC is undefined is omitted only from that
metric and counted through `valid_samples`.

Intervals quantify sampling uncertainty in the logged validation impressions. They do
not correct exposure bias, establish causality, or predict online lift.

## Fixed logistic ablations

Every engineered feature belongs to exactly one registry group: `context`, `article`,
`personalization`, or `semantic`. The deterministic matrix is:

| Ablation | Included features | Question |
| --- | --- | --- |
| `context_only` | Context group only | How much does request and exposure context explain? |
| `no_position` | Full set except `candidate_position` | How sensitive is quality to an exposure-biased signal? |
| `no_semantic` | Full set except semantic similarity | Does lightweight text matching add evidence? |
| `no_personalization` | Full set except history length and category affinity | Does user history add evidence? |
| `full` | All registered features | Reference point for deltas |

Ablations use the same split, rows, seed, estimator, and validation impressions. A zero
delta on synthetic data means the generator/run did not provide evidence for that
feature family; it is not proof that the family is useless on real data.

## Segment interpretation

Ranking slices retain whole impressions and cover three impression-constant signals:

- device type;
- candidate count: `small` (≤5), `medium` (6–10), `large` (>10);
- history length: `cold` (0), `short` (1–5), `long` (>5).

A slice with fewer than five impressions remains in the table but is marked
`low_support=true`. Segment differences are descriptive. Do not claim fairness,
causality, or product impact from them.

## Candidate diagnostics are not ranking metrics

Candidate position and article freshness vary inside an impression. Filtering to one
position or freshness bucket would alter the original ranking problem, so these fields
are reported separately as candidate counts, clicks, and empirical click rates.

Freshness buckets are `<24h`, `1-3d`, `3-7d`, and `7d+`. Position click-rate patterns
are explicitly labeled exposure-bias diagnostics. They do not estimate the causal
effect of moving an item to a different position.

## Reproducibility and artifact policy

The benchmark stores its configuration, dataset fingerprint, quality tables,
confidence intervals, ablations, segments, candidate diagnostics, per-model runs, and
a Markdown report. Output is assembled in a temporary sibling directory and renamed
only after every requested model and report succeeds.

`train` also saves a schema-versioned `model.joblib` and JSON metadata with feature
names and library versions. Joblib is pickle-based: load only artifacts created by a
trusted source.

## Interpretation rules

- An offline gain does not prove an online CTR gain.
- Position may predict clicks because an earlier system selected the exposure order.
- AUC measures pair ordering; MRR and NDCG emphasize top ranks. Discuss disagreements.
- Overlapping bootstrap intervals weaken claims of a reliable model difference.
- Synthetic results validate pipeline behavior only.
- A real portfolio claim requires a licensed EB-NeRD run and a reviewed report.
