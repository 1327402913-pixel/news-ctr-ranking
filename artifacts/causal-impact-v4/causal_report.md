# Causal Impact V4: News Ranking Rollout

> Evidence tier: `synthetic-quasi-experiment`. This is deterministic synthetic quasi-experimental
> teaching evidence, not production lift and not a real deployment recommendation.

## Decision question

Did activating the candidate ranking policy increase market-week CTR by more than the
predeclared practical threshold of 0.2000 percentage points?

**Policy result:** `supports_incremental_impact`

## Identification strategy

The primary Difference-in-Differences model uses candidate-exposure-weighted least squares,
market fixed effects, week fixed effects, and uncertainty clustered by `market_id`.
The estimand is the treated-versus-control change after rollout, conditional on common time
shocks and time-invariant market differences.

## Integrity summary

- Markets: 60
- Pre/post weeks: 20/12
- Missing market-week share: 0.0000
- Structural integrity passed: True

## Primary estimate

The estimated absolute CTR effect is **0.6123 percentage points**
(confidence interval: 0.5589 to 0.6658 percentage points;
p=0.0000). The interval, not the point estimate alone, drives the
predeclared decision status.

## Event study and Parallel trends

The event study omits relative week -1 and estimates treated-market
interactions across weeks -12 through 8.
The joint Parallel trends test covers 11 pre-treatment leads:
p=0.6314; passed=True. Passing is supportive
evidence for the identifying assumption, not proof.

## Placebo rollout

The Placebo rollout at week -8 uses only observations before the real
rollout. Its estimate is -0.0115 percentage points with interval
[-0.0768, 0.0538];
passed=True.

## Pre-period balance

| Metric | Treated mean | Control mean | Standardized difference |
|---|---:|---:|---:|
| market_size_index | 1.2259 | 0.8598 | 2.6407 |
| candidate_exposures | 17069.5983 | 12007.6467 | 2.6111 |
| ctr | 0.1244 | 0.1080 | 2.2858 |

Balance is descriptive. It does not establish that post-treatment trends are unconfounded.

## Business translation

Per 1,000,000 candidate exposures, the model implies
**6,123 incremental clicks** with interval
[5,589, 6,658]. No revenue,
margin, or lifetime-value assumption is added.

## Interpretation, assumptions, and limitations

The result is `supports_incremental_impact` under the predeclared policy. Interpretation requires
parallel untreated potential-outcome trends, no rollout anticipation, stable treatment, and no
spillovers between markets. Event studies and placebos can reveal some violations but cannot
exclude unobserved time-varying confounding. The data are synthetic, so the analysis demonstrates
workflow competence rather than external validity or production impact.

## Provenance

- `config_sha256`: `8e79b1e74f48e005ea1456895d7b3372b93a2407a6ad108babed2ac0498c0482`
- `input_sha256`: `6ee7241c37c8ee66597b5488f69cfaf562bb2de3f375b30b56d5f650e89e1b88`
- `metadata_sha256`: `aa9c096da4567bc19d29fd3499765c999dc5aa1381032f582aee62635d1182ba`

## Reproduce

Run `make causal-impact-v4` from a clean checkout. Stored CSV files retain full numeric precision;
rounding above is presentation-only.
