# Causal Impact V4 Design

## Goal

Add a recruiter-readable quasi-experimental impact study that demonstrates how to
measure a ranking-policy rollout when user-level randomization is unavailable. The
study must complement, not replace, the existing randomized-experiment workflow and
must remain explicit that all committed results use deterministic synthetic data.

## Why This Is the Next Portfolio Increment

The repository already demonstrates leakage-safe ranking, product SQL, randomized
experimentation, CUPED, guardrails, and launch decisioning. A second ranking model or
another isolated dashboard would repeat existing evidence. A market-level
difference-in-differences study instead connects the owner's economics training to
their statistics and data-science training while covering a common product-science
constraint: estimating incremental impact from a phased operational rollout.

V4 is intentionally narrow. It supports a single common adoption date with
never-treated comparison markets. It does not claim to solve staggered-adoption,
interference, endogenous treatment assignment, or unobserved time-varying confounding.

## Evidence Boundary

The committed case study is `synthetic-quasi-experiment` evidence. It validates the
data contracts, estimators, diagnostics, reporting, and release process. It is not a
production result, a licensed external-data result, or proof that a real ranking
change causes additional clicks.

Every public summary, report, plot, and resume bullet must preserve that boundary.
The word `causal` may describe the methodology or estimand, but the synthetic result
must be described as "evidence consistent with incremental impact under the stated
identification assumptions," not as production causal lift.

## User Story

A product team rolls out a new ranking policy to a selected set of markets at the
start of week zero. User-level A/B assignment was operationally unavailable. The data
scientist must determine whether treated markets changed relative to never-treated
markets, whether pre-treatment trends undermine the design, whether a fake earlier
rollout also appears to create an effect, and what the estimated CTR change means per
one million candidate exposures.

## Public Workflow

The workflow adds two CLI commands and one Make target:

```bash
news-ctr make-quasi-experiment \
  --output data/synthetic-market-panel-v4.parquet \
  --seed 42 \
  --markets 60 \
  --pre-weeks 20 \
  --post-weeks 12

news-ctr causal-impact \
  --input data/synthetic-market-panel-v4.parquet \
  --output artifacts/causal-impact-v4 \
  --config configs/causal-impact-v4.json

make causal-impact-v4
```

The generator refuses to overwrite an existing destination. The analyzer writes to
a temporary sibling directory, validates the complete artifact set, and atomically
publishes the requested output only after every step succeeds.

## Data Contract

The panel contains one row per market and integer relative week:

| Column | Type | Meaning |
| --- | --- | --- |
| `market_id` | string | Stable market-level clustering unit |
| `relative_week` | integer | Week relative to the real rollout; zero is first treated week |
| `treated_market` | integer | One for rollout markets, zero for never-treated controls |
| `policy_active` | integer | `treated_market * (relative_week >= 0)` |
| `candidate_exposures` | integer | Positive weekly denominator |
| `clicks` | integer | Weekly clicks in `[0, candidate_exposures]` |
| `ctr` | float | `clicks / candidate_exposures` |
| `market_size_index` | float | Pre-treatment market attribute used for balance reporting only |

The deterministic generator creates 60 markets, split evenly between treatment and
control, with 20 pre-period and 12 post-period weeks. It includes market fixed
effects, common seasonality, exposure variation, and a known +0.006 absolute CTR
treatment effect after week zero. Treatment assignment may depend on observed
pre-treatment market size so that the case study illustrates why level imbalance does
not itself invalidate difference-in-differences. The generator must not introduce a
differential pre-trend.

The generator also writes `<output-stem>.metadata.json` beside the Parquet file,
following the existing synthetic-RCT convention. The sidecar contains the seed,
parameters, known data-generating effect, evidence tier, row count, and SHA-256 of the
Parquet file. The analyzer validates that metadata against the input rather than
trusting it.

## Configuration

`configs/causal-impact-v4.json` predeclares:

- unit, time, treatment-group, active-treatment, exposure, click, and outcome columns;
- rollout week `0` and event-study reference week `-1`;
- analysis window `[-12, 8]` so remote tails do not dominate the diagnostic;
- two-sided `alpha = 0.05`;
- practical threshold `0.002` absolute CTR;
- minimum 20 markets, minimum 8 treated and 8 control markets;
- minimum 8 pre weeks and 4 post weeks;
- maximum missing market-week share `0.0` for the committed balanced-panel study;
- placebo rollout week `-8`, estimated only from weeks before the real rollout;
- business-impact scale of `1_000_000` candidate exposures.

The configuration is immutable after loading, rejects unknown outcome semantics, and
is copied byte-for-byte into the published output. Its SHA-256 is recorded in the run
manifest.

## Validation and Integrity Diagnostics

Analysis fails before estimation when any of these conditions holds:

- required columns are missing;
- the frame is empty or contains duplicate market-week rows;
- market IDs or treatment labels are null;
- treatment membership changes within a market;
- `policy_active` differs from the predeclared rollout rule;
- relative weeks are non-integer or duplicated within a market;
- exposures are non-positive, clicks are negative, or clicks exceed exposures;
- stored CTR differs from clicks divided by exposures beyond `1e-12`;
- outcome, exposure, or balance fields are non-finite;
- treatment/control cluster counts or pre/post windows are below their minima;
- the balanced-panel missingness threshold is exceeded;
- the input metadata hash, evidence tier, or row count is inconsistent.

Published diagnostics include row count, market counts by arm, week range, missing
market-week share, pre/post observations, exposure totals, and all integrity pass/fail
states. Input validation failures leave no output directory.

## Estimation

### Primary Difference-in-Differences

The primary estimand is the exposure-weighted change in market-week CTR for treated
markets after rollout relative to the contemporaneous change in control markets.

The implementation uses weighted least squares with:

- market fixed effects;
- week fixed effects;
- `policy_active` as the treatment term;
- `candidate_exposures` as analytic weights;
- a heteroskedasticity-robust covariance matrix clustered by `market_id`.

The published row includes coefficient, clustered standard error, test statistic,
p-value, 95% confidence interval, treated/control market counts, observation count,
and total exposure. `statsmodels>=0.14` is an explicit `causal` optional dependency.

### Event Study and Parallel Trends

The event study fits exposure-weighted treated-market interactions for each relative
week in `[-12, 8]`, omitting week `-1`. It uses the same market and week fixed effects
and market-clustered covariance. The output contains one ordered row per estimable
relative week with effect, standard error, p-value, and confidence interval.

A joint Wald test evaluates whether all included pre-treatment lead coefficients
equal zero. The diagnostic passes when the p-value is at least `0.05`. Passing is
supportive evidence for the identifying assumption, not proof of parallel trends.

### Placebo Rollout

The placebo analysis discards every row at or after the real rollout and pretends the
policy began at relative week `-8`. It fits the same fixed-effect model on this
pre-period subset. The placebo passes only when its 95% confidence interval contains
zero. The report displays the estimate and interval rather than reducing it to a
badge.

### Balance and Business Translation

Balance reporting compares pre-treatment market-level means for
`market_size_index`, average exposure, and average CTR. It reports treatment/control
means and standardized mean differences. Balance is descriptive and never used to
claim that post-treatment trends are unconfounded.

The business translation multiplies the primary CTR estimate and confidence limits
by 1,000,000. It reports incremental clicks per one million candidate exposures and
does not introduce currency, revenue, or lifetime-value assumptions.

## Interpretation Policy

The report uses four mutually exclusive states:

- `invalid_design`: integrity, cluster-count, parallel-trend, or placebo diagnostics fail;
- `supports_incremental_impact`: diagnostics pass and the primary lower confidence
  bound exceeds the +0.002 practical threshold;
- `evidence_of_no_benefit`: diagnostics pass and the primary upper confidence bound
  is below or equal to zero;
- `inconclusive`: diagnostics pass but neither decisive condition is met.

The committed deterministic fixture should return
`supports_incremental_impact`. This is a test of the policy implementation, not a
recommendation to deploy a real system.

## Components and Boundaries

### `src/news_ctr/quasi_data.py`

Owns deterministic panel generation, generator argument validation, metadata writing,
and input hash creation. It does not fit models or render reports.

### `src/news_ctr/causal.py`

Owns immutable configuration, panel validation, balance tables, primary DiD,
event-study estimation, joint pre-trend testing, placebo estimation, business-impact
translation, and structured analysis results. It does not perform filesystem
publication.

### `src/news_ctr/causal_reporting.py`

Owns interpretation states and the recruiter-readable Markdown report. It consumes
structured analysis results and provenance but does not re-estimate effects.

### `src/news_ctr/causal_visualization.py`

Owns a deterministic event-study PNG with a zero reference line, rollout marker,
confidence intervals, evidence-tier description metadata, fixed dimensions, fixed
colors, and no timestamps. It is byte-stable within one runtime and structurally
verified across platforms.

### `src/news_ctr/cli.py`

Adds `make-quasi-experiment` and `causal-impact`, following the existing CLI and
atomic-output conventions.

### Artifacts

`artifacts/causal-impact-v4/` contains exactly:

1. `causal_report.md`
2. `did_estimate.csv`
3. `event_study.csv`
4. `event_study.png`
5. `diagnostics.json`
6. `placebo_estimate.csv`
7. `balance.csv`
8. `business_impact.csv`
9. `config_snapshot.json`
10. `run_manifest.json`

The manifest records the evidence tier, input and metadata hashes, config hash,
package/runtime versions, command, and exact artifact names. Runtime-version fields
are documented provenance; scientific outputs must remain deterministic for the same
input and configuration.

## Reporting

The report begins with the synthetic-evidence banner and answers, in order:

1. what decision is being evaluated;
2. why randomization was unavailable in the scenario;
3. what the DiD estimand and identifying assumptions are;
4. whether data integrity, pre-trends, and placebo checks pass;
5. the primary estimate and clustered interval;
6. the per-million-exposure translation;
7. what balance and event-study patterns show;
8. the interpretation state;
9. limitations and exact reproduction commands.

README updates add the quasi-experiment to the 60-second tour, architecture diagram,
role-fit table, evidence links, and one truthful resume bullet. The README keeps the
existing randomized experiment prominent so reviewers can contrast experimental and
observational identification.

## Testing Strategy

All production behavior is test-driven. Tests cover:

- deterministic generation and truthful sidecar metadata;
- every validation failure class;
- recovery of the known synthetic effect within its confidence interval;
- clustered standard-error fields and finite estimates;
- ordered event-study rows and the omitted `-1` reference week;
- joint pre-trend and placebo diagnostics on passing and deliberately failing fixtures;
- balance-table and per-million-exposure arithmetic;
- all four interpretation states;
- complete, candid report content;
- atomic CLI publication and overwrite refusal;
- same-runtime PNG byte stability and cross-platform structural validation;
- committed artifact completeness and end-to-end reproduction.

The full package must retain at least 90% statement coverage. Ruff check, Ruff format,
source distribution, and wheel builds must pass.

## Continuous Integration

A `causal-extra` Ubuntu/Python 3.12 job installs `.[dev,causal]`, runs the focused
causal tests, generates the deterministic panel, executes the full analysis, and
compares committed scientific CSV/JSON/Markdown artifacts byte-for-byte. It validates
the PNG's dimensions, mode, description metadata, and non-empty pixel content rather
than claiming cross-platform raster byte identity.

The existing Python 3.10-3.12, ranking-extra, and decision-extra jobs remain green.

## Acceptance Criteria

V4 is complete when:

- both new CLI commands and `make causal-impact-v4` run from a clean checkout;
- the committed fixture returns `supports_incremental_impact` and its interval covers
  the known +0.006 generating effect;
- integrity, joint pre-trend, and placebo diagnostics pass on the committed fixture;
- all ten declared artifacts reproduce under the documented comparison contract;
- the README and report state the identification assumptions and synthetic boundary;
- full tests pass with at least 90% coverage;
- all GitHub Actions jobs pass on the pull request and again on `main`;
- an independent code review reports no Critical or Important findings.

## Explicit Non-Goals

- No staggered-adoption estimator.
- No synthetic-control or propensity-score implementation in V4.
- No notebook-only workflow or interactive dashboard.
- No scraping, bundling, or implying use of proprietary production data.
- No currency or revenue claim without a user-supplied business value assumption.
- No automatic real-world launch recommendation from observational evidence.
