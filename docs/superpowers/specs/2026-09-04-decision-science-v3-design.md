# Decision Science V3 Design

## Objective

Upgrade the ranking portfolio from a strong offline machine-learning benchmark into a
decision-science case study that demonstrates the candidate's combined mathematics,
economics, statistics, and applied-ML background. The project must answer a product
decision, not merely identify the model with the highest offline score:

> Under exposure bias and heterogeneous user response, is a new ranking policy worth
> testing or launching, for whom, and with what statistical evidence?

The primary hiring target is product/decision data science, with recommendation data
science as a secondary target. Pure ML infrastructure and deep neural ranking are not
the priority for this release.

## Hiring Evidence Contract

V3 must make five capabilities visible within a 90-second repository review:

1. **SQL analytics:** reproducible metric marts, funnels, cohorts, segments, and data
   quality checks built from Parquet inputs.
2. **Statistical experimentation:** power and minimum-detectable-effect calculations,
   sample-ratio-mismatch checks, intent-to-treat estimation, confidence intervals,
   CUPED variance reduction, guardrails, and predeclared decision rules.
3. **Applied machine learning:** the existing leakage-safe ranking benchmark, saved
   models, and batch scoring remain intact.
4. **Economic reasoning:** treatment effects, selection and exposure bias, opportunity
   cost, metric trade-offs, and uncertainty are translated into a launch decision.
5. **Communication:** a concise decision memo and visual evidence explain the result to
   both technical and product reviewers.

Every artifact must identify its evidence tier:

- `synthetic-rct`: a randomized simulation with known assignment and outcome process;
- `synthetic-observational`: deterministic EB-NeRD-shaped engineering evidence;
- `licensed-ebnerd`: aggregate evidence generated locally from user-obtained data.

Synthetic or observational outputs must never be described as production lift. Raw or
row-level licensed EB-NeRD data must never be committed.

## Chosen Approach

### Recommended: integrated decision-science layer

Keep the existing ranking package and add SQL analytics, a statistically rigorous
experiment workflow, and decision reporting around it. This preserves the strongest
V2 evidence while addressing the largest data-science hiring gaps.

### Rejected: deep-learning-first upgrade

A two-tower, DIN, or transformer ranker would add algorithmic breadth but would not
demonstrate SQL, causal reasoning, experiment design, or product judgment. It can be a
later extension after real-data baselines exist.

### Rejected: separate analytics-only rewrite

A standalone dashboard or notebook would show product analytics but discard the
existing auditable ranking system. It would also make the portfolio appear fragmented
rather than end to end.

## User-Facing Workflows

### SQL product analytics

```bash
news-ctr analyze \
  --data data/synthetic \
  --output artifacts/analytics-v3
```

The command loads normalized candidate-level impressions into an ephemeral DuckDB
database and executes versioned SQL files. It publishes aggregate CSV and Markdown
artifacts atomically. The same command accepts licensed EB-NeRD bundles without
changing metric definitions.

Required outputs:

- `data_quality.csv`: row counts, key uniqueness, null rates, timestamp bounds, and
  label validity;
- `kpi_summary.csv`: impressions, candidates, clicks, CTR, mean candidates, and active
  users;
- `exposure_funnel.csv`: impression, click, and available deep-read stages;
- `segment_kpis.csv`: device, history-length, candidate-count, position, and freshness
  slices with denominators;
- `user_cohorts.csv`: first-observed cohort and subsequent active-period retention when
  the source has sufficient dates;
- `analytics_report.md`: definitions, SQL provenance, limitations, and interview-ready
  findings.

The SQL is executed in production code and tested; it must not be decorative example
text. Metric definitions live in one catalog and each output includes its denominator.

### Experiment planning

```bash
news-ctr experiment-design \
  --baseline-rate 0.12 \
  --relative-mde 0.05 \
  --alpha 0.05 \
  --power 0.80
```

The command reports absolute MDE, required users per arm, expected duration for a given
daily eligible population, and assumptions. Binary outcomes use a two-proportion
normal approximation with conservative pooled variance. Inputs are validated against
explicit numeric domains.

### Randomized experiment analysis

```bash
news-ctr make-experiment \
  --output data/synthetic-rct.parquet \
  --seed 42 \
  --users 20000

news-ctr experiment \
  --input data/synthetic-rct.parquet \
  --output artifacts/experiment-v3 \
  --config configs/experiment-v3.json
```

The deterministic generator creates a public teaching fixture with:

- stable user-level random assignment;
- a pre-period covariate correlated with the primary outcome;
- one primary binary outcome;
- two guardrails, including one continuous outcome;
- device and history-length segments;
- a declared treatment effect and optional assignment corruption for SRM tests.

The analyzer requires one row per randomization unit and refuses duplicated users,
missing arms, invalid binary outcomes, non-finite values, or post-treatment covariates
declared as CUPED inputs.

Required outputs:

- `experiment_summary.csv`: arm counts and raw metrics;
- `srm.json`: chi-square statistic, p-value, and pass/fail result;
- `effects.csv`: absolute and relative effects, standard errors, confidence intervals,
  and p-values for primary and guardrail metrics;
- `cuped_effects.csv`: predeclared covariate coefficient, adjusted effect, standard
  error, confidence interval, and variance-reduction percentage;
- `heterogeneous_effects.csv`: predeclared segment estimates with sample counts,
  Holm-adjusted p-values/rejections, Bonferroni simultaneous intervals, or explicit
  exploratory labels;
- `decision_report.md`: launch/no-launch/continue decision and rationale;
- chart images suitable for direct rendering in GitHub README.

The primary estimator is intent-to-treat. CUPED is allowed only with a pre-treatment
covariate. Segment analyses are exploratory unless declared confirmatory in the config.
The implementation will not apply propensity-score methods to EB-NeRD click logs when
the logging propensity is unavailable.

## Decision Rule

The experiment configuration declares before analysis:

- the primary metric and direction;
- alpha and confidence level;
- minimum practically important effect;
- guardrail metrics, directions, and non-inferiority margins;
- expected allocation ratio;
- confirmatory segments, if any.

The generated recommendation is deterministic:

- `launch` only when SRM passes, the primary confidence interval clears the practical
  threshold, and every guardrail passes its margin;
- `do_not_launch` when SRM passes but the primary effect is credibly harmful or a
  guardrail fails;
- `continue_experiment` when evidence is inconclusive;
- `invalid_experiment` when SRM or data-integrity checks fail.

The report must show the inputs to the decision so that readers can disagree with the
policy without reverse-engineering code.

## Architecture

```text
Parquet bundle ──> existing schema audit/expansion
                         │
                         ├──> ranking benchmark (V2, unchanged)
                         │
                         └──> DuckDB metric layer ──> aggregate analytics report

Synthetic RCT/config ──> experiment validation
                         ├──> SRM and raw ITT effects
                         ├──> CUPED and segment effects
                         └──> predeclared decision policy ──> decision memo + charts
```

New components:

```text
configs/
└── experiment-v3.json # predeclared metrics, margins, allocation, and decision rule

sql/
├── data_quality.sql
├── kpi_summary.sql
├── exposure_funnel.sql
├── segment_kpis.sql
└── user_cohorts.sql

src/news_ctr/
├── analytics.py       # DuckDB setup, SQL execution, result contracts
├── experiment_data.py # deterministic randomized teaching fixture
├── experiments.py     # design, validation, SRM, ITT, CUPED, segments
├── decisioning.py     # predeclared launch policy and memo model
└── visualization.py   # deterministic static charts
```

`cli.py` remains a thin adapter. Orchestration and atomic output publication follow the
existing benchmark pattern. Statistical calculations use NumPy, pandas, and SciPy;
DuckDB and Matplotlib are optional `decision` dependencies so the core ranking install
stays lightweight.

## Data and Privacy Boundaries

- Randomization and analysis operate at user level to avoid treating repeated rows as
  independent units.
- Public synthetic fixtures contain no personal information.
- Licensed data stays under ignored `data/`; only aggregate outputs with minimum cell
  counts are eligible for publication.
- Segment tables suppress or flag cells below a configurable minimum support.
- Data fingerprints are content-stable and included in every report.
- The analyzer never silently drops invalid observations; validation errors identify
  the offending field and count.

## Visual and Recruiter Experience

The README receives a new three-minute decision-science tour after the current ranking
summary. It will show:

1. one KPI/funnel chart produced by SQL;
2. one treatment-effect forest plot with primary and guardrail intervals;
3. the deterministic decision and its rule;
4. a capability map connecting SQL, statistics, economics, and ML;
5. truthful resume bullets for synthetic evidence and a separate template for a future
   licensed EB-NeRD run.

The public artifacts use the synthetic RCT and remain explicitly labeled. A licensed
EB-NeRD result is a separate, optional evidence upgrade and never blocks CI.

## Testing and Verification

Implementation is test-driven. Tests must cover:

- SQL outputs against small hand-computed fixtures;
- metric denominators, empty slices, nulls, duplicate keys, and insufficient dates;
- power monotonicity and known numerical examples;
- SRM pass and fail fixtures;
- binary and continuous ITT estimates against hand calculations;
- CUPED invariance at zero correlation and variance reduction on a fixed correlated
  fixture;
- rejection of post-treatment CUPED covariates;
- segment support and multiplicity behavior;
- all four decision states;
- deterministic synthetic generation and chart/report reproduction;
- atomic publication on success and cleanup on failure;
- end-to-end CLI operation on Linux and Python 3.10 through 3.12.

The release gate is:

- all tests pass with at least 90% package coverage;
- Ruff check and format pass;
- source and wheel builds pass;
- ranking-extra CI remains green;
- a new decision-extra CI job runs SQL analytics plus the complete synthetic RCT flow;
- clean-checkout artifacts reproduce byte-for-byte except documented runtime fields;
- an independent code review reports no critical or important findings.

## Scope Boundaries

V3 deliberately excludes:

- claims of online lift or causal impact from observational EB-NeRD clicks;
- acceptance of the EB-NeRD license on the user's behalf;
- deep neural recommenders, GPU training, feature stores, or streaming;
- Docker, Kubernetes, and low-latency HTTP serving;
- a large collection of shallow causal estimators without a credible identification
  strategy;
- live production experimentation.

These can be reconsidered after V3 demonstrates the higher-value data-science evidence.

## Success Criteria

V3 is complete when a reviewer can reproduce the public synthetic workflow and answer:

- What business decision is being made?
- Which SQL-defined metrics and denominators support it?
- Was the experiment valid and adequately powered?
- What is the estimated effect and uncertainty before and after CUPED?
- Did any guardrail or segment change the decision?
- Which evidence is randomized, observational, synthetic, or licensed?
- How does the ranking model connect to the proposed experiment without claiming that
  offline ranking metrics are online causal effects?

The project should then support applications to product data scientist, decision
scientist, experimentation scientist, growth data scientist, and recommendation data
scientist roles without misrepresenting synthetic evidence.
