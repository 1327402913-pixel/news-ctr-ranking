# Causal Impact V4: two-minute interview brief

## One-sentence summary

I built a market-level quasi-experimental workflow that estimates the incremental CTR effect
of a non-random ranking rollout with exposure-weighted Difference-in-Differences, tests the
identification story, translates uncertainty into clicks, and publishes an auditable evidence
bundle atomically.

## Business context

The product question is whether activating a candidate-ranking policy in selected markets
increased CTR enough to clear a predeclared +0.2 percentage-point practical threshold. The
unit is a market-week, the outcome is clicks divided by candidate exposures, and rollout starts
at relative week 0.

## Why not just use an A/B test?

Randomization is preferable and the repository includes a separate user-randomized RCT
workflow. V4 covers the common case where historical rollout decisions were operational rather
than randomized, but treated and untreated markets and a long pre-period remain available. DiD
can then be informative under stronger assumptions; it does not turn observational assignment
into randomization.

## Estimator

The primary model is candidate-exposure-weighted least squares with market fixed effects, week
fixed effects, and `policy_active = treated_market × post_rollout`. Market effects absorb
stable differences between markets; week effects absorb shocks shared across markets. Exposure
weights target the average candidate exposure. Standard errors are clustered by market because
repeated weekly outcomes within one market are not independent.

## Assumptions and diagnostics

The central assumption is parallel untreated potential-outcome trends. I make it inspectable in
three ways:

- an event study estimates weeks -12 through 8 relative to omitted week -1;
- a joint Wald test evaluates all 11 pre-treatment leads;
- a fake rollout at week -8 is estimated using only genuine pre-treatment rows.

The pre-period balance table is market-level. Large level differences are visible by design,
which is why simple post-period comparisons would be biased. Fixed effects can absorb stable
levels, but neither they nor passing diagnostics eliminate unobserved time-varying confounding,
anticipation, or spillovers.

## Committed synthetic result

The exposure-weighted DiD estimate is **+0.6123 percentage points**, with a market-clustered
95% confidence interval of **[+0.5589, +0.6658]**. The joint pre-trend test has **p=0.6314**,
and the placebo is **-0.0115 percentage points** with interval **[-0.0768, +0.0538]**. Both
diagnostics pass, so the predeclared policy returns `supports_incremental_impact`.

At one million candidate exposures, that corresponds to approximately **6,123 incremental
clicks**, with interval **[5,589, 6,658]**. I deliberately stop there: no revenue or lifetime
value is assumed.

These values come from deterministic `synthetic-quasi-experiment` data. They test the code and
decision policy, not production lift or external validity.

## Engineering controls

The CLI validates the balanced panel, treatment timing, CTR identity, minimum clusters, metadata
tier, row counts, and SHA-256 hashes. It copies the exact configuration, keeps full precision in
CSV outputs, creates a deterministic event-study plot, and moves the staging directory into
place only after all ten declared artifacts exist. Existing evidence is never overwritten.

## Next real-data step

Before applying this to a real rollout, I would document why markets were selected, inventory
concurrent campaigns and product changes, check anticipation and spillovers, predeclare
exclusions and sensitivity specifications, and seek a randomized holdout or phased randomized
rollout. For staggered adoption, I would replace the common-timing estimator with an approach
that explicitly handles heterogeneous cohort and time effects.

## Useful links

- [Causal report](../artifacts/causal-impact-v4/causal_report.md)
- [Event-study estimates](../artifacts/causal-impact-v4/event_study.csv)
- [Diagnostics](../artifacts/causal-impact-v4/diagnostics.json)
- [Analysis contract](../configs/causal-impact-v4.json)
- [Full experiment protocol](experiment-protocol.md)
