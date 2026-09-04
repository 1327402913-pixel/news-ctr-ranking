# Product Analytics Report

> Synthetic observational engineering evidence. It validates the ranking and analytics workflow; it is not production lift or real-world model quality.

## Scope

- Dataset split: `synthetic:validation`
- Dataset fingerprint: `b3f4dedb235d441bb470464f2ce054a565ff1e777c29832127ec46fa5347b79c`
- Impressions: 36
- Candidate exposures: 216
- Clicks: 36
- Candidate CTR: 0.1667

## Denominator contract

CTR uses candidate exposures as its denominator. `units` in segment outputs is the
number of distinct impressions represented by the cell; `candidates` is the exposure
denominator and `clicks` is the numerator. Cells with fewer than
5 impression units are marked `low_support` and should not drive
a product decision. This run contains 0 such cells.

## Deliverables

- `data_quality.csv`: contract checks and anomaly counts.
- `kpi_summary.csv`: top-line traffic and engagement measures.
- `exposure_funnel.csv`: users → impressions → candidates → clicks.
- `segment_kpis.csv`: device, position, slate-size, history, and freshness cuts.
- `user_cohorts.csv`: weekly retention when at least two activity weeks exist.

## Interpretation boundary

These tables describe observed exposure and click logs. They do not identify a causal
treatment effect; use the randomized-experiment workflow for launch decisions.
