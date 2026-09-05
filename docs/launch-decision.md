# Launch decision memo

## Decision

**Launch under the predeclared teaching policy.** This recommendation applies only to
the deterministic 20,000-user synthetic randomized fixture. It demonstrates how the
analysis converts experiment evidence into a decision; it is **not production lift**
and does not authorize a real deployment.

## Product question

Would a candidate ranking treatment improve user click probability by a practically
important amount without materially reducing dwell time or increasing serving
latency?

## Predeclared rule

- Reject interpretation if the 50/50 assignment fails the SRM check at alpha 0.05.
- Require the CUPED-adjusted click-effect lower 95% confidence bound to exceed +0.8
  percentage points.
- Require the dwell-time effect lower bound to remain above -2 seconds.
- Require the latency effect upper bound to remain below +8 milliseconds.
- Continue rather than launch when an interval crosses a decision margin.

## Evidence

| Decision input | Estimate | Result |
| --- | ---: | --- |
| Assignment balance | SRM p = 0.3806 | Pass |
| Raw click ITT | +2.02 pp [1.09, 2.96] | Positive |
| CUPED click ITT | +2.05 pp [1.12, 2.98] | Clears +0.8 pp threshold |
| Dwell time | +0.41 s [-0.14, 0.97] | Passes -2 s margin |
| Latency | +2.93 ms [2.60, 3.27] | Passes +8 ms margin |

CUPED uses only the pre-treatment `pre_ctr` covariate and reduces the estimated
primary-effect variance by 1.79%. The six device/history cells are exploratory; none
falls below the declared 200-user support threshold.

## Risks and follow-up

- Synthetic behavior cannot establish external validity, novelty, diversity,
  editorial quality, or longer-term retention.
- A real launch needs production instrumentation, eligibility checks, exposure logs,
  monitoring, and an independently reviewed experiment configuration.
- Before applying the policy to users, repeat the offline ranking benchmark on
  licensed EB-NeRD data and run an actual randomized product experiment.

The machine-readable evidence is in [`artifacts/portfolio-v3`](../artifacts/portfolio-v3),
and the full generated report is
[`decision_report.md`](../artifacts/portfolio-v3/decision_report.md).
