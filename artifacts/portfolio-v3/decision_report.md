# Experiment Decision Report

> Synthetic randomized teaching evidence. It validates the experiment-analysis workflow; it is not production lift.

- Evidence tier: `synthetic-rct`

## Provenance

- Input SHA-256: `441561c3c8ae22c17b6c6ca0f025235bf34ede419ff708484e4978d227e775a5`
- Metadata SHA-256: `e5e304183f0094c75d24d89da400b9c23651dbfec36686f8c10bc1dc78a083d5`
- Config SHA-256: `43803defb446aac85c2dd7a4957a66f68b0029bf1a0892f91ede34cd4045251d`

## Recommendation

**launch** — Launch under the declared policy and monitor guardrails.

## Experiment validity

- Randomization unit: `user_id`
- SRM status: **pass**
- SRM p-value: 0.380589
- Observed allocation: `{'control': 10062, 'treatment': 9938}`
- Expected allocation: `{'control': 10000.0, 'treatment': 10000.0}`

## Primary metric

- Raw ITT (95% CI): 0.0202 [0.0109, 0.0296]
- CUPED-adjusted ITT used for the decision (95% CI): 0.0205 [0.0112, 0.0298]
- Minimum practically important effect: 0.0080

## Guardrails

| Metric | Direction | Margin | Effect and 95% CI | State |
| --- | --- | ---: | ---: | --- |
| `dwell_seconds` | non_decrease | -2.000 | 0.4125 [-0.1429, 0.9678] | **pass** |
| `latency_ms` | non_increase | 8.000 | 2.9345 [2.5999, 3.2692] | **pass** |

## Heterogeneous effects

All undeclared segment cuts are exploratory. 6 cells were estimated;
0 are marked low support at the threshold of 200
randomization units.

## Predeclared decision rule

1. SRM failure makes the experiment invalid.
2. Launch requires the CUPED primary lower confidence bound to exceed
   0.0080.
3. A credibly harmful primary effect or failed guardrail means do not launch.
4. Any interval crossing a decision margin means continue the experiment.

## Reproduce

```bash
news-ctr make-experiment --output data/synthetic-rct.parquet --seed 42 --users 20000
news-ctr experiment --input data/synthetic-rct.parquet \
  --output artifacts/experiment-v3 --config configs/experiment-v3.json
```

## Limitations

- This workflow estimates user-level intent-to-treat effects for a randomized fixture.
- Segment effects are exploratory unless explicitly predeclared as confirmatory.
- Product, editorial, novelty, diversity, and longer-term effects remain out of scope.
