# Ranking Benchmark Report

> **Evaluation status:** Synthetic engineering evidence — this is not an EB-NeRD benchmark and does not measure real-world recommendation lift.

## 60-second summary

- Dataset split: `synthetic:validation`.
- Every model is evaluated on the same validation impressions.
- Popularity statistics and feature fitting use training-visible rows only.
- Uncertainty uses `200` paired, impression-level bootstrap samples.
- This is offline observational evidence; it does not establish causal product impact.

## Leaderboard

| Rank | Model | Group AUC | MRR | NDCG@5 | NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| 1 | logistic | 0.772222 | 0.653704 | 0.720183 | 0.739972 |
| 2 | lightgbm | 0.750000 | 0.609259 | 0.697030 | 0.706924 |
| 3 | position | 0.616667 | 0.487037 | 0.553516 | 0.612884 |
| 4 | popularity | 0.555556 | 0.447222 | 0.541418 | 0.580996 |

## 95% confidence intervals

Whole impressions are resampled, and identical draws are reused for every model.

| Model | Metric | Mean | 95% lower | 95% upper | Valid samples |
| --- | --- | --- | --- | --- | --- |
| lightgbm | group_auc | 0.746222 | 0.661111 | 0.833333 | 200/200 |
| lightgbm | mrr | 0.605581 | 0.510104 | 0.704699 | 200/200 |
| lightgbm | ndcg@10 | 0.704104 | 0.632502 | 0.779895 | 200/200 |
| lightgbm | ndcg@5 | 0.693665 | 0.619325 | 0.774485 | 200/200 |
| logistic | group_auc | 0.768056 | 0.666667 | 0.850139 | 200/200 |
| logistic | mrr | 0.648391 | 0.538866 | 0.745428 | 200/200 |
| logistic | ndcg@10 | 0.735938 | 0.651928 | 0.809344 | 200/200 |
| logistic | ndcg@5 | 0.716198 | 0.612114 | 0.800140 | 200/200 |
| popularity | group_auc | 0.548389 | 0.438750 | 0.647292 | 200/200 |
| popularity | mrr | 0.441447 | 0.351377 | 0.539502 | 200/200 |
| popularity | ndcg@10 | 0.576529 | 0.508896 | 0.651491 | 200/200 |
| popularity | ndcg@5 | 0.534081 | 0.440179 | 0.615561 | 200/200 |
| position | group_auc | 0.619222 | 0.522222 | 0.705556 | 200/200 |
| position | mrr | 0.486160 | 0.401366 | 0.576412 | 200/200 |
| position | ndcg@10 | 0.612376 | 0.547920 | 0.679103 | 200/200 |
| position | ndcg@5 | 0.553700 | 0.459674 | 0.629527 | 200/200 |

## Feature ablations

Deltas compare each deterministic logistic-regression variant with the full feature set.

| Ablation | Features | Group AUC | Δ vs full | MRR | Δ vs full | NDCG@5 | Δ vs full | NDCG@10 | Δ vs full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context_only | 9 | 0.616667 | -0.155556 | 0.487037 | -0.166667 | 0.553516 | -0.166667 | 0.612884 | -0.127088 |
| no_position | 20 | 0.633333 | -0.138889 | 0.568981 | -0.084722 | 0.633147 | -0.087036 | 0.672726 | -0.067246 |
| no_semantic | 19 | 0.772222 | 0.000000 | 0.653704 | 0.000000 | 0.720183 | 0.000000 | 0.739972 | 0.000000 |
| no_personalization | 19 | 0.772222 | 0.000000 | 0.653704 | 0.000000 | 0.720183 | 0.000000 | 0.739972 | 0.000000 |
| full | 21 | 0.772222 | 0.000000 | 0.653704 | 0.000000 | 0.720183 | 0.000000 | 0.739972 | 0.000000 |

## Segment checks

Analysis model: `logistic`.

Slices marked `LOW` contain fewer than five impressions and should not drive conclusions.

| Segment | Value | Impressions | Support | Group AUC | MRR | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| device_type | 1 | 12 | OK | 0.816667 | 0.625000 | 0.721221 |
| device_type | 2 | 12 | OK | 0.833333 | 0.736111 | 0.801934 |
| device_type | 3 | 12 | OK | 0.666667 | 0.600000 | 0.696761 |
| candidate_count | medium | 36 | OK | 0.772222 | 0.653704 | 0.739972 |
| history_length | short | 36 | OK | 0.772222 | 0.653704 | 0.739972 |

## Exposure-bias diagnostic

These are candidate-level empirical click rates, not ranking metrics or causal effects.
Position differences are evidence of exposure bias in logged feedback.

| Diagnostic | Bucket | Candidates | Clicks | Click rate |
| --- | --- | --- | --- | --- |
| candidate_position | 0 | 36 | 8 | 0.222222 |
| candidate_position | 1 | 36 | 9 | 0.250000 |
| candidate_position | 2 | 36 | 10 | 0.277778 |
| candidate_position | 3 | 36 | 2 | 0.055556 |
| candidate_position | 4 | 36 | 1 | 0.027778 |
| candidate_position | 5 | 36 | 6 | 0.166667 |
| article_freshness | 7d+ | 216 | 36 | 0.166667 |

## Runtime and artifact size

Timing is environment-dependent and is included for operational context only.

| Model | Fit seconds | Score ms / 1k | Serialized bytes |
| --- | --- | --- | --- |
| position | 0.000106 | 0.103778 | 0 |
| popularity | 0.000451 | 0.525847 | 0 |
| logistic | 0.007154 | 2.252699 | 254880 |
| lightgbm | 0.870388 | 5.472995 | 954384 |

## Leakage safeguards

- Temporal ordering keeps validation impressions strictly after training impressions.
- Candidate lists remain intact for ranking metrics and bootstrap resampling.
- Smoothed article popularity is estimated from training labels only.
- Text vocabulary, imputers, encoders, and user representations are fitted
  without validation clicks.

## Limitations

- Logged clicks mix user preference with exposure and position effects.
- Synthetic runs validate engineering behavior, not recommender quality on EB-NeRD.
- Offline metrics do not measure diversity, novelty, fairness, editorial quality, or online lift.
- Segment differences are descriptive and must not be interpreted as causal.
- Joblib artifacts use pickle semantics; load only artifacts you created or trust.

## Reproduce exactly

```bash
news-ctr make-synthetic --output data/portfolio-v2 --seed 42 --users 30 --articles 80 --impressions 180

news-ctr benchmark \
  --data data/portfolio-v2 \
  --output artifacts/portfolio-v2-reproduced \
  --models position,popularity,logistic,lightgbm \
  --bootstrap-samples 200 \
  --text-components 32 \
  --valid-fraction 0.2 \
  --seed 42
```

The reproduction destination must not already exist. The adjacent CSV files contain
the complete machine-readable evidence.
