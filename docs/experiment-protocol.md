# Experiment protocol

## Research question

Can user-history and article-content features improve the within-impression ranking of clicked news beyond simple context and freshness signals?

## Primary comparison

1. Logistic regression on the complete leakage-safe feature set.
2. LightGBM LambdaRank on the same feature set and identical temporal split.

A meaningful real-data study should also add context-only and non-personalized baselines. The repository's committed result is only the deterministic synthetic engineering check.

## Split policy

Use EB-NeRD's supplied train and validation bundles when both are present. The maximum training impression timestamp must be strictly earlier than the minimum validation timestamp.

If only one behavior bundle exists, split ordered unique timestamps. Every impression at the cutoff timestamp remains in one fold. Never split random candidate rows.

## Fitting policy

- Fit text vocabulary, SVD, category encoders, and imputation values on training-visible data.
- Derive user interests from click histories supplied before the evaluated behavior period.
- Calculate article age from `impression_time - published_time`.
- Do not include raw user, article, or impression IDs.
- Do not use article aggregate fields whose windows extend past the impression time.

## Metrics

Report group AUC, MRR, NDCG@5, NDCG@10, and the number of AUC groups skipped for containing one class. Report log loss only for models that return calibrated click probabilities.

Metrics are calculated independently per impression, then averaged. Preserve a prediction table with group IDs so the calculation can be independently reproduced.

## Minimum ablation matrix for EB-NeRD

| Experiment | Context | Article metadata | History affinity | Text similarity |
| --- | :---: | :---: | :---: | :---: |
| Context baseline | ✓ |  |  |  |
| Metadata | ✓ | ✓ |  |  |
| Personalized | ✓ | ✓ | ✓ |  |
| Full | ✓ | ✓ | ✓ | ✓ |

Use one split and seed for initial debugging. For a final report, repeat at least three seeds, show the mean and standard deviation, and test whether improvements are consistent across device, subscription, history-length, and article-age segments.

## Interpretation rules

- An offline gain does not prove an online CTR gain.
- Position can be predictive because the previous system chose the position; it is a bias signal, not necessarily a useful causal feature.
- AUC measures pair ordering, while MRR and NDCG emphasize top positions. Discuss disagreements rather than selecting only the most flattering metric.
- Synthetic results validate the pipeline only.

## Reproducibility record

Every run stores the input file fingerprint, source split, seed, model, feature list, row counts, metrics, predictions, importance values, and a model card. Retain the full run directory for real EB-NeRD experiments, subject to its license and organizational policy.
