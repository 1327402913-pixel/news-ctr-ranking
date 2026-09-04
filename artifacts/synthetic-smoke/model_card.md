# Model Card

## Model

- Estimator: `logistic`
- Dataset split: `synthetic:validation`
- Random seed: `42`
- Dataset fingerprint: `57da7f4641b0c12e146c3d9d7c4cbb02bf12fa42bb7911835631935440908bca`

## Intended use

Rank candidate news articles within an observed impression for offline research and portfolio demonstration. The model is not intended for production editorial decisions.

## Evaluation status

This is an engineering smoke test on deterministic synthetic data. It is not an EB-NeRD benchmark and must not be used to claim real-world quality.

| Metric | Value |
| --- | ---: |
| `group_auc` | 0.761111 |
| `mrr` | 0.646759 |
| `ndcg@5` | 0.714620 |
| `ndcg@10` | 0.734409 |
| `log_loss` | 0.641807 |

## Important limitations

- Offline clicks reflect exposure and position bias, not pure user preference.
- No causal claim can be made from these observational logs.
- The baseline does not optimize diversity, novelty, fairness, or editorial values.
- User-history representations are fitted from the supplied training history only.

## Reproduce

```bash
news-ctr train --data data/synthetic --output artifacts/synthetic-smoke --model logistic --seed 42 --text-components 16
```
