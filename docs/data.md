# Data contract

## License boundary

EB-NeRD is downloaded directly from <https://recsys.eb.dk/> under the terms shown there. This project never downloads on a user's behalf, redistributes, or commits the raw dataset. `data/` is ignored by Git.

## Bundle layout

`articles.parquet` lives at the bundle root. Each split has its own `behaviors.parquet` and `history.parquet`. The loader accepts `train` and `validation` split names.

## Required canonical columns

### Articles

| Column | Meaning |
| --- | --- |
| `article_id` | Unique article identifier |
| `title` | Article headline |
| `subtitle` | Article subtitle or abstract |
| `published_time` | Publication timestamp |
| `category` | Category ID or label |
| `premium` | Paywall flag |

Optional model inputs include `article_type` and `sentiment_score`.

### Behaviors

| Column | Meaning |
| --- | --- |
| `impression_id` | Unique ranking group |
| `user_id` | Anonymized user identifier |
| `impression_time` | Exposure timestamp |
| `article_ids_inview` | Candidate article list |
| `article_ids_clicked` | Clicked subset of the candidate list |

Optional context includes `device_type`, `is_sso_user`, and `is_subscriber`.

### History

| Column | Meaning |
| --- | --- |
| `user_id` | Anonymized user identifier |
| `article_id_fixed` | Articles clicked before the behavior split |

`impression_time_fixed` is optional and identifies the most recent historical click. The loader also accepts the official-style aliases `article_ids` and `timestamps`.

## Audit invariants

`news-ctr audit` fails when:

- a required file or column is absent;
- impression or article IDs are duplicated;
- timestamps are invalid;
- an impression has no candidates or repeats a candidate;
- a clicked article was not in view;
- an in-view article is absent from the article catalog;
- a supposed list column contains a scalar.

The audit prints counts and the observed time range as JSON so it can be captured in an experiment log.

## Candidate expansion

Each behavior row becomes one row per in-view article. Labels are assigned only from `article_ids_clicked` in that same behavior row. `candidate_position` is zero-based and `candidate_count` is retained for every expanded row.

This transformation deliberately does not sample negatives from other impressions.

## Synthetic data

`news-ctr make-synthetic` generates the same essential tables. Every impression contains one click, and click propensity depends on the user's preferred category, article freshness, content similarity, and candidate position. It is designed for fast integration tests, not for estimating real-world model quality.
