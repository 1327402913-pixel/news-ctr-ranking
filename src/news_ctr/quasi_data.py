"""Deterministic market panels for teaching quasi-experimental analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_ctr.evidence import EvidenceTier


def write_synthetic_market_panel(
    path: Path,
    *,
    seed: int,
    markets: int,
    pre_weeks: int,
    post_weeks: int,
) -> Path:
    """Write a deterministic common-rollout market panel and truthful metadata."""

    if markets < 4:
        raise ValueError("markets must be at least four")
    if markets % 2:
        raise ValueError("markets must be even for a balanced treatment split")
    if pre_weeks < 1:
        raise ValueError("pre_weeks must be positive")
    if post_weeks < 1:
        raise ValueError("post_weeks must be positive")

    output = Path(path)
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise ValueError(f"market panel output already exists: {output}")

    rng = np.random.default_rng(seed)
    market_size = rng.lognormal(mean=0.0, sigma=0.28, size=markets)
    standardized_size = (market_size - market_size.mean()) / market_size.std(ddof=0)
    market_effect = 0.010 * standardized_size + rng.normal(0.0, 0.006, size=markets)
    treatment_order = np.argsort(market_size, kind="stable")
    treated = np.zeros(markets, dtype=np.int8)
    treated[treatment_order[markets // 2 :]] = 1

    rows: list[dict[str, object]] = []
    for market_index in range(markets):
        for relative_week in range(-pre_weeks, post_weeks):
            active = int(treated[market_index] and relative_week >= 0)
            exposure_mean = (
                14_000
                * market_size[market_index]
                * (1 + 0.08 * np.cos(2 * np.pi * relative_week / 13))
            )
            candidate_exposures = max(500, round(exposure_mean + rng.normal(0, 500)))
            click_probability = float(
                np.clip(
                    0.12
                    + market_effect[market_index]
                    + 0.006 * np.sin(2 * np.pi * relative_week / 13)
                    + 0.00015 * relative_week
                    + 0.006 * active,
                    0.02,
                    0.40,
                )
            )
            clicks = int(rng.binomial(candidate_exposures, click_probability))
            rows.append(
                {
                    "market_id": f"market_{market_index + 1:03d}",
                    "relative_week": relative_week,
                    "treated_market": int(treated[market_index]),
                    "policy_active": active,
                    "candidate_exposures": candidate_exposures,
                    "clicks": clicks,
                    "ctr": clicks / candidate_exposures,
                    "market_size_index": float(market_size[market_index]),
                }
            )

    frame = pd.DataFrame(rows).astype(
        {
            "relative_week": "int16",
            "treated_market": "int8",
            "policy_active": "int8",
            "candidate_exposures": "int64",
            "clicks": "int64",
        }
    )
    frame = frame.sort_values(["market_id", "relative_week"], kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(output, index=False)
        metadata = {
            "evidence_tier": EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT.value,
            "known_effects": {"ctr_absolute": 0.006},
            "markets": markets,
            "panel_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "post_weeks": post_weeks,
            "pre_weeks": pre_weeks,
            "rows": len(frame),
            "seed": seed,
            "treated_markets": int(treated.sum()),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output
