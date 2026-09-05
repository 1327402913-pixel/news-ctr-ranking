"""Deterministic randomized fixtures for teaching experiment analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_ctr.evidence import EvidenceTier


def write_synthetic_experiment(
    path: Path,
    *,
    seed: int,
    users: int,
    corrupt_allocation: bool = False,
) -> Path:
    """Write one deterministic user-level randomized experiment row per unit."""

    if users < 2:
        raise ValueError("users must be at least two")
    output = Path(path)
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists() or metadata_path.exists():
        raise ValueError(f"experiment output already exists: {output}")

    rng = np.random.default_rng(seed)
    treatment_probability = 0.80 if corrupt_allocation else 0.50
    is_treatment = rng.random(users) < treatment_probability
    variant = np.where(is_treatment, "treatment", "control")

    latent_engagement = rng.normal(0.0, 1.0, users)
    pre_ctr = np.clip(0.12 + 0.045 * latent_engagement + rng.normal(0, 0.015, users), 0.01, 0.45)
    click_probability = np.clip(pre_ctr + 0.025 * is_treatment, 0.001, 0.95)
    click = rng.binomial(1, click_probability)
    dwell_seconds = np.maximum(
        1.0,
        48.0 + 18.0 * latent_engagement + 0.3 * is_treatment + rng.normal(0, 9, users),
    )
    latency_ms = np.maximum(1.0, 120.0 + 3.0 * is_treatment + rng.normal(0, 12, users))

    frame = pd.DataFrame(
        {
            "user_id": np.arange(1, users + 1, dtype=np.int64),
            "variant": variant,
            "pre_ctr": np.round(pre_ctr, 6),
            "click": click.astype(np.int8),
            "dwell_seconds": np.round(dwell_seconds, 3),
            "latency_ms": np.round(latency_ms, 3),
            "device_type": rng.choice(["desktop", "mobile", "tablet"], users, p=[0.3, 0.6, 0.1]),
            "history_segment": rng.choice(["new", "light", "heavy"], users, p=[0.2, 0.5, 0.3]),
        }
    )
    allocation = frame["variant"].value_counts().sort_index().to_dict()
    metadata = {
        "allocation": {label: int(count) for label, count in allocation.items()},
        "declared_effects": {
            "click_absolute": 0.025,
            "dwell_seconds": 0.3,
            "latency_ms": 3.0,
        },
        "evidence_tier": EvidenceTier.SYNTHETIC_RCT.value,
        "randomization_unit": "user_id",
        "seed": seed,
        "target_treatment_probability": treatment_probability,
        "units": users,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
