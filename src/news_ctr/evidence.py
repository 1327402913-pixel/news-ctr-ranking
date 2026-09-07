"""Evidence tiers and truthful claim boundaries for portfolio artifacts."""

from __future__ import annotations

from enum import Enum


class EvidenceTier(str, Enum):
    """How strongly an artifact can support a data-science claim."""

    SYNTHETIC_RCT = "synthetic-rct"
    SYNTHETIC_QUASI_EXPERIMENT = "synthetic-quasi-experiment"
    SYNTHETIC_OBSERVATIONAL = "synthetic-observational"
    LICENSED_EBNERD = "licensed-ebnerd"


def classify_dataset_evidence(source_name: str) -> EvidenceTier:
    """Map a declared dataset source to its evidence boundary."""

    if source_name == EvidenceTier.SYNTHETIC_RCT.value:
        return EvidenceTier.SYNTHETIC_RCT
    if source_name == EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT.value:
        return EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT
    if source_name == EvidenceTier.SYNTHETIC_OBSERVATIONAL.value or source_name.startswith(
        "synthetic:"
    ):
        return EvidenceTier.SYNTHETIC_OBSERVATIONAL
    if source_name == EvidenceTier.LICENSED_EBNERD.value or source_name.startswith("ebnerd:"):
        return EvidenceTier.LICENSED_EBNERD
    raise ValueError(f"unknown evidence source: {source_name}")


def evidence_banner(tier: EvidenceTier) -> str:
    """Return the mandatory human-readable limitation for an evidence tier."""

    banners = {
        EvidenceTier.SYNTHETIC_RCT: (
            "Synthetic randomized teaching evidence. It validates the experiment-analysis "
            "workflow; it is not production lift."
        ),
        EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT: (
            "Synthetic quasi-experimental teaching evidence. It validates the "
            "identification, diagnostic, and reporting workflow; it is not production lift."
        ),
        EvidenceTier.SYNTHETIC_OBSERVATIONAL: (
            "Synthetic observational engineering evidence. It validates the ranking and "
            "analytics workflow; it is not production lift or real-world model quality."
        ),
        EvidenceTier.LICENSED_EBNERD: (
            "Licensed EB-NeRD aggregate observational evidence. It is not an online causal "
            "estimate or production lift."
        ),
    }
    return banners[tier]
