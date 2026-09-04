import pytest

from news_ctr.evidence import EvidenceTier, classify_dataset_evidence, evidence_banner


def test_dataset_evidence_is_explicit_and_truthful() -> None:
    """Catches synthetic or licensed datasets being assigned the wrong claim boundary."""

    assert classify_dataset_evidence("synthetic:validation") is EvidenceTier.SYNTHETIC_OBSERVATIONAL
    assert classify_dataset_evidence("ebnerd:validation") is EvidenceTier.LICENSED_EBNERD
    assert "not production lift" in evidence_banner(EvidenceTier.SYNTHETIC_RCT).lower()


def test_unknown_dataset_source_is_rejected() -> None:
    """Catches silently treating an unknown dataset as credible portfolio evidence."""

    with pytest.raises(ValueError, match="unknown evidence source"):
        classify_dataset_evidence("mystery:validation")
