from pathlib import Path


def test_ci_checks_png_semantics_without_cross_platform_byte_claim() -> None:
    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert 'if committed.suffix == ".png":' in workflow
    assert "expected.size == actual.size" in workflow
    assert 'expected.info["Description"] == actual.info["Description"]' in workflow
    assert "expected.getbbox() is not None" in workflow
    assert "actual.getbbox() is not None" in workflow
    assert "committed.read_bytes() == regenerated.read_bytes()" in workflow


def test_ci_has_a_complete_causal_extra_release_gate() -> None:
    """Catches V4 evidence merging without exercising its optional statistical stack."""

    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    required = (
        "causal-extra:",
        'python -m pip install -e ".[dev,causal]"',
        "tests/test_causal.py",
        "tests/test_causal_reporting.py",
        "tests/test_causal_visualization.py",
        "tests/test_causal_workflow.py",
        "tests/test_cli.py",
        "news-ctr make-quasi-experiment",
        "data/ci-market-panel.parquet",
        "news-ctr causal-impact",
        "artifacts/causal-ci",
        "CAUSAL_ARTIFACTS",
        "artifacts/causal-impact-v4",
        'committed.suffix == ".png"',
        "committed.read_bytes() == regenerated.read_bytes()",
        "synthetic-quasi-experiment",
        "not production lift",
    )
    assert all(item in workflow for item in required)
