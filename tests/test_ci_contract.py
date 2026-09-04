from pathlib import Path


def test_ci_compares_deterministic_png_artifact_bytes() -> None:
    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert 'committed.suffix != ".png"' not in workflow
    assert "committed.read_bytes() == regenerated.read_bytes()" in workflow
