from pathlib import Path


def test_ci_checks_png_semantics_without_cross_platform_byte_claim() -> None:
    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert 'if committed.suffix == ".png":' in workflow
    assert "expected.size == actual.size" in workflow
    assert 'expected.info["Description"] == actual.info["Description"]' in workflow
    assert "expected.getbbox() is not None" in workflow
    assert "actual.getbbox() is not None" in workflow
    assert "committed.read_bytes() == regenerated.read_bytes()" in workflow
