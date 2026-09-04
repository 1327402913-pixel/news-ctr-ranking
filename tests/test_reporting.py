from __future__ import annotations

import os
from pathlib import Path

from news_ctr.reporting import dataset_fingerprint


def test_dataset_fingerprint_is_content_stable(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    marker = dataset / "dataset.json"
    marker.write_text('{"source":"synthetic"}\n', encoding="utf-8")

    original = dataset_fingerprint(dataset)
    os.utime(marker, (marker.stat().st_atime + 100, marker.stat().st_mtime + 100))

    assert dataset_fingerprint(dataset) == original

    marker.write_text('{"source":"different"}\n', encoding="utf-8")
    assert dataset_fingerprint(dataset) != original


def test_dataset_fingerprint_detects_changes_in_large_file_middle(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    large = dataset / "articles.parquet"
    large.write_bytes(b"a" * (3 * 1024 * 1024))
    original = dataset_fingerprint(dataset)

    with large.open("r+b") as stream:
        stream.seek(1536 * 1024)
        stream.write(b"changed")

    assert dataset_fingerprint(dataset) != original
