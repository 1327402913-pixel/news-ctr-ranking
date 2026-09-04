from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news_ctr.cli import main


def test_cli_generates_audits_and_trains_reproducibly(tmp_path: Path, capsys) -> None:
    """Catches a workflow that works only through private notebook state."""
    data_path = tmp_path / "synthetic"
    first_output = tmp_path / "run-one"
    second_output = tmp_path / "run-two"

    assert (
        main(
            [
                "make-synthetic",
                "--output",
                str(data_path),
                "--seed",
                "23",
                "--users",
                "12",
                "--articles",
                "30",
                "--impressions",
                "60",
            ]
        )
        == 0
    )
    assert main(["audit", "--data", str(data_path), "--split", "train"]) == 0
    audit_output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert audit_output["source_name"] == "synthetic:train"
    assert audit_output["impressions"] == 48

    common = [
        "train",
        "--data",
        str(data_path),
        "--model",
        "logistic",
        "--seed",
        "23",
        "--text-components",
        "4",
    ]
    assert main([*common, "--output", str(first_output)]) == 0
    capsys.readouterr()
    assert main([*common, "--output", str(second_output)]) == 0

    expected_artifacts = {
        "metrics.json",
        "predictions.parquet",
        "feature_importance.csv",
        "run_config.json",
        "model_card.md",
    }
    assert {item.name for item in first_output.iterdir()} == expected_artifacts
    metrics = json.loads((first_output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"] == "synthetic:validation"
    assert metrics["model"] == "logistic"
    for name in ("group_auc", "mrr", "ndcg@5", "ndcg@10", "log_loss"):
        assert np.isfinite(metrics["metrics"][name])
    assert 0 <= metrics["metrics"]["mrr"] <= 1

    first_predictions = pd.read_parquet(first_output / "predictions.parquet")
    second_predictions = pd.read_parquet(second_output / "predictions.parquet")
    assert first_predictions.columns.tolist() == [
        "impression_id",
        "article_id",
        "label",
        "score",
    ]
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    model_card = (first_output / "model_card.md").read_text(encoding="utf-8")
    assert "engineering smoke test" in model_card.lower()
    assert "not an EB-NeRD benchmark" in model_card


def test_cli_returns_nonzero_with_an_actionable_data_error(tmp_path: Path, capsys) -> None:
    """Catches raw file tracebacks for a normal user input mistake."""
    missing = tmp_path / "does-not-exist"

    code = main(["audit", "--data", str(missing)])

    assert code == 2
    assert "dataset files not found" in capsys.readouterr().err
