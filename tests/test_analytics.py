from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

import news_ctr.analytics as analytics
from news_ctr.analytics import AnalyticsConfig, run_analytics
from news_ctr.data import write_synthetic_bundle


def _synthetic_data(tmp_path: Path, *, seed: int = 7) -> Path:
    return write_synthetic_bundle(
        tmp_path / "data",
        seed=seed,
        n_users=8,
        n_articles=20,
        n_impressions=40,
    )


def test_run_analytics_computes_kpis_and_segments(tmp_path: Path) -> None:
    data = _synthetic_data(tmp_path)
    output = run_analytics(
        AnalyticsConfig(data=data, output=tmp_path / "analytics", min_cell_count=2)
    )

    assert {item.name for item in output.iterdir()} == {
        "analytics_report.md",
        "data_quality.csv",
        "exposure_funnel.csv",
        "kpi_summary.csv",
        "segment_kpis.csv",
        "user_cohorts.csv",
    }
    summary = pd.read_csv(output / "kpi_summary.csv").iloc[0]
    assert summary["impressions"] == 8
    assert summary["candidates"] == 48
    assert summary["clicks"] == 8
    assert summary["ctr"] == pytest.approx(1 / 6)
    assert summary["active_users"] == 8

    quality = pd.read_csv(output / "data_quality.csv").set_index("metric")
    assert {
        "candidate_rows",
        "duplicate_impression_article",
        "invalid_label_rows",
        "null_user_id_rate",
        "min_impression_time",
        "max_impression_time",
    } <= set(quality.index)
    assert quality.at["candidate_rows", "numeric_value"] == 48
    assert quality.at["invalid_label_rows", "numeric_value"] == 0
    assert quality.at["null_user_id_rate", "numeric_value"] == 0
    assert (
        quality.at["min_impression_time", "text_value"]
        <= quality.at["max_impression_time", "text_value"]
    )

    segments = pd.read_csv(output / "segment_kpis.csv")
    assert set(segments["segment"]) >= {
        "device_type",
        "candidate_position",
        "candidate_count",
        "history_length",
        "freshness",
    }
    assert (
        segments.sort_values(["segment", "value"], kind="stable")
        .reset_index(drop=True)
        .equals(segments)
    )

    funnel = pd.read_csv(output / "exposure_funnel.csv")
    assert funnel["stage"].tolist() == ["active_users", "impressions", "candidates", "clicks"]
    report = (output / "analytics_report.md").read_text(encoding="utf-8")
    assert "not production lift" in report
    assert "Denominator" in report


def test_small_analytics_cells_are_flagged(tmp_path: Path) -> None:
    data = _synthetic_data(tmp_path, seed=8)
    output = run_analytics(
        AnalyticsConfig(data=data, output=tmp_path / "analytics", min_cell_count=10)
    )

    segments = pd.read_csv(output / "segment_kpis.csv")
    assert segments.loc[segments["units"] < 10, "low_support"].all()
    assert not segments.loc[segments["units"] >= 10, "low_support"].any()


def test_analytics_sql_is_packaged_and_single_week_cohort_is_typed(tmp_path: Path) -> None:
    for sql_name in analytics.SQL_OUTPUTS:
        sql_resource = files("news_ctr").joinpath("sql", sql_name)
        assert sql_resource.is_file()
        assert sql_resource.read_text(encoding="utf-8").strip()

    output = run_analytics(
        AnalyticsConfig(data=_synthetic_data(tmp_path), output=tmp_path / "analytics")
    )
    cohorts = pd.read_csv(output / "user_cohorts.csv")
    assert cohorts.empty
    assert cohorts.columns.tolist() == [
        "cohort_week",
        "week_offset",
        "cohort_users",
        "active_users",
        "retention_rate",
    ]


def test_analytics_validates_config_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _synthetic_data(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run_analytics(AnalyticsConfig(data=data, output=existing))
    with pytest.raises(ValueError, match="min_cell_count"):
        run_analytics(AnalyticsConfig(data=data, output=tmp_path / "invalid", min_cell_count=0))

    output = tmp_path / "failed"

    def fail_query(*args, **kwargs):
        raise RuntimeError("injected SQL failure")

    monkeypatch.setattr(analytics, "_execute_sql_query", fail_query)
    with pytest.raises(RuntimeError, match="injected SQL failure"):
        run_analytics(AnalyticsConfig(data=data, output=output))
    assert not output.exists()


def test_data_quality_query_detects_invalid_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _synthetic_data(tmp_path)
    original = analytics._prepare_analytics_frame

    def inject_bad_label(bundle):
        frame = original(bundle)
        frame.loc[frame.index[0], "label"] = 2
        return frame

    monkeypatch.setattr(analytics, "_prepare_analytics_frame", inject_bad_label)
    output = run_analytics(AnalyticsConfig(data=data, output=tmp_path / "analytics"))
    quality = pd.read_csv(output / "data_quality.csv").set_index("metric")
    assert quality.at["invalid_label_rows", "numeric_value"] == 1
