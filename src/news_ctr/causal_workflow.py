"""Atomic publication workflow for quasi-experimental causal impact evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from news_ctr.causal import (
    CausalAnalysis,
    CausalConfig,
    analyze_causal_frame,
    validate_market_panel,
)
from news_ctr.causal_reporting import (
    CausalDecision,
    evaluate_causal_decision,
    render_causal_report,
)
from news_ctr.evidence import EvidenceTier

CAUSAL_ARTIFACTS = (
    "balance.csv",
    "business_impact.csv",
    "causal_report.md",
    "config_snapshot.json",
    "diagnostics.json",
    "did_estimate.csv",
    "event_study.csv",
    "event_study.png",
    "placebo_estimate.csv",
    "run_manifest.json",
)


def _canonicalize_diagnostics(value: object) -> object:
    """Normalize computed floats to the evidence bundle's published precision."""

    if isinstance(value, dict):
        return {key: _canonicalize_diagnostics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_diagnostics(item) for item in value]
    if isinstance(value, float):
        rounded = round(value, 10)
        return 0.0 if rounded == 0 else rounded
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_metadata(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"causal evidence metadata is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("causal evidence metadata must be a JSON object")
    return payload


def _validate_metadata(
    metadata: dict[str, object],
    frame: pd.DataFrame,
    config: CausalConfig,
    input_sha256: str,
) -> None:
    expected = {
        "evidence_tier": EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT.value,
        "panel_sha256": input_sha256,
        "rows": len(frame),
        "markets": int(frame[config.unit_column].nunique()),
        "treated_markets": int(
            frame.groupby(config.unit_column)[config.group_column].first().sum()
        ),
        "pre_weeks": int(
            frame.loc[frame[config.time_column] < config.rollout_week, config.time_column].nunique()
        ),
        "post_weeks": int(
            frame.loc[
                frame[config.time_column] >= config.rollout_week, config.time_column
            ].nunique()
        ),
        "known_effects": {"ctr_absolute": 0.006},
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"causal evidence metadata {key} does not match the input panel")


def _write_causal_outputs(
    output: Path,
    analysis: CausalAnalysis,
    decision: CausalDecision,
    config: CausalConfig,
    config_bytes: bytes,
    provenance: dict[str, object],
    manifest: dict[str, object],
) -> None:
    from news_ctr.causal_visualization import write_event_study_plot

    analysis.balance.to_csv(output / "balance.csv", index=False, float_format="%.10f")
    analysis.business_impact.to_csv(
        output / "business_impact.csv", index=False, float_format="%.10f"
    )
    analysis.did_estimate.to_csv(output / "did_estimate.csv", index=False, float_format="%.10f")
    analysis.event_study.to_csv(output / "event_study.csv", index=False, float_format="%.10f")
    analysis.placebo_estimate.to_csv(
        output / "placebo_estimate.csv", index=False, float_format="%.10f"
    )
    (output / "config_snapshot.json").write_bytes(config_bytes)
    diagnostics = {
        **analysis.diagnostics,
        "decision": {"status": decision.status.value},
    }
    (output / "diagnostics.json").write_text(
        json.dumps(_canonicalize_diagnostics(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "causal_report.md").write_text(
        render_causal_report(
            decision,
            analysis,
            config,
            evidence_tier=EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT.value,
            provenance=provenance,
        ),
        encoding="utf-8",
    )
    write_event_study_plot(
        analysis.event_study,
        output / "event_study.png",
        confidence_level=1 - config.alpha,
    )


def run_causal_impact(input_path: Path, output: Path, config_path: Path) -> Path:
    """Analyze one panel and atomically publish its complete causal evidence bundle."""

    input_path = Path(input_path)
    output = Path(output)
    config_path = Path(config_path)
    if output.exists():
        raise ValueError(f"causal impact output already exists: {output}")
    if not input_path.is_file():
        raise ValueError(f"causal impact input does not exist: {input_path}")
    if not config_path.is_file():
        raise ValueError(f"causal impact config does not exist: {config_path}")

    config_bytes = config_path.read_bytes()
    config = CausalConfig.from_json(config_path)
    frame = pd.read_parquet(input_path)
    integrity = validate_market_panel(frame, config)

    metadata_path = input_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"causal evidence metadata does not exist: {metadata_path}")
    metadata = _load_metadata(metadata_path)
    input_sha256 = _sha256(input_path)
    _validate_metadata(metadata, frame, config, input_sha256)

    analysis = analyze_causal_frame(frame, config)
    decision = evaluate_causal_decision(analysis, config)
    provenance = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "input_sha256": input_sha256,
        "metadata_sha256": _sha256(metadata_path),
    }
    manifest = {
        "schema_version": 1,
        "evidence_tier": EvidenceTier.SYNTHETIC_QUASI_EXPERIMENT.value,
        "artifacts": list(CAUSAL_ARTIFACTS),
        "analysis_contract": {
            "alpha": config.alpha,
            "covariance": f"cluster:{config.unit_column}",
            "event_window": list(config.event_window),
            "outcome": config.outcome_column,
            "placebo_week": config.placebo_week,
            "practical_threshold": config.practical_threshold,
            "reference_week": config.reference_week,
            "rollout_week": config.rollout_week,
            "weighting": config.exposure_column,
        },
        "panel": {
            "markets": integrity["markets"],
            "rows": integrity["rows"],
            "week_max": integrity["week_max"],
            "week_min": integrity["week_min"],
        },
        "input": {"sha256": provenance["input_sha256"]},
        "metadata": {"sha256": provenance["metadata_sha256"]},
        "config": {"sha256": provenance["config_sha256"]},
        "reproduce": "make causal-impact-v4",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.name}.staging-",
    ) as temporary_directory:
        staging = Path(temporary_directory)
        _write_causal_outputs(
            staging,
            analysis,
            decision,
            config,
            config_bytes,
            provenance,
            manifest,
        )
        written = tuple(sorted(item.name for item in staging.iterdir()))
        if written != CAUSAL_ARTIFACTS:
            raise RuntimeError("causal impact artifact set is incomplete")
        os.replace(staging, output)
    return output
