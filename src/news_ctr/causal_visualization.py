"""Deterministic visualization for quasi-experimental event studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DESCRIPTION = "Synthetic quasi-experiment event study with 95% confidence intervals"


def write_event_study_plot(
    event_study: pd.DataFrame,
    path: Path,
    *,
    confidence_level: float = 0.95,
) -> Path:
    """Write a fixed-style event-study PNG without overwriting existing evidence."""

    required = {"relative_week", "effect", "ci_lower", "ci_upper"}
    missing = sorted(required - set(event_study.columns))
    if missing:
        raise ValueError(f"event-study plot is missing columns: {', '.join(missing)}")
    if event_study.empty:
        raise ValueError("event-study plot data must not be empty")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    destination = Path(path)
    if destination.exists():
        raise ValueError(f"event-study plot already exists: {destination}")

    plotting = event_study.loc[:, sorted(required)].copy()
    values = plotting.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("event-study plot data must contain finite values")
    if plotting["relative_week"].duplicated().any():
        raise ValueError("event-study plot weeks must be unique")
    if (plotting["ci_lower"] > plotting["effect"]).any() or (
        plotting["effect"] > plotting["ci_upper"]
    ).any():
        raise ValueError("event-study confidence intervals must contain their effects")
    plotting = plotting.sort_values("relative_week", kind="stable")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - installed by the causal extra
        raise RuntimeError(
            "causal visualization requires the optional dependency group: pip install '.[causal]'"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(9, 5), dpi=100)
    try:
        weeks = plotting["relative_week"].to_numpy(dtype=float)
        effects = plotting["effect"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                effects - plotting["ci_lower"].to_numpy(dtype=float),
                plotting["ci_upper"].to_numpy(dtype=float) - effects,
            ]
        )
        axis.axhline(0, color="#566573", linewidth=1.2, linestyle="--", zorder=1)
        axis.axvline(0, color="#C0392B", linewidth=1.2, linestyle=":", zorder=1)
        axis.errorbar(
            weeks,
            effects,
            yerr=errors,
            fmt="o-",
            color="#1F4E79",
            ecolor="#5B9BD5",
            capsize=4,
            linewidth=1.6,
            markersize=5,
            zorder=2,
        )
        axis.set_title("Dynamic CTR Effect Around Policy Rollout")
        axis.set_xlabel("Relative week (rollout = 0)")
        axis.set_ylabel("Absolute CTR effect")
        axis.set_xticks(weeks)
        axis.grid(axis="y", color="#D9E2F3", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(
            destination,
            dpi=100,
            metadata={
                "Description": _DESCRIPTION.replace("95%", f"{confidence_level:.0%}"),
                "Software": "news-ctr-ranking",
            },
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)
    return destination
