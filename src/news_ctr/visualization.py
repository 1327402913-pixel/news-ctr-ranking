"""Deterministic static visualizations for experiment evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def write_effect_plot(
    effects: pd.DataFrame,
    path: Path,
    *,
    confidence_level: float = 0.95,
) -> Path:
    """Write same-runtime byte-stable treatment-effect confidence intervals."""

    required = {"metric", "effect", "ci_lower", "ci_upper"}
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError(f"effects are missing required columns: {', '.join(missing)}")
    if effects.empty:
        raise ValueError("effects must contain at least one row")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    output = Path(path)
    if output.exists():
        raise ValueError(f"effect plot already exists: {output}")
    numeric = effects[["effect", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("effect estimates and intervals must be finite")

    ordered = effects.sort_values("metric", kind="stable").reset_index(drop=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "text.color": "#0F172A",
        }
    ):
        figure, axes = plt.subplots(
            len(ordered),
            1,
            figsize=(8, 1.65 * len(ordered) + 0.8),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, row in zip(axes[:, 0], ordered.itertuples(index=False), strict=True):
            effect = float(row.effect)
            lower = float(row.ci_lower)
            upper = float(row.ci_upper)
            axis.axvline(0, color="#94A3B8", linewidth=1, linestyle="--")
            axis.errorbar(
                effect,
                0,
                xerr=[[effect - lower], [upper - effect]],
                fmt="o",
                color="#2563EB",
                ecolor="#2563EB",
                capsize=4,
                linewidth=2,
            )
            padding = max((upper - lower) * 0.35, abs(effect) * 0.15, 0.01)
            axis.set_xlim(min(0, lower) - padding, max(0, upper) + padding)
            axis.set_yticks([])
            axis.set_title(str(row.metric), loc="left", fontsize=10, fontweight="bold")
            axis.grid(axis="x", color="#E2E8F0", linewidth=0.7)
            axis.spines[["left", "right", "top"]].set_visible(False)
        confidence_label = f"{100 * confidence_level:g}% CI"
        axes[-1, 0].set_xlabel(f"Treatment - control effect ({confidence_label})")
        figure.suptitle("Randomized experiment treatment effects", fontweight="bold")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output,
            dpi=150,
            format="png",
            facecolor="white",
            metadata={
                "Software": "news-ctr-ranking",
                "Description": (
                    f"Treatment effects with {100 * confidence_level:g}% confidence intervals"
                ),
            },
        )
        plt.close(figure)
    return output
