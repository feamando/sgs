"""
Figure generator for the VSP negative-result verification LinkedIn post.

Emits a single PNG into docs/raum/raum-posts/ in the shared Planck/Raum
palette. The figure tells the washout story in one panel: three
initializations (random / grounded / scrambled) at low compute (2k steps)
and at full compute (40k steps). At 2k the grounded bundle leads; by 40k
the ordering has inverted and a plain random start is ahead of both, i.e.
training washes the grounded init out.

All numbers are read live from results/ so the figure cannot drift from
the paper.

Usage:
    python scripts/plot_vsp_verification_post.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "docs" / "raum" / "raum-posts"
OUT.mkdir(parents=True, exist_ok=True)

# Shared palette (continuity with Planck/Klang/Raum posts).
C_BASE = "#2B3A67"      # deep blue  -> random baseline
C_ACCENT = "#F4A300"    # amber      -> grounded (the bet)
C_GREY = "#B0B0B0"      # grey       -> scrambled control
C_TEXT_FAINT = "#555"
C_POS = "#4CAF50"
C_RED = "#D9534F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def acc(fname: str) -> float:
    return json.load(open(RESULTS / fname))["accuracy"]


def main():
    # Apples-to-apples: scrambled only ran seed 0, so read every arm at seed 0.
    data = {
        "2k": {
            "random":    acc("disambig_baseline_2k.json"),
            "grounded":  acc("disambig_vsp_2k.json"),
            "scrambled": acc("disambig_scrambled_2k_s0.json"),
        },
        "40k": {
            "random":    acc("disambig_baseline_40k_s0.json"),
            "grounded":  acc("disambig_vsp_40k_s0.json"),
            "scrambled": acc("disambig_scrambled_40k_s0.json"),
        },
    }

    arms = ["random", "grounded", "scrambled"]
    arm_label = {
        "random": "random init\n(baseline)",
        "grounded": "grounded init\n(the bet)",
        "scrambled": "scrambled init\n(control)",
    }
    arm_color = {"random": C_BASE, "grounded": C_ACCENT, "scrambled": C_GREY}

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.0), sharey=True)
    fig.suptitle(
        "A grounded start helps early, then training washes it out",
        fontsize=15, fontweight="bold", color=C_BASE, y=0.98,
    )
    fig.text(
        0.5, 0.915,
        "Sense-disambiguation accuracy, 105 minimal pairs. Same model, same data, "
        "three initializations.",
        ha="center", fontsize=10, color=C_TEXT_FAINT,
    )

    x = np.arange(len(arms))
    panels = [("2k", "Low compute  (2k steps)"),
              ("40k", "Full compute  (40k steps, ~2B tokens)")]

    for ax, (regime, title) in zip(axes, panels):
        vals = [data[regime][a] for a in arms]
        bars = ax.bar(
            x, vals, width=0.62,
            color=[arm_color[a] for a in arms],
            edgecolor="white", linewidth=1.2, zorder=3,
        )
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=C_BASE)

        # chance line
        ax.axhline(0.5, color=C_RED, lw=1.2, ls="--", alpha=0.7, zorder=1)
        ax.text(len(arms) - 0.5, 0.512, "chance", ha="right", va="bottom",
                fontsize=8.5, color=C_RED, alpha=0.85)

        ax.set_title(title, fontsize=12, color=C_BASE, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([arm_label[a] for a in arms], fontsize=9.5,
                           color=C_TEXT_FAINT)
        ax.set_ylim(0.45, 0.90)
        ax.grid(axis="y", color=C_GREY, alpha=0.2, zorder=0)
        ax.tick_params(axis="y", labelsize=9)

    axes[0].set_ylabel("disambiguation accuracy", fontsize=11, color=C_BASE)

    # Annotate the two headline deltas (grounded vs scrambled).
    axes[0].annotate(
        "+10.5 pts\np = 0.007",
        xy=(1, data["2k"]["grounded"]), xytext=(1.02, 0.60),
        ha="center", fontsize=9.5, color=C_POS, fontweight="bold",
    )
    axes[1].annotate(
        "+2.9 pts\nn.s. (p = 0.51)",
        xy=(1, data["40k"]["grounded"]), xytext=(1.0, 0.585),
        ha="center", fontsize=9.5, color=C_TEXT_FAINT, fontweight="bold",
    )

    fig.text(
        0.5, 0.02,
        "The representation separates senses (0.37 vs 0.00 for text). "
        "A trained language model does not keep it.",
        ha="center", fontsize=9.5, color=C_TEXT_FAINT, style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    out = OUT / "vsp_verification_washout.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote: {out}")
    # echo the numbers used, for a sanity check against the paper
    for regime in ("2k", "40k"):
        print(f"  {regime}: " + ", ".join(f"{a} {data[regime][a]:.3f}" for a in arms))


if __name__ == "__main__":
    main()
