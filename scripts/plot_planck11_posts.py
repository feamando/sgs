"""
Figure generator for the Planck 1.1 LinkedIn post series.

Reads results/planck11_validation.json and emits PNGs into
docs/planck/planck-posts/ with post-{n}-{slot}_{description}.png naming.

Usage:
    python scripts/plot_planck11_posts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "planck11_validation.json"
OUT = REPO / "docs" / "planck" / "planck-posts"
OUT.mkdir(parents=True, exist_ok=True)

# Brand-adjacent palette (orange + dark blue already used in Klang plots)
C_BASE = "#2B3A67"
C_BLOB = "#F4A300"
C_POS = "#4CAF50"
C_NEG = "#D9534F"
C_GREY = "#B0B0B0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


def _load() -> dict:
    with open(RESULTS) as f:
        return json.load(f)


def post1_architecture():
    """A simple diagram: token stream → base forward + parallel blob retrieval → gate → logits."""
    fig, ax = plt.subplots(figsize=(10, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 9.0)
    ax.axis("off")

    def box(x, y, w, h, label, color, text_color="white", fontsize=11):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=color, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold")

    def arrow(x1, y1, x2, y2, color="#444", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.6))

    # Title
    ax.text(5.0, 8.4, "H-SGS: retrieval is built in",
            ha="center", fontsize=14, fontweight="bold", color=C_BASE)
    ax.text(5.0, 7.9,
            "Blobs live inside the model, indexed by the model's own hidden states.",
            ha="center", fontsize=10, color="#555")

    # Input
    box(0.2, 3.0, 1.5, 0.8, "prompt\ntokens", C_BASE)

    # Base forward (upper path)
    box(2.3, 5.1, 2.6, 1.0, "SGS base\n(Planck 1.0)", C_BASE)
    # Blob pathway (lower path)
    box(2.3, 2.0, 2.6, 1.0, "blob store\n(50k Gaussians)", C_BLOB, "white")
    box(2.3, 0.4, 2.6, 1.0, "top-k Mahalanobis +\ntransmittance composite",
        "#E69500", "white", fontsize=10)

    # Gate
    box(5.8, 2.9, 1.6, 1.2, "learned\ngate", "#555", "white")

    # Output
    box(8.2, 3.0, 1.6, 0.8, "logits", C_BASE)

    # Arrows
    arrow(1.7, 3.4, 2.3, 5.6)           # tokens → base
    arrow(1.7, 3.4, 2.3, 2.5)           # tokens → blob store
    arrow(3.6, 2.0, 3.6, 1.4)           # blob store → retrieval
    arrow(4.9, 5.6, 5.8, 3.8)           # base → gate
    arrow(4.9, 0.9, 5.8, 3.0)           # retrieval → gate
    arrow(7.4, 3.4, 8.2, 3.4)           # gate → logits

    out = OUT / "post-1-1_architecture.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def post2_gates_setup():
    """Visual summary of the 4 gates: what they test, before results."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    ax.text(5.0, 5.5, "Four-gate validator: what we required",
            ha="center", fontsize=14, fontweight="bold", color=C_BASE)

    rows = [
        ("Gate 3: perplexity", "Planck 1.1 val loss strictly lower than Planck 1.0"),
        ("Gate 2: utilisation", "Mean effective blob weight > 0.05 (pathway actually used)"),
        ("Gate 1: base intact", "With blobs off, |Δ val loss vs 1.0| ≤ 0.05"),
        ("Gate 4a: intra-sample", "Within-sample 4-gram repeats ≤ Planck 1.0"),
    ]
    y = 4.5
    for name, desc in rows:
        box = mpatches.FancyBboxPatch(
            (0.5, y - 0.45), 9.0, 0.85, boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#ccc", facecolor="#F7F4EE",
        )
        ax.add_patch(box)
        ax.text(0.9, y, name, ha="left", va="center",
                fontsize=12, fontweight="bold", color=C_BASE)
        ax.text(4.0, y, desc, ha="left", va="center", fontsize=11, color="#333")
        y -= 1.05

    out = OUT / "post-2-1_gates.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def post3_headline_metrics():
    """Gate 2 + Gate 3 headline chart: val loss, perplexity, blob usage."""
    d = _load()
    g3 = d["gates"]["gate_3_perplexity"]
    g2 = d["gates"]["gate_2_blob_usage"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    # Panel 1.val loss
    ax = axes[0]
    ax.bar([0, 1], [g3["planck10_loss"], g3["planck11_loss"]],
           color=[C_BASE, C_BLOB], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Planck 1.0", "Planck 1.1"])
    ax.set_ylabel("val cross-entropy")
    ax.set_title("Gate 3: validation loss")
    ax.text(0, g3["planck10_loss"] + 0.02, f"{g3['planck10_loss']:.3f}",
            ha="center", fontsize=10)
    ax.text(1, g3["planck11_loss"] + 0.02, f"{g3['planck11_loss']:.3f}",
            ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(g3["planck10_loss"], g3["planck11_loss"]) * 1.25)

    # Panel 2.perplexity
    ax = axes[1]
    ax.bar([0, 1], [g3["planck10_ppl"], g3["planck11_ppl"]],
           color=[C_BASE, C_BLOB], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Planck 1.0", "Planck 1.1"])
    ax.set_ylabel("perplexity")
    ax.set_title("Gate 3: perplexity (−14%)")
    ax.text(0, g3["planck10_ppl"] + 0.1, f"{g3['planck10_ppl']:.2f}",
            ha="center", fontsize=10)
    ax.text(1, g3["planck11_ppl"] + 0.1, f"{g3['planck11_ppl']:.2f}",
            ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(g3["planck10_ppl"], g3["planck11_ppl"]) * 1.2)

    # Panel 3.blob usage
    ax = axes[2]
    threshold = 0.05
    ax.bar([0], [g2["mean_blob_weight"]], color=C_POS, width=0.4)
    ax.axhline(threshold, color=C_NEG, linestyle="--", linewidth=1.2,
               label=f"pass threshold = {threshold}")
    ax.set_xticks([0])
    ax.set_xticklabels(["Planck 1.1"])
    ax.set_ylabel("mean effective blob weight")
    ax.set_title("Gate 2: blob utilisation")
    ax.set_ylim(0, 0.08)
    ax.text(0, g2["mean_blob_weight"] + 0.002,
            f"{g2['mean_blob_weight']:.3f}", ha="center",
            fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Gates 2 & 3 PASS: blobs help likelihood and are being used",
                 fontsize=13, fontweight="bold", color=C_BASE, y=1.02)
    plt.tight_layout()
    out = OUT / "post-3-1_gate_2_3_pass.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def post4_gate4a_failure():
    """The headline failure: Gate 4a intra-sample repetition worsened."""
    d = _load()
    g4a = d["gates"]["gate_4a_intra_sample_repetition"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.array([0, 1])
    ys = np.array([g4a["planck10_mean_per_sample"], g4a["planck11_mean_per_sample"]])
    bars = ax.bar(xs, ys, color=[C_BASE, C_NEG], width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Planck 1.0", "Planck 1.1"])
    ax.set_ylabel("mean 4-gram repeats per sample")
    ax.set_title("Gate 4a: within-sample looping got worse (+62%)",
                 color=C_BASE)
    for i, v in enumerate(ys):
        ax.text(xs[i], v + 0.15, f"{v:.2f}", ha="center",
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(ys) * 1.2)

    # Annotation for the delta
    ax.annotate(
        f"Δ = +{g4a['delta_mean']:.2f}",
        xy=(1, g4a["planck11_mean_per_sample"]),
        xytext=(0.5, max(ys) * 1.05),
        ha="center", fontsize=11, color=C_NEG, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2),
    )
    plt.tight_layout()
    out = OUT / "post-4-1_gate_4a_fail.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def post4_gate4b_weak_upside():
    """Informational Gate 4b: small move toward consistency, not enough."""
    d = _load()
    g4b = d["gates"]["gate_4b_cross_sample_diversity"]
    p10 = g4b["planck10"]
    p11 = g4b["planck11"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    ax = axes[0]
    xs = [0, 1]
    ys = [p10["unique_ratio"], p11["unique_ratio"]]
    ax.bar(xs, ys, color=[C_BASE, C_BLOB], width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Planck 1.0", "Planck 1.1"])
    ax.set_ylabel("unique 4-gram ratio")
    ax.set_title("Cross-sample uniqueness")
    for i, v in enumerate(ys):
        ax.text(xs[i], v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.text(0.5, 0.18, "lower = more consistent",
            ha="center", fontsize=9, color="#666", style="italic")

    ax = axes[1]
    ys = [p10["jaccard_mean"], p11["jaccard_mean"]]
    ax.bar(xs, ys, color=[C_BASE, C_BLOB], width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Planck 1.0", "Planck 1.1"])
    ax.set_ylabel("mean pairwise Jaccard")
    ax.set_title("Cross-sample agreement")
    for i, v in enumerate(ys):
        ax.text(xs[i], v + 0.0003, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(ys) * 1.3)
    ax.text(0.5, max(ys) * 0.25, "higher = more agreement",
            ha="center", fontsize=9, color="#666", style="italic")

    fig.suptitle(
        "Gate 4b: small drift toward consistency, not enough to offset 4a",
        fontsize=12, fontweight="bold", color=C_BASE, y=1.02,
    )
    plt.tight_layout()
    out = OUT / "post-4-2_gate_4b_weak_upside.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def post4_roadmap():
    """Planck 1.3 roadmap as a visual stack."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    ax.text(5.0, 5.6, "Planck 1.3: fix Gate 4a without losing Gate 3",
            ha="center", fontsize=14, fontweight="bold", color=C_BASE)

    items = [
        ("1", "Inference-time top-k / t_max sweep",
         "no retraining.dials before weights"),
        ("2", "Frozen-base retrain for Gate 1",
         "cleans up the gate-design false fail"),
        ("3", "Blob count sweep: 50k → 200k → 500k",
         "decisive experiment on collision hypothesis"),
        ("4", "Live blob addition (in-model RAG)",
         "can the index grow without retraining?"),
        ("5", "Inference-time t_max as CLI flag",
         "one checkpoint, creative vs consistent"),
    ]
    y = 4.7
    colors = [C_POS, C_BASE, C_BLOB, C_BASE, C_POS]
    for (num, title, subtitle), color in zip(items, colors):
        rect = mpatches.FancyBboxPatch(
            (0.5, y - 0.45), 9.0, 0.85, boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#ddd", facecolor="#F7F4EE",
        )
        ax.add_patch(rect)
        # Numeric pill
        circle = mpatches.Circle((1.1, y), 0.3, color=color)
        ax.add_patch(circle)
        ax.text(1.1, y, num, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        ax.text(1.7, y + 0.12, title, ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=C_BASE)
        ax.text(1.7, y - 0.18, subtitle, ha="left", va="center",
                fontsize=10, color="#555", style="italic")
        y -= 1.02

    plt.tight_layout()
    out = OUT / "post-4-3_planck13_roadmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main():
    paths = [
        post1_architecture(),
        post2_gates_setup(),
        post3_headline_metrics(),
        post4_gate4a_failure(),
        post4_gate4b_weak_upside(),
        post4_roadmap(),
    ]
    print("Generated:")
    for p in paths:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
