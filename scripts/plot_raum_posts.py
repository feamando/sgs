"""
Figure generator for the Raum 0.0 LinkedIn post series.

Emits PNGs into docs/raum/raum-posts/ matching the style of the
Planck/Klang series (Planck palette, rounded fancy-boxed diagrams,
single title + subtitle).

Post 1 image: the core idea. A sentence becomes a cloud of Gaussians
in 3D space, visually.

Post 2 image: the Raum 0.0 architecture. Prompt → bridge heads →
template library stamping → WebGL renderer.

Usage:
    python scripts/plot_raum_posts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "raum" / "raum-posts"
OUT.mkdir(parents=True, exist_ok=True)

# Shared palette (continuity with Planck/Klang posts).
C_BASE = "#2B3A67"      # deep blue
C_ACCENT = "#F4A300"    # amber
C_SECOND = "#E69500"    # amber-dark
C_POS = "#4CAF50"       # green
C_RED = "#D9534F"       # red
C_BLUE = "#3B82F6"      # blue
C_GREY = "#B0B0B0"
C_TEXT_FAINT = "#555"

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


def _box(ax, x, y, w, h, label, color, text_color="white", fontsize=11):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor=color, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        color=text_color, fontsize=fontsize, fontweight="bold",
    )


def _arrow(ax, x1, y1, x2, y2, color="#444", lw=1.6):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )


# ──────────────────────────────────────────────────────────────────
# Post 1 — the core idea.
#
# Left: a handwritten-looking sentence. Arrow. Right: a 3D-ish
# scatter of Gaussian blobs (two grouped clusters, one red sphere
# above one blue cube, rendered as 2D gaussian splats with a simple
# painter's algorithm so the image looks Gaussian-splat-ish without
# needing a real 3D renderer).
# ──────────────────────────────────────────────────────────────────

def post1_core_idea():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Title
    ax.text(
        5.5, 5.8, "Raum: a sentence becomes a 3D Gaussian cloud",
        ha="center", fontsize=15, fontweight="bold", color=C_BASE,
    )
    ax.text(
        5.5, 5.35,
        "Objects are templates, relations are placement operators. The splatting math is reused.",
        ha="center", fontsize=10, color=C_TEXT_FAINT,
    )

    # Left panel, the prompt card.
    _box(ax, 0.6, 2.2, 3.6, 1.6, "", "#F7F3EA", text_color=C_BASE)
    ax.text(
        2.4, 3.35, '"a red sphere above',
        ha="center", fontsize=14, color=C_BASE, fontweight="bold",
    )
    ax.text(
        2.4, 2.85, 'a blue cube"',
        ha="center", fontsize=14, color=C_BASE, fontweight="bold",
    )
    ax.text(
        2.4, 1.85, "prompt",
        ha="center", fontsize=10, color=C_TEXT_FAINT, style="italic",
    )

    # Middle arrow + label
    _arrow(ax, 4.5, 3.0, 6.0, 3.0, color=C_BASE, lw=2.2)
    ax.text(
        5.25, 3.3, "Raum bridge",
        ha="center", fontsize=10, color=C_BASE, fontweight="bold",
    )
    ax.text(
        5.25, 2.62, "+ template library",
        ha="center", fontsize=9, color=C_TEXT_FAINT,
    )

    # Right panel, the gaussian scene.
    _box(ax, 6.2, 0.6, 4.4, 4.0, "", "#F9FAFC", text_color=C_BASE)
    ax.text(
        8.4, 0.95, "Gaussian splat scene",
        ha="center", fontsize=10, color=C_TEXT_FAINT, style="italic",
    )

    # Draw two "splatted" clusters with stacked alpha blobs.
    # Red sphere, upper. Blue cube, lower.
    rng = np.random.default_rng(42)

    def scatter_cluster(center, color, spread=0.35, n=200, alpha=0.035, radius=0.55):
        pts = rng.normal(loc=0.0, scale=spread, size=(n, 2))
        r = np.linalg.norm(pts, axis=1)
        keep = pts[r < radius]
        ax.scatter(
            center[0] + keep[:, 0],
            center[1] + keep[:, 1],
            s=140, c=color, alpha=alpha,
            edgecolors="none",
        )

    # Cube (blue) lower
    cube_center = (8.4, 1.9)
    # Render cube-ish shape by layering square-aligned gaussian blobs.
    for dx in np.linspace(-0.45, 0.45, 6):
        for dy in np.linspace(-0.35, 0.35, 5):
            ax.scatter(
                cube_center[0] + dx + rng.normal(0, 0.03),
                cube_center[1] + dy + rng.normal(0, 0.03),
                s=220, c=C_BLUE, alpha=0.08, edgecolors="none",
            )
    scatter_cluster(cube_center, C_BLUE, spread=0.28, n=400, alpha=0.03, radius=0.55)

    # Sphere (red) upper
    sphere_center = (8.4, 3.5)
    scatter_cluster(sphere_center, C_RED, spread=0.32, n=600, alpha=0.035, radius=0.55)

    # Subtle axes / perspective grid suggestion
    for yline in [1.0, 2.5, 4.1]:
        ax.plot([6.5, 10.3], [yline, yline], color=C_GREY, alpha=0.15, lw=0.8)

    out = OUT / "post-1-1_core_idea.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


# ──────────────────────────────────────────────────────────────────
# Post 2 — architecture diagram.
# ──────────────────────────────────────────────────────────────────

def post2_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    # Title
    ax.text(
        5.75, 8.5, "Raum 0.0: prompt → scene graph → Gaussian cloud",
        ha="center", fontsize=14, fontweight="bold", color=C_BASE,
    )
    ax.text(
        5.75, 8.05,
        "Three stages. Bridge is trained on analytic labels, no pixels in the loss.",
        ha="center", fontsize=10, color=C_TEXT_FAINT,
    )

    # Input prompt
    _box(ax, 0.3, 4.0, 1.9, 1.0, "prompt\ntokens", C_BASE)

    # ─── Stage 1: bridge with 4 heads ───
    bridge_x, bridge_y, bridge_w, bridge_h = 2.8, 2.4, 3.2, 4.2
    _box(ax, bridge_x, bridge_y, bridge_w, bridge_h, "", "#EAEFF7", text_color=C_BASE)
    ax.text(
        bridge_x + bridge_w / 2, bridge_y + bridge_h - 0.35,
        "Stage 1: Raum bridge  (~2M params)",
        ha="center", fontsize=10.5, color=C_BASE, fontweight="bold",
    )
    ax.text(
        bridge_x + bridge_w / 2, bridge_y + bridge_h - 0.7,
        "per-token heads, analytic labels",
        ha="center", fontsize=8.8, color=C_TEXT_FAINT, style="italic",
    )

    head_w, head_h = 2.6, 0.65
    head_x = bridge_x + (bridge_w - head_w) / 2
    heads = [
        ("template head", C_ACCENT),
        ("color head", C_RED),
        ("scale head", C_POS),
        ("relation head", C_BLUE),
    ]
    for i, (label, color) in enumerate(heads):
        hy = bridge_y + 0.45 + i * (head_h + 0.18)
        _box(ax, head_x, hy, head_w, head_h, label, color, fontsize=10)

    # ─── Stage 2: template library ───
    lib_x, lib_y, lib_w, lib_h = 6.6, 4.7, 2.4, 2.2
    _box(ax, lib_x, lib_y, lib_w, lib_h, "", "#FDEFD3", text_color=C_BASE)
    ax.text(
        lib_x + lib_w / 2, lib_y + lib_h - 0.3,
        "Stage 2: template library",
        ha="center", fontsize=10.5, color=C_BASE, fontweight="bold",
    )
    ax.text(
        lib_x + lib_w / 2, lib_y + lib_h - 0.6,
        "6 shapes × ~800 Gaussians",
        ha="center", fontsize=8.8, color=C_TEXT_FAINT, style="italic",
    )
    shapes = ["sphere", "cube", "cylinder", "cone", "plane", "torus"]
    for i, s in enumerate(shapes):
        col = i % 3
        row = i // 3
        sx = lib_x + 0.2 + col * 0.72
        sy = lib_y + 0.3 + (1 - row) * 0.55
        _box(ax, sx, sy, 0.65, 0.45, s, C_SECOND, fontsize=8)

    # Stamping box
    stamp_x, stamp_y, stamp_w, stamp_h = 6.6, 2.6, 2.4, 1.5
    _box(
        ax, stamp_x, stamp_y, stamp_w, stamp_h,
        "stamp templates\nposition · color · scale",
        C_SECOND, fontsize=9.5,
    )

    # ─── Stage 3: WebGL viewer ───
    view_x, view_y, view_w, view_h = 9.4, 3.4, 1.9, 2.2
    _box(ax, view_x, view_y, view_w, view_h, "", "#F9FAFC", text_color=C_BASE)
    ax.text(
        view_x + view_w / 2, view_y + view_h - 0.35,
        "Stage 3",
        ha="center", fontsize=10.5, color=C_BASE, fontweight="bold",
    )
    ax.text(
        view_x + view_w / 2, view_y + view_h - 0.65,
        "WebGL2 splat viewer",
        ha="center", fontsize=9, color=C_TEXT_FAINT, style="italic",
    )
    # mini splat icon inside the viewer box
    rng = np.random.default_rng(7)
    cx, cy = view_x + view_w / 2, view_y + view_h / 2 - 0.45
    for _ in range(80):
        dx = rng.normal(0, 0.22)
        dy = rng.normal(0, 0.22)
        ax.scatter(cx + dx, cy + dy - 0.05, s=140, c=C_RED, alpha=0.06, edgecolors="none")
    for _ in range(60):
        dx = rng.normal(0, 0.20)
        dy = rng.normal(0, 0.15)
        ax.scatter(cx + dx, cy + dy - 0.55, s=140, c=C_BLUE, alpha=0.06, edgecolors="none")

    # Output card
    _box(ax, 0.3, 0.6, 2.2, 0.9, "FastAPI\nlocal demo", "#555", fontsize=10)

    # Arrows
    _arrow(ax, 2.2, 4.5, 2.8, 4.5, color=C_BASE, lw=2.0)            # tokens → bridge
    _arrow(ax, 6.0, 5.7, 6.6, 5.7, color=C_BASE, lw=1.8)            # bridge → library
    _arrow(ax, 6.0, 3.3, 6.6, 3.3, color=C_BASE, lw=1.8)            # bridge → stamping
    _arrow(ax, 7.8, 4.7, 7.8, 4.1, color=C_BASE, lw=1.6)            # library → stamping
    _arrow(ax, 9.0, 3.3, 9.4, 4.0, color=C_BASE, lw=1.8)            # stamping → viewer
    _arrow(ax, 9.4, 4.5, 9.4, 4.5, color=C_BASE, lw=0.0)            # (no-op placeholder)
    _arrow(ax, 1.4, 2.4, 1.4, 1.5, color=C_GREY, lw=1.2)            # tokens → demo
    _arrow(ax, 9.4, 3.4, 2.5, 1.2, color=C_GREY, lw=1.0)            # viewer → demo (browser)

    # Legend-ish caption at bottom
    ax.text(
        5.75, 0.3,
        "Training loss is analytic (positions, colors, relations). No rendered ground truth.",
        ha="center", fontsize=9.5, color=C_TEXT_FAINT, style="italic",
    )

    out = OUT / "post-2-1_architecture.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


# ──────────────────────────────────────────────────────────────────

def main():
    written = [
        post1_core_idea(),
        post2_architecture(),
    ]
    print("wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
