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
    """
    Clean left-to-right three-stage diagram. Single spine, vertical
    middle column for the template library feed into stamp. No
    diagonal return arrows, no stray connectors. Heads in reading
    order (template → color → scale → relation).
    """
    fig, ax = plt.subplots(figsize=(13.0, 6.8))
    W, H = 13.0, 6.8
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ─── Title ───
    ax.text(
        W / 2, H - 0.35,
        "Raum 0.0 architecture",
        ha="center", fontsize=15, fontweight="bold", color=C_BASE,
    )
    ax.text(
        W / 2, H - 0.75,
        "Three stages, left to right. Bridge trained on analytic labels; no pixels in the loss.",
        ha="center", fontsize=10, color=C_TEXT_FAINT,
    )

    # Horizontal centre-line the three stage bodies share.
    SPINE_Y = 2.8

    # ─── 0. Prompt (input) ───
    p_x, p_y, p_w, p_h = 0.4, SPINE_Y - 0.5, 1.7, 1.0
    _box(ax, p_x, p_y, p_w, p_h, "prompt\ntokens", C_BASE, fontsize=11)

    # ─── Stage 1: bridge ───
    b_x, b_y, b_w, b_h = 2.6, 1.3, 3.3, 4.1
    _box(ax, b_x, b_y, b_w, b_h, "", "#EAEFF7", text_color=C_BASE)
    ax.text(
        b_x + b_w / 2, b_y + b_h - 0.35,
        "Stage 1  ·  Raum bridge",
        ha="center", fontsize=11, color=C_BASE, fontweight="bold",
    )
    ax.text(
        b_x + b_w / 2, b_y + b_h - 0.7,
        "~2M params, 4 per-token heads",
        ha="center", fontsize=9, color=C_TEXT_FAINT, style="italic",
    )

    # Heads in reading order top→bottom.
    heads = [
        ("template head", C_ACCENT),
        ("color head",    C_RED),
        ("scale head",    C_POS),
        ("relation head", C_BLUE),
    ]
    head_w, head_h = 2.7, 0.52
    head_x = b_x + (b_w - head_w) / 2
    head_top = b_y + b_h - 1.35
    for i, (label, color) in enumerate(heads):
        hy = head_top - i * (head_h + 0.15)
        _box(ax, head_x, hy, head_w, head_h, label, color, fontsize=10)

    # ─── Stage 2: template library (top) above stamp (bottom) ───
    lib_x, lib_y, lib_w, lib_h = 6.4, 4.0, 3.3, 2.0
    _box(ax, lib_x, lib_y, lib_w, lib_h, "", "#FDEFD3", text_color=C_BASE)
    ax.text(
        lib_x + lib_w / 2, lib_y + lib_h - 0.3,
        "Stage 2  ·  template library",
        ha="center", fontsize=11, color=C_BASE, fontweight="bold",
    )
    ax.text(
        lib_x + lib_w / 2, lib_y + lib_h - 0.6,
        "6 shapes × ~800 Gaussians",
        ha="center", fontsize=9, color=C_TEXT_FAINT, style="italic",
    )
    shapes = ["sphere", "cube", "cylinder", "cone", "plane", "torus"]
    cell_w, cell_h = 0.92, 0.38
    grid_w = cell_w * 3 + 0.16 * 2
    gx0 = lib_x + (lib_w - grid_w) / 2
    for i, s in enumerate(shapes):
        col = i % 3
        row = i // 3
        sx = gx0 + col * (cell_w + 0.16)
        sy = lib_y + 0.2 + (1 - row) * (cell_h + 0.12)
        _box(ax, sx, sy, cell_w, cell_h, s, C_SECOND, fontsize=9)

    stamp_x, stamp_y, stamp_w, stamp_h = 6.4, 1.9, 3.3, 1.4
    _box(
        ax, stamp_x, stamp_y, stamp_w, stamp_h,
        "stamp templates\nposition · color · scale",
        C_SECOND, fontsize=10,
    )

    # ─── Stage 3: viewer ───
    v_x, v_y, v_w, v_h = 10.2, 1.3, 2.5, 4.1
    _box(ax, v_x, v_y, v_w, v_h, "", "#F9FAFC", text_color=C_BASE)
    ax.text(
        v_x + v_w / 2, v_y + v_h - 0.35,
        "Stage 3  ·  viewer",
        ha="center", fontsize=11, color=C_BASE, fontweight="bold",
    )
    ax.text(
        v_x + v_w / 2, v_y + v_h - 0.7,
        "WebGL2 splat renderer",
        ha="center", fontsize=9.5, color=C_TEXT_FAINT, style="italic",
    )

    # Mini splat preview. Red sphere above blue cube, vertically stacked.
    rng = np.random.default_rng(7)
    cx = v_x + v_w / 2
    cy_red = v_y + v_h / 2 + 0.35
    cy_blue = v_y + 1.15
    for _ in range(160):
        dx = rng.normal(0, 0.26)
        dy = rng.normal(0, 0.24)
        ax.scatter(cx + dx, cy_red + dy, s=150, c=C_RED, alpha=0.05, edgecolors="none")
    for _ in range(140):
        dx = rng.normal(0, 0.32)
        dy = rng.normal(0, 0.20)
        ax.scatter(cx + dx, cy_blue + dy, s=150, c=C_BLUE, alpha=0.05, edgecolors="none")

    # ─── Arrows ───
    # prompt → bridge
    _arrow(ax, p_x + p_w, SPINE_Y, b_x, SPINE_Y, color=C_BASE, lw=2.0)
    # bridge → stamp (single horizontal arrow; heads are internal to the bridge)
    _arrow(ax, b_x + b_w, stamp_y + stamp_h / 2, stamp_x, stamp_y + stamp_h / 2,
           color=C_BASE, lw=2.0)
    # library → stamp
    _arrow(ax, lib_x + lib_w / 2, lib_y, lib_x + lib_w / 2, stamp_y + stamp_h,
           color=C_BASE, lw=1.8)
    # stamp → viewer
    _arrow(ax, stamp_x + stamp_w, stamp_y + stamp_h / 2,
           v_x, stamp_y + stamp_h / 2, color=C_BASE, lw=2.0)

    # ─── Footer strip (demo wrapper) ───
    footer_h = 0.5
    footer_y = 0.35
    footer_x = p_x
    footer_w = v_x + v_w - p_x
    _box(ax, footer_x, footer_y, footer_w, footer_h,
         "FastAPI local demo  ·  single-page browser app  ·  runs offline",
         "#555", fontsize=10)

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
