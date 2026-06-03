"""
Raum 0.6 §5.2 / 1.7 foundation: route a grammar PART to a real scanned Gaussian
splat instead of a hand-authored procedural primitive.

The 109 Sketchfab architecture scans (data/architecture_gs/{category}/{id}/
model.pt, each with positions/scales/rotations/opacities/colors) are already
correctly proportioned -- a real scanned tower has the right aspect ratio, a
real wall the right thickness. Routing parts to them replaces the hand-tuned
procedural geometry (the thing we kept fixing by eye) with measured geometry.

Design:
- A part name -> category (tower_NE -> "tower", wall_S -> "wall", ...).
- For a category, pick a scan deterministically (seeded by the part name so the
  same part is stable across renders) and return its Gaussians as a
  CompositionNode, normalized to the part's canonical local box.
- Graceful fallback: if no scan library is present (e.g. off the 4090) or the
  category has no scan, return None so the caller uses the procedural builder.

This module is import-safe with no scans present: load_scan_library() returns
{} and route_part_to_scan() returns None.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from src.raum.decomposition import CompositionNode, GaussianParams


# Canonical per-axis half-extent each part should occupy in its LOCAL frame,
# so a scan is stretched to the part's real shape (tower = tall, wall = long &
# thin) instead of squashed into a uniform cube. Mirrors the procedural
# builders' footprints in castle_grammar.
_PART_HALF_EXTENT = {
    "tower": (0.30, 0.55, 0.30),
    "keep": (0.40, 0.70, 0.40),
    "wall": (0.70, 0.33, 0.10),
    "gate": (0.70, 0.33, 0.10),
    "gatehouse": (0.70, 0.45, 0.30),
    "arch": (0.30, 0.35, 0.10),
    "door": (0.12, 0.20, 0.06),
    "window": (0.10, 0.14, 0.06),
    "arrow_slit": (0.05, 0.16, 0.06),
    "slit": (0.05, 0.16, 0.06),
    "rock": (0.25, 0.20, 0.25),
    "cliff": (0.9, 0.9, 0.9),
    "roof": (0.35, 0.25, 0.35),
}


# part-kind -> scan category directory name (architecture_gs/<category>)
_KIND_TO_CATEGORY = {
    "tower": "tower",
    "wall": "wall",
    "gate": "gate",
    "gatehouse": "gate",
    "keep": "tower",       # no "keep" scans; a tower scan is the closest body
    "arch": "arch",
    "door": "gate",
    "window": "window",
    "arrow_slit": "window",
    "slit": "window",
    "rock": "rock",
    "cliff": "rock",
    "roof": "roof",
    "stairs": "stairs",
    "column": "column",
    "brick": "brick",
}


def load_scan_library(templates_dir: str | Path) -> dict[str, list[dict]]:
    """Load full Gaussian scans per category.

    Returns {category: [ {positions, scales, rotations, opacities, colors}, ... ]}.
    Empty dict if the directory is absent (the no-scans fallback case).
    """
    templates_dir = Path(templates_dir) if templates_dir else None
    library: dict[str, list[dict]] = {}
    if not templates_dir or not templates_dir.exists():
        return library
    for cat_dir in sorted(templates_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        scans = []
        for obj_dir in sorted(cat_dir.iterdir()):
            mp = obj_dir / "model.pt"
            if mp.exists():
                try:
                    data = torch.load(mp, map_location="cpu", weights_only=True)
                except Exception:
                    continue
                if "positions" in data and len(data["positions"]) > 0:
                    scans.append(data)
        if scans:
            library[cat_dir.name] = scans
    return library


def _part_category(name: str) -> str | None:
    """Map a part node name to a scan category via its kind."""
    try:
        from scripts.castle_grammar import _part_kind
        kind = _part_kind(name)
    except Exception:
        kind = None
    if kind is None:
        # fall back to direct substring match on the category names
        n = name.lower()
        for cat in set(_KIND_TO_CATEGORY.values()):
            if cat in n:
                return cat
        return None
    return _KIND_TO_CATEGORY.get(kind)


def _stable_index(name: str, n: int) -> int:
    """Deterministic scan choice per part name (stable across renders)."""
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % max(n, 1)


def _part_kind_for(name: str) -> str | None:
    try:
        from scripts.castle_grammar import _part_kind
        return _part_kind(name)
    except Exception:
        return None


def route_part_to_scan(name: str, library: dict[str, list[dict]],
                       color=None) -> CompositionNode | None:
    """Return a CompositionNode of real scan Gaussians for this part, or None.

    The scan is stretched to the part's CANONICAL PER-AXIS half-extent (tower =
    tall, wall = long & thin) so it keeps the part's silhouette instead of
    being squashed into a uniform cube. The caller keeps the part node's own
    position/scale; optional tint toward `color`.
    """
    if not library:
        return None
    category = _part_category(name)
    if category is None or category not in library:
        return None
    scans = library[category]
    scan = scans[_stable_index(name, len(scans))]

    pos = scan["positions"].float()
    if pos.shape[0] == 0:
        return None
    # per-axis normalize: map the scan's bbox to the part's canonical half-extent
    kind = _part_kind_for(name) or category
    half = _PART_HALF_EXTENT.get(kind, (0.35, 0.35, 0.35))
    center = pos.mean(dim=0)
    pos = pos - center
    span = pos.abs().amax(dim=0).clamp(min=1e-6)   # per-axis half-span of the scan
    target = torch.tensor(half, dtype=pos.dtype)
    pos = pos / span * target
    # rest fields, with sensible defaults if missing
    n = pos.shape[0]
    scales = scan.get("scales")
    if scales is None:
        # splat size ~ a small fraction of the part's smallest axis
        v = math.log(max(min(half) * 0.12, 1e-3))
        scales = torch.full((n, 3), v)
    else:
        scales = scales.float()
        if scales.dim() == 1:
            scales = scales.unsqueeze(1).repeat(1, 3)
    rots = scan.get("rotations")
    rots = rots.float() if rots is not None else torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    cols = scan.get("colors")
    if cols is None:
        base = color or [0.6, 0.58, 0.53]
        cols = torch.tensor(base).repeat(n, 1)
    else:
        cols = cols.float()
        if color is not None:
            # blend scan color 50/50 toward the requested palette color
            tint = torch.tensor(color)
            cols = 0.5 * cols + 0.5 * tint

    node = CompositionNode(name=name, color=color)
    for i in range(n):
        node.gaussians.append(GaussianParams(
            position=pos[i].tolist(),
            scale=scales[i].tolist() if scales.dim() == 2 else [scales[i].item()] * 3,
            opacity=2.0,
            color=cols[i].tolist(),
            rotation=rots[i].tolist(),
        ))
    return node
