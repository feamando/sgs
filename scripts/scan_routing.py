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


def route_part_to_scan(name: str, library: dict[str, list[dict]],
                       color=None, target_half: float = 0.35) -> CompositionNode | None:
    """Return a CompositionNode of real scan Gaussians for this part, or None.

    The scan is normalized to a unit-ish box of half-extent `target_half` in
    local frame (the caller keeps the part node's own position/scale), and
    optionally tinted toward `color` so it blends with the scene palette.
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
    # normalize to centered box of half-extent target_half
    center = pos.mean(dim=0)
    pos = pos - center
    extent = pos.abs().max().clamp(min=1e-6)
    pos = pos / extent * target_half
    # rest fields, with sensible defaults if missing
    n = pos.shape[0]
    scales = scan.get("scales")
    if scales is None:
        v = math.log(max(target_half * 0.05, 1e-3))
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
