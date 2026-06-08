"""
Raum 0.5: deterministic castle scene grammar.

Builds composition trees for "an honest castle on a hill" out of ATOMIC
parts: walls and towers are courses of small stones, not smooth boxes;
tops carry crenellations; the gate is a wall segment with the center
omitted plus an arch. Every leaf is sized small relative to its parent so
the downstream densify stage grows features instead of blobs.

The grammar is the single source of truth for both:
  1. The §3 fixed-scene demo (render with no model in the loop).
  2. The §4 Raum 1.5 training data (sample thousands of labeled trees).

Two ways to use it:

  # One scene to a JSON file (for the demo --scene-file path)
  python scripts/castle_grammar.py --preset castle_on_hill \
    --towers 4 --wall-courses 6 --with-gate --trees 8 \
    --output output/castle_05.json

  # A training set: many randomized scenes + paraphrased prompts
  python scripts/castle_grammar.py --sample 8000 \
    --domain castle,hill,village,tower,keep --paraphrase \
    --output-dir data/decomposition_trees/castle_15

Design rules (the compositional awareness the 1.4 decomposer lacked):
  - 4 towers at the corners of a square wall ring
  - a wall segment between each adjacent tower pair
  - the gate replaces the front (south, +Z) wall segment
  - the keep sits centered inside the ring, taller than the walls
  - the whole castle sits on top of the hill (y-offset = hill height)
  - trees scatter on the hill skirt, below the wall base, never in the ring
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import (
    CompositionNode, GaussianParams, save_tree, print_tree, tree_to_tensors,
)


# ── Atomic primitive generators ───────────────────────────────────────
#
# Each returns a list[GaussianParams] in the node's LOCAL frame. Scales are
# log-scale; small relative to the part so neighbours overlap into a surface
# and densification adds detail rather than inflating one blob.

def _stone_log_scale(stone_size: float) -> list[float]:
    """Log-scale for an isotropic stone roughly stone_size across."""
    v = math.log(max(stone_size * 0.5, 1e-3))
    return [v, v, v]


def _slab_log_scale(sx: float, sy: float, sz: float) -> list[float]:
    """Anisotropic log-scale (Raum 0.6): a flat brick-shaped stone.

    Walls/faces use a thin Z (depth into the wall) and wider X/Y so the
    oriented-ellipsoid renderer shows masonry blocks, not round beads.
    """
    return [math.log(max(sx, 1e-3)), math.log(max(sy, 1e-3)),
            math.log(max(sz, 1e-3))]


def stone_course(n_around: int, radius: float, y: float, stone_size: float,
                 color, jitter: float = 0.0, y_span: float = 1.0) -> list[GaussianParams]:
    """One horizontal ring of stones at height y (a course of a round tower)."""
    out = []
    for i in range(n_around):
        theta = 2 * math.pi * i / n_around
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        if jitter:
            x += random.uniform(-jitter, jitter)
            z += random.uniform(-jitter, jitter)
        out.append(GaussianParams(
            position=[x, y, z], scale=_stone_log_scale(stone_size),
            opacity=2.0, color=_weathered(color, 0.04, y - y_span * 0.5, y_span),
        ))
    return out


def stone_wall_face(length: float, height: float, courses: int, stone_size: float,
                    color, gap_center: float = 0.0,
                    openings: list[tuple] | None = None) -> list[GaussianParams]:
    """
    A flat wall built from courses of stones along X, stacked in Y.

    gap_center > 0 omits stones within that half-width of x=0 (the gate void).
    openings: list of (cx, cy, half_w, half_h) rectangles (wall-local) to carve
    out of the face -- where windows/doors/slits sit.
    """
    out = []
    openings = openings or []
    n_along = max(2, int(length / stone_size))

    def in_opening(x, y):
        for (cx, cy, hw, hh) in openings:
            if abs(x - cx) < hw and abs(y - cy) < hh:
                return True
        return False

    for c in range(courses):
        y = (c / max(courses - 1, 1) - 0.5) * height
        # brick-offset alternate courses
        offset = (stone_size * 0.5) if (c % 2) else 0.0
        for i in range(n_along):
            x = (i / max(n_along - 1, 1) - 0.5) * length + offset
            if abs(x) > length / 2:
                continue
            # gate void: leave a hole in the lower courses
            if gap_center > 0 and abs(x) < gap_center and y < height * 0.15:
                continue
            # window/door/slit voids
            if in_opening(x, y):
                continue
            # flat brick: wide in X/Y, thin in Z (into the wall)
            out.append(GaussianParams(
                position=[x, y, 0.0],
                scale=_slab_log_scale(stone_size * 0.55, stone_size * 0.40, stone_size * 0.22),
                opacity=2.0, color=_weathered(color, 0.04, y, height),
            ))
    return out


def crenellation_strip(length: float, stone_size: float, color,
                       axis: str = "x") -> list[GaussianParams]:
    """Alternating merlons (blocks) and gaps along the top edge."""
    out = []
    n = max(3, int(length / (stone_size * 2)))
    for i in range(n):
        if i % 2:  # gap
            continue
        t = (i / max(n - 1, 1) - 0.5) * length
        pos = [t, 0.0, 0.0] if axis == "x" else [0.0, 0.0, t]
        out.append(GaussianParams(
            position=pos, scale=_stone_log_scale(stone_size * 1.2),
            opacity=2.0, color=_shift(color, 0.03),
        ))
    return out


def crenellation_ring(radius: float, stone_size: float, color) -> list[GaussianParams]:
    """Merlons around the rim of a round tower."""
    out = []
    n = max(6, int(2 * math.pi * radius / (stone_size * 2)))
    for i in range(n):
        if i % 2:
            continue
        theta = 2 * math.pi * i / n
        out.append(GaussianParams(
            position=[radius * math.cos(theta), 0.0, radius * math.sin(theta)],
            scale=_stone_log_scale(stone_size * 1.2), opacity=2.0,
            color=_shift(color, 0.03),
        ))
    return out


def cone_roof(radius: float, height: float, stone_size: float, color) -> list[GaussianParams]:
    """Filled-ish conical roof from stacked shrinking rings."""
    out = []
    layers = max(3, int(height / stone_size))
    for L in range(layers):
        t = L / max(layers - 1, 1)
        r = radius * (1 - t)
        y = t * height
        n_around = max(3, int(2 * math.pi * r / stone_size))
        for i in range(n_around):
            theta = 2 * math.pi * i / n_around
            out.append(GaussianParams(
                position=[r * math.cos(theta), y, r * math.sin(theta)],
                scale=_stone_log_scale(stone_size), opacity=2.0,
                color=_shift(color, 0.03),
            ))
    return out


def dome(radius: float, height: float, stone_size: float, color) -> list[GaussianParams]:
    """Hemisphere mound (the hill), filled in shells so it reads solid."""
    out = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    n = max(80, int((radius * radius) / (stone_size * stone_size) * 2))
    for i in range(n):
        theta = 2.0 * math.pi * i / golden
        phi = math.acos(1.0 - (i + 0.5) / n)  # upper hemisphere
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi) * height
        z = radius * math.sin(phi) * math.sin(theta)
        out.append(GaussianParams(
            position=[x, y, z], scale=_stone_log_scale(stone_size * 1.5),
            opacity=2.0, color=_shift(color, 0.05),
        ))
    return out


def tree_parts(trunk_h: float, canopy_r: float, stone_size: float):
    """Returns (trunk_gaussians, canopy_gaussians) for a conifer."""
    trunk, canopy = [], []
    trunk_color = [0.4, 0.26, 0.13]
    canopy_color = [0.12, 0.45, 0.15]
    # trunk: short stack
    for c in range(max(2, int(trunk_h / stone_size))):
        y = (c / max(int(trunk_h / stone_size), 1)) * trunk_h
        trunk.append(GaussianParams(
            position=[0, y, 0], scale=_stone_log_scale(stone_size * 0.7),
            opacity=2.0, color=_shift(trunk_color, 0.03)))
    # canopy: cone of foliage
    canopy = cone_roof(canopy_r, canopy_r * 1.6, stone_size, canopy_color)
    return trunk, canopy


def _shift(color, var: float):
    """Small per-stone color variation so surfaces are not flat."""
    return [min(1.0, max(0.0, c + random.uniform(-var, var))) for c in color]


# Raum 0.7 material: weathering palettes blended into stone for aged masonry.
_MOSS = [0.30, 0.42, 0.22]      # green growth in damp/low areas
_DIRT = [0.34, 0.28, 0.20]      # brown grime
_LICHEN = [0.66, 0.66, 0.58]    # pale crusty patches


def _weathered(base, var: float = 0.04, y: float = 0.0, y_span: float = 1.0):
    """Per-stone aged-masonry color (Raum 0.7 material axis).

    Beyond flat jitter: stones near the bottom of a part trend toward moss/dirt,
    tops are lighter (sun-bleached), and a fraction get a lichen/dirt tint. Gives
    correlated, believable variation instead of uniform noise.
    """
    r, g, b = base
    # height term: -1 at the part's base, +1 near its top
    h = max(-1.0, min(1.0, (y / max(y_span, 1e-6)) * 2.0)) if y_span else 0.0
    # STRONG height shading: tops clearly sun-bleached, bases clearly in shadow.
    light = 0.18 * h
    r += light; g += light; b += light
    # weathering tints, applied to a large fraction and at high strength so the
    # variation actually reads through the splat blending.
    roll = random.random()
    if h < 0.1 and roll < 0.55:         # moss/dirt over the lower half
        tint = _MOSS if random.random() < 0.6 else _DIRT
        m = random.uniform(0.4, 0.75)   # heavy blend
        r = r*(1-m) + tint[0]*m; g = g*(1-m) + tint[1]*m; b = b*(1-m) + tint[2]*m
    elif roll < 0.30:                   # frequent lichen patches anywhere
        m = random.uniform(0.3, 0.55)
        r = r*(1-m) + _LICHEN[0]*m; g = g*(1-m) + _LICHEN[1]*m; b = b*(1-m) + _LICHEN[2]*m
    # coarse per-stone jitter on top (bigger so individual stones differ)
    j = max(var, 0.07)
    r += random.uniform(-j, j); g += random.uniform(-j, j); b += random.uniform(-j, j)
    return [min(1.0, max(0.0, c)) for c in (r, g, b)]


# ── Compositional builders ────────────────────────────────────────────

STONE = 0.06  # atomic stone size in castle-local units


def build_tower(name: str, pos, courses: int, radius: float, stone_color,
                with_roof: bool = True) -> CompositionNode:
    """Round tower: stacked stone courses + crenellation ring + optional roof."""
    tower = CompositionNode(name=name, position=pos, scale=1.0)
    height = courses * STONE * 1.6
    body = CompositionNode(name=f"{name}_body", color=stone_color)
    for c in range(courses):
        y = (c / max(courses - 1, 1)) * height
        body.gaussians.extend(stone_course(
            max(6, int(2 * math.pi * radius / STONE)), radius, y, STONE, stone_color,
            y_span=height))
    tower.children.append(body)
    tower.children.append(CompositionNode(
        name=f"{name}_crenellation", position=[0, height, 0],
        color=stone_color, gaussians=crenellation_ring(radius, STONE, stone_color)))
    if with_roof:
        # shallower, slightly-overhanging conical roof (was radius*1.8 tall ->
        # spiky "christmas tree"; radius*1.1 reads as a stocky tower cap)
        tower.children.append(CompositionNode(
            name=f"{name}_roof", position=[0, height + STONE, 0],
            color=[0.5, 0.2, 0.12],
            gaussians=cone_roof(radius * 1.25, radius * 1.1, STONE, [0.5, 0.2, 0.12])))
    return tower


def build_wall(name: str, pos, length: float, courses: int, stone_color,
               rot=None, is_gate: bool = False, windows: int = 0,
               slits: int = 0) -> CompositionNode:
    """Wall segment: stone face + crenellation top. Gate omits center stones.

    windows/slits > 0 carve evenly-spaced recessed openings in the face.
    """
    height = courses * STONE * 1.6
    wall = CompositionNode(name=name, position=pos, scale=1.0,
                           rotation=rot or [1.0, 0.0, 0.0, 0.0])
    gap = length * 0.22 if is_gate else 0.0
    # stone_wall_face centres courses on y=0 (spans +/-height/2). Lift the whole
    # wall by height/2 so it is BASE-ALIGNED like the towers/keep (base at y=0),
    # otherwise the lower half sinks below the ground.
    base_lift = height / 2

    # plan the openings (carve from the face, then place recessed dark nodes)
    openings, opening_nodes = [], []
    win_hw, win_hh = STONE * 1.2, STONE * 1.6
    slit_hw, slit_hh = STONE * 0.5, STONE * 2.0
    n_open = windows + slits
    if n_open and not is_gate:
        for k in range(n_open):
            cx = (((k + 1) / (n_open + 1)) - 0.5) * length * 0.9
            cy = height * 0.12
            is_slit = k >= windows
            hw, hh = (slit_hw, slit_hh) if is_slit else (win_hw, win_hh)
            openings.append((cx, cy, hw, hh))
            node = build_arrow_slit(f"{name}_slit_{k}", [cx, cy + base_lift, -STONE * 0.5]) if is_slit \
                else build_window(f"{name}_window_{k}", [cx, cy + base_lift, -STONE * 0.5])
            opening_nodes.append(node)

    wall.children.append(CompositionNode(
        name=f"{name}_face", position=[0, base_lift, 0],
        gaussians=stone_wall_face(length, height, courses, STONE, stone_color,
                                  gap_center=gap, openings=openings)))
    wall.children.extend(opening_nodes)
    wall.children.append(CompositionNode(
        name=f"{name}_crenellation", position=[0, height + STONE, 0],
        color=stone_color, gaussians=crenellation_strip(length, STONE, stone_color)))
    if is_gate:
        # arch lintel over the void
        arch_color = [0.4, 0.3, 0.2]
        arch = []
        for i in range(7):
            t = i / 6
            ax = (t - 0.5) * gap * 2
            ay = base_lift - height * 0.5 + height * 0.18 + math.sin(t * math.pi) * STONE * 2
            arch.append(GaussianParams(position=[ax, ay, 0],
                        scale=_stone_log_scale(STONE), opacity=2.0, color=arch_color))
        wall.children.append(CompositionNode(
            name=f"{name}_arch", color=arch_color, gaussians=arch))
    return wall


def build_keep(courses: int, stone_color) -> CompositionNode:
    """Central square keep: taller than the walls, stone faces + roof."""
    keep = CompositionNode(name="keep", position=[0, 0, 0], scale=1.0)
    height = courses * STONE * 1.6
    side = 0.5
    faces = CompositionNode(name="keep_body")
    # four faces. stone_wall_face centres courses on y=0 (spans +/-height/2);
    # lift by height/2 so the keep is BASE-ALIGNED like a tower (base at y=0),
    # otherwise its lower half sinks below the ground.
    for (px, pz, rot, ln) in [
        (0, side / 2, None, side), (0, -side / 2, None, side),
        (side / 2, 0, [0.7071, 0, 0.7071, 0], side),
        (-side / 2, 0, [0.7071, 0, 0.7071, 0], side),
    ]:
        faces.gaussians.extend(
            _placed(stone_wall_face(ln, height, courses, STONE, stone_color),
                    dx=px, dz=pz, rot=rot, dy=height / 2))
    keep.children.append(faces)
    keep.children.append(CompositionNode(
        name="keep_roof", position=[0, height + STONE, 0], color=[0.45, 0.2, 0.13],
        gaussians=cone_roof(side * 0.8, side * 0.9, STONE, [0.45, 0.2, 0.13])))
    return keep


def _placed(gaussians, dx=0.0, dz=0.0, rot=None, dy=0.0):
    """Translate (and optionally yaw 90deg) a list of gaussians in-place copy."""
    out = []
    yaw = rot is not None
    for g in gaussians:
        x, y, z = g.position
        if yaw:  # swap x,z for a 90deg face
            x, z = z, x
        out.append(GaussianParams(position=[x + dx, y + dy, z + dz], scale=g.scale,
                                  opacity=g.opacity, color=g.color))
    return out


def build_tree(name: str, pos, scale: float) -> CompositionNode:
    trunk, canopy = tree_parts(0.25, 0.22, STONE)
    t = CompositionNode(name=name, position=pos, scale=scale)
    t.children.append(CompositionNode(name=f"{name}_trunk", color=[0.4, 0.26, 0.13],
                                      gaussians=trunk))
    t.children.append(CompositionNode(name=f"{name}_canopy", position=[0, 0.25, 0],
                                      color=[0.12, 0.45, 0.15], gaussians=canopy))
    return t


# ── Additional atomic elements (Raum 1.6 compositional catalog) ───────

def recessed_box(w: float, h: float, d: float, color, n: int = 40):
    """A small filled box of dark stones — a recessed opening (door/window)."""
    out = []
    nx = max(2, int(w / STONE)); ny = max(2, int(h / STONE))
    for ix in range(nx):
        for iy in range(ny):
            x = (ix / max(nx - 1, 1) - 0.5) * w
            y = (iy / max(ny - 1, 1) - 0.5) * h
            out.append(GaussianParams(position=[x, y, -d], scale=_stone_log_scale(STONE),
                                      opacity=2.0, color=_shift(color, 0.02)))
    return out


def build_door(name: str = "door", pos=None, color=None) -> CompositionNode:
    """A wooden door: a tall recessed dark panel with an arched top."""
    wood = color or [0.32, 0.2, 0.1]
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0, color=wood)
    node.gaussians = recessed_box(0.22, 0.34, STONE, wood)
    # arched lintel
    for i in range(7):
        t = i / 6
        node.gaussians.append(GaussianParams(
            position=[(t - 0.5) * 0.22, 0.17 + math.sin(t * math.pi) * STONE * 1.5, 0],
            scale=_stone_log_scale(STONE), opacity=2.0, color=_shift([0.45, 0.3, 0.2], 0.02)))
    return node


def build_window(name: str = "window", pos=None, color=None, arched: bool = True) -> CompositionNode:
    """A window: a small recessed dark opening, optionally arched."""
    dark = color or [0.1, 0.1, 0.13]
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0, color=dark)
    node.gaussians = recessed_box(0.14, 0.2, STONE, dark)
    if arched:
        for i in range(5):
            t = i / 4
            node.gaussians.append(GaussianParams(
                position=[(t - 0.5) * 0.14, 0.1 + math.sin(t * math.pi) * STONE, 0],
                scale=_stone_log_scale(STONE), opacity=2.0, color=_shift([0.55, 0.5, 0.45], 0.02)))
    return node


def build_arrow_slit(name: str = "arrow_slit", pos=None, color=None) -> CompositionNode:
    """A tall thin recessed slit for archers."""
    dark = color or [0.08, 0.08, 0.1]
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0, color=dark)
    node.gaussians = recessed_box(STONE * 1.2, 0.26, STONE, dark)
    return node


def build_arch(name: str = "arch", pos=None, color=None) -> CompositionNode:
    """A freestanding stone arch: two piers + a curved span."""
    stone = color or [0.62, 0.58, 0.53]
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0, color=stone)
    g = []
    # two piers
    for side in (-1, 1):
        for c in range(6):
            g.append(GaussianParams(position=[side * 0.2, c * STONE * 1.5 - 0.15, 0],
                     scale=_stone_log_scale(STONE), opacity=2.0, color=_shift(stone, 0.03)))
    # curved span
    for i in range(11):
        t = i / 10
        g.append(GaussianParams(position=[(t - 0.5) * 0.4, 0.3 + math.sin(t * math.pi) * 0.18, 0],
                 scale=_stone_log_scale(STONE), opacity=2.0, color=_shift(stone, 0.03)))
    node.gaussians = g
    return node


def build_gatehouse(name: str = "gatehouse", pos=None, courses: int = 6,
                    stone_color=None) -> CompositionNode:
    """A gate flanked by two short towers."""
    stone = stone_color or [0.62, 0.58, 0.53]
    gh = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0)
    gh.children.append(build_wall(f"{name}_wall", [0, 0, 0], 0.9, courses, stone, is_gate=True))
    gh.children.append(build_tower(f"{name}_tower_L", [-0.5, 0, 0], courses + 1, 0.15, stone))
    gh.children.append(build_tower(f"{name}_tower_R", [0.5, 0, 0], courses + 1, 0.15, stone))
    return gh


def build_cliff(name: str = "cliff", pos=None, color=None) -> CompositionNode:
    """A steep rocky cliff face (taller, blockier, greyer than a hill)."""
    rock = color or [0.45, 0.42, 0.4]
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0, color=rock)
    g = []
    # stacked irregular rock shelves
    for c in range(10):
        y = c * STONE * 2.2
        r = 0.9 * (1 - c / 14)
        n_around = max(6, int(2 * math.pi * r / (STONE * 2)))
        for i in range(n_around):
            theta = 2 * math.pi * i / n_around
            jit = random.uniform(-STONE, STONE)
            g.append(GaussianParams(position=[(r + jit) * math.cos(theta), y,
                                              (r + jit) * math.sin(theta)],
                     scale=_stone_log_scale(STONE * 2), opacity=2.0, color=_shift(rock, 0.06)))
    node.gaussians = g
    return node


def build_rock(name: str = "rock", pos=None, color=None, scale: float = 1.0) -> CompositionNode:
    """A boulder: a small irregular sphere of stones."""
    rock = color or [0.5, 0.47, 0.43]
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=scale, color=rock)
    golden = (1 + math.sqrt(5)) / 2
    r = 0.18
    g = []
    for i in range(24):
        theta = 2 * math.pi * i / golden
        phi = math.acos(1 - 2 * (i + 0.5) / 24)
        jit = random.uniform(-0.03, 0.03)
        g.append(GaussianParams(
            position=[(r + jit) * math.sin(phi) * math.cos(theta),
                      (r + jit) * math.cos(phi), (r + jit) * math.sin(phi) * math.sin(theta)],
            scale=_stone_log_scale(STONE * 1.5), opacity=2.0, color=_shift(rock, 0.05)))
    node.gaussians = g
    return node


def build_woods(name: str = "woods", pos=None, n_trees: int = 7,
                rng: random.Random = None) -> CompositionNode:
    """A small cluster of trees."""
    rng = rng or random
    node = CompositionNode(name=name, position=pos or [0, 0, 0], scale=1.0)
    for i in range(n_trees):
        ang = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.1, 0.7)
        node.children.append(build_tree(
            f"{name}_tree_{i}", [rr * math.cos(ang), 0, rr * math.sin(ang)],
            rng.uniform(0.6, 1.0)))
    return node


def build_square_tower(name: str, pos, courses: int, stone_color,
                       with_roof: bool = True) -> CompositionNode:
    """Square tower: four stone faces + crenellation + optional pyramidal roof."""
    tower = CompositionNode(name=name, position=pos, scale=1.0)
    height = courses * STONE * 1.6
    side = 0.34
    body = CompositionNode(name=f"{name}_body", color=stone_color)
    for (px, pz, rot) in [(0, side / 2, None), (0, -side / 2, None),
                          (side / 2, 0, [0.7071, 0, 0.7071, 0]),
                          (-side / 2, 0, [0.7071, 0, 0.7071, 0])]:
        for c in range(courses):
            y = (c / max(courses - 1, 1)) * height
            face = stone_wall_face(side, STONE * 1.6, 1, STONE, stone_color)
            body.gaussians.extend(_placed(
                [GaussianParams(position=[gp.position[0], y, gp.position[2]],
                                scale=gp.scale, opacity=gp.opacity, color=gp.color)
                 for gp in face], dx=px, dz=pz, rot=rot))
    tower.children.append(body)
    tower.children.append(CompositionNode(
        name=f"{name}_crenellation", position=[0, height, 0], color=stone_color,
        gaussians=crenellation_strip(side, STONE, stone_color)))
    if with_roof:
        tower.children.append(CompositionNode(
            name=f"{name}_roof", position=[0, height + STONE, 0], color=[0.5, 0.2, 0.12],
            gaussians=cone_roof(side * 0.8, side * 1.4, STONE, [0.5, 0.2, 0.12])))
    return tower


# ── Scene assembly ────────────────────────────────────────────────────

def build_castle_on_hill(towers: int = 4, wall_courses: int = 6,
                         with_gate: bool = True, trees: int = 8,
                         hill_radius: float = 1.6, rng: random.Random = None
                         ) -> CompositionNode:
    """Assemble a full castle-on-a-hill tree following the spatial rules."""
    rng = rng or random
    stone = [rng.uniform(0.58, 0.68), rng.uniform(0.55, 0.62), rng.uniform(0.5, 0.58)]

    ring = 0.7          # half-width of the square wall ring
    hill_h = 0.55
    castle_scale = 0.9
    # The hill must comfortably contain the castle footprint with a grass skirt.
    # Footprint half-width = (ring + tower_radius) * castle_scale; give ~1.9x so
    # towers sit well inside the dome rather than spilling over its edge.
    footprint = (ring + 0.26) * castle_scale
    hill_radius = max(hill_radius, footprint * 1.9)
    # Seat the castle base ON the hill surface, not sunk into it. The dome height
    # at horizontal radius r is hill_h * sqrt(hill_radius**2 - r**2), minus the
    # hill node's own -0.1 y-offset. Seat at the corner-tower radius so all four
    # towers rest on the ground; walls/keep (nearer the centre, where the dome is
    # higher) then sit at-or-above the surface. The old flat `hill_h*hill_radius
    # *0.45` ignored dome curvature and put the base ~0.25 below the surface ->
    # walls and keep buried (the "sunk castle" bug, 2026-06-05 screenshots).
    corner_r = ring * castle_scale * math.sqrt(2.0)
    castle_y = hill_h * math.sqrt(max(hill_radius ** 2 - corner_r ** 2, 0.0)) - 0.1

    castle = CompositionNode(name="castle", position=[0, castle_y, 0], scale=castle_scale)

    # 4 corner towers (or fewer if requested)
    corners = [(-ring, -ring), (ring, -ring), (ring, ring), (-ring, ring)]
    names = ["tower_SW", "tower_SE", "tower_NE", "tower_NW"]
    for i in range(min(towers, 4)):
        cx, cz = corners[i]
        castle.children.append(build_tower(
            names[i], [cx, 0, cz], wall_courses + 1, 0.26, stone))

    # walls between adjacent towers; front (south, -Z toward viewer) is the gate
    wall_specs = [
        ("wall_S", [0, 0, -ring], None, with_gate),          # front -> gate
        ("wall_N", [0, 0, ring], None, False),
        ("wall_E", [ring, 0, 0], [0.7071, 0, 0.7071, 0], False),
        ("wall_W", [-ring, 0, 0], [0.7071, 0, 0.7071, 0], False),
    ]
    for nm, pos, rot, gate in wall_specs:
        # non-gate walls get a couple of arrow-slits; the back wall a window
        slits = 0 if gate else 2
        windows = 1 if (nm == "wall_N" and not gate) else 0
        castle.children.append(build_wall(
            nm, pos, ring * 2, wall_courses, stone, rot=rot, is_gate=gate,
            windows=windows, slits=slits))

    # keep, centered, taller
    castle.children.append(build_keep(wall_courses + 3, stone))

    # hill
    grass = [rng.uniform(0.25, 0.35), rng.uniform(0.5, 0.62), rng.uniform(0.15, 0.25)]
    hill = CompositionNode(name="hill", position=[0, -0.1, 0], scale=1.0,
                           color=grass,
                           gaussians=dome(hill_radius, hill_h, STONE * 2, grass))

    scene = CompositionNode(name="scene", position=[0, 0, 0], scale=1.0)
    scene.children.append(hill)
    scene.children.append(castle)

    # trees on the hill skirt, outside the wall ring, below the castle base
    for i in range(trees):
        ang = (i / max(trees, 1)) * 2 * math.pi + rng.uniform(-0.3, 0.3)
        rr = rng.uniform(hill_radius * 0.55, hill_radius * 0.85)
        tx, tz = rr * math.cos(ang), rr * math.sin(ang)
        ty = hill_h * hill_radius * 0.8 * (1 - (rr / hill_radius) ** 2) - 0.15
        scene.children.append(build_tree(f"tree_{i}", [tx, ty, tz],
                                         rng.uniform(0.6, 1.0)))
    return scene


def build_lone_tower(rng: random.Random = None) -> CompositionNode:
    """'a tower' -> one tower on a small patch of ground. Prompt-faithful."""
    rng = rng or random
    stone = [rng.uniform(0.58, 0.68), rng.uniform(0.55, 0.62), rng.uniform(0.5, 0.58)]
    grass = [0.3, 0.55, 0.2]
    scene = CompositionNode(name="scene", position=[0, 0, 0], scale=1.0)
    scene.children.append(CompositionNode(
        name="ground", position=[0, -0.1, 0], color=grass,
        gaussians=dome(0.7, 0.15, STONE * 2, grass)))
    scene.children.append(build_tower("tower", [0, 0.1, 0], 9, 0.3, stone))
    return scene


PRESETS = {
    "castle_on_hill": build_castle_on_hill,
    "tower": lambda **kw: build_lone_tower(rng=kw.get("rng")),
}


# ── Part expansion (for inference: shallow skeleton -> atomic compound) ─

def expand_part(name: str, color=None, courses: int = 6,
                rng: random.Random = None) -> CompositionNode | None:
    """
    Expand a shallow part leaf (e.g. "tower_NE", "wall_S", "keep", "hill",
    "tree_3") into its atomic compound CompositionNode using the grammar
    builders. Returns None if the name is not a known part (caller falls
    back to the generic fill).

    This is the inference-side mirror of the grammar: a model that emits a
    shallow skeleton gets the SAME atomic geometry the 0.5 full grammar
    produces. The returned node is in LOCAL frame (position [0,0,0]); the
    caller keeps the skeleton node's own position/scale.
    """
    rng = rng or random
    stone = color or [0.62, 0.58, 0.53]
    kind = _part_kind(name)
    n = name.lower()
    if kind == "tower":
        if "square" in n:
            return build_square_tower(name, [0, 0, 0], courses + 1, stone)
        return build_tower(name, [0, 0, 0], courses + 1, 0.26, stone)
    if kind == "gatehouse":
        return build_gatehouse(name, [0, 0, 0], courses, stone)
    if kind == "wall":
        is_gate = "gate" in n or n.endswith("_s")
        # match the castle assembly: non-gate walls carry slits (+ a window on N)
        slits = 0 if is_gate else 2
        windows = 1 if (n.endswith("_n") and not is_gate) else 0
        return build_wall(name, [0, 0, 0], 1.4, courses, stone, is_gate=is_gate,
                          windows=windows, slits=slits)
    if kind == "gate":
        return build_wall(name, [0, 0, 0], 1.4, courses, stone, is_gate=True)
    if kind == "keep":
        return build_keep(courses + 3, stone)
    if kind == "tree":
        return build_tree(name, [0, 0, 0], 1.0)
    if kind == "woods":
        return build_woods(name, rng=rng)
    if kind == "door":
        return build_door(name)
    if kind == "window":
        return build_window(name, arched="arch" in n)
    if kind in ("arrow_slit", "slit"):
        return build_arrow_slit(name)
    if kind == "arch":
        return build_arch(name)
    if kind == "cliff":
        return build_cliff(name)
    if kind == "rock":
        return build_rock(name)
    return None


# ── Training-data sampling (for Raum 1.5) ─────────────────────────────

PARAPHRASES = {
    "castle_on_hill": [
        "a castle on a hill", "a fortress atop a hill", "a hilltop castle",
        "a castle on a hill with trees", "a stone castle on a green hill",
        "a medieval fortress on a hill surrounded by trees",
    ],
    "tower": ["a tower", "a stone tower", "a lone watchtower", "a tall tower"],
}


# Part names the fill stage knows how to expand into an atomic compound.
# A SHALLOW skeleton stops at these; the fill stage rebuilds their sub-parts.
# Order matters: more specific names first (gatehouse before gate, arrow_slit
# before slit) so the longest match wins.
EXPANDABLE_PARTS = ("gatehouse", "arrow_slit", "slit", "tower", "wall",
                    "keep", "woods", "tree", "gate", "door", "window",
                    "arch", "cliff", "rock")


def _part_kind(name: str) -> str | None:
    """Map a node name to an expandable part kind (tower_NE -> tower)."""
    n = name.lower()
    for kind in EXPANDABLE_PARTS:
        if n == kind or n.startswith(kind + "_") or ("_" + kind) in n:
            return kind
    return None


def skeleton_dict(node: CompositionNode, shallow: bool = True) -> dict:
    """
    Serialize a tree to its STRUCTURAL skeleton: name, position, scale, color.

    shallow=True (default): stop at expandable PARTS (tower/wall/keep/tree),
    dropping their sub-parts. This keeps the JSON inside the model's 512-token
    context. The fill stage re-expands each part into its atomic compound
    (tower -> body + crenellation + roof) at inference, using the SAME grammar
    builders, so the model path and the 0.5 grammar path render identically.

    shallow=False: keep the full nested structure (no Gaussians either way).
    """
    d = {"name": node.name, "position": [round(v, 3) for v in node.position],
         "scale": round(node.scale, 3)}
    if node.color:
        d["color"] = [round(c, 3) for c in node.color]
    # Collapse expandable parts to leaves in shallow mode.
    if shallow and _part_kind(node.name) is not None:
        return d
    if node.children:
        d["children"] = [skeleton_dict(c, shallow) for c in node.children]
    return d


# ── SceneSpec: prompt-conditioned structure (Raum 1.6 accuracy core) ──
#
# A SceneSpec is the single source for BOTH the tree and the prompt text, so
# they can never disagree. This is the fix for "a wall with a gate -> whole
# castle": the structure is determined by the spec, and the prompt describes
# exactly that spec. Counts and presence become learnable signal.

# Numbers as words, for prompt synthesis.
_NUMWORDS = {0: "no", 1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def build_from_spec(spec: dict, rng: random.Random) -> CompositionNode:
    """Build a scene tree from a structured spec.

    spec = {
      "archetype": "single" | "feature" | "composition" | "castle",
      "element": str,            # primary element for single/feature
      "feature": str | None,     # an added element (gate/door/window/...)
      "towers": int, "with_gate": bool, "trees": int,
      "ground": "hill"|"cliff"|"ground"|None, "tower_shape": "round"|"square",
    }
    """
    stone = [rng.uniform(0.58, 0.68), rng.uniform(0.55, 0.62), rng.uniform(0.5, 0.58)]
    grass = [rng.uniform(0.25, 0.35), rng.uniform(0.5, 0.62), rng.uniform(0.15, 0.25)]
    arch = spec["archetype"]

    if arch == "castle":
        return build_castle_on_hill(
            towers=spec.get("towers", 4), wall_courses=rng.choice([5, 6, 7]),
            with_gate=spec.get("with_gate", True), trees=spec.get("trees", 6),
            hill_radius=rng.uniform(1.3, 1.9), rng=rng)

    scene = CompositionNode(name="scene", position=[0, 0, 0], scale=1.0)
    # ground for standalone elements
    gtype = spec.get("ground")
    if gtype == "hill":
        scene.children.append(CompositionNode(name="hill", position=[0, -0.1, 0],
                              color=grass, gaussians=dome(1.2, 0.5, STONE * 2, grass)))
    elif gtype == "cliff":
        scene.children.append(build_cliff("cliff", [0, -0.3, 0]))
    elif gtype == "ground":
        scene.children.append(CompositionNode(name="ground", position=[0, -0.1, 0],
                              color=grass, gaussians=dome(0.8, 0.15, STONE * 2, grass)))

    elem = spec["element"]
    feature = spec.get("feature")
    y = 0.1
    # primary element
    def _tower():
        if spec.get("tower_shape") == "square":
            return build_square_tower("tower", [0, y, 0], 10, stone)
        return build_tower("tower", [0, y, 0], 9, 0.3, stone)

    builders = {
        "tower": _tower,
        "wall": lambda: build_wall("wall", [0, y, 0], 1.4, 6, stone,
                                   is_gate=(feature == "gate")),
        "gate": lambda: build_wall("gate_wall", [0, y, 0], 1.4, 6, stone, is_gate=True),
        "gatehouse": lambda: build_gatehouse("gatehouse", [0, y, 0], 6, stone),
        "keep": lambda: build_keep(9, stone),
        "arch": lambda: build_arch("arch", [0, y, 0]),
        "door": lambda: build_door("door", [0, y, 0]),
        "window": lambda: build_window("window", [0, y, 0]),
        "arrow_slit": lambda: build_arrow_slit("arrow_slit", [0, y, 0]),
        "tree": lambda: build_tree("tree", [0, y, 0], 1.0),
        "woods": lambda: build_woods("woods", [0, y, 0], rng.choice([5, 7, 9]), rng=rng),
        "rock": lambda: build_rock("rock", [0, y, 0]),
        "cliff": lambda: build_cliff("cliff", [0, y, 0]),
    }
    node = builders.get(elem, builders["tower"])()
    scene.children.append(node)

    # a feature added to the primary element (e.g. tower + gate, wall + window)
    # walls embed the gate themselves; others are placed adjacent
    if feature and not (elem == "wall" and feature == "gate"):
        feat_builders = {
            "gate": lambda: build_wall("gate", [0, y - 0.1, 0.3], 0.8, 5, stone, is_gate=True),
            "door": lambda: build_door("door", [0, y - 0.05, 0.25]),
            "window": lambda: build_window("window", [0, y + 0.2, 0.25]),
            "arrow_slit": lambda: build_arrow_slit("arrow_slit", [0, y + 0.1, 0.25]),
        }
        if feature in feat_builders:
            scene.children.append(feat_builders[feature]())
    return scene


def spec_to_prompts(spec: dict, rng: random.Random, paraphrase: bool) -> list[str]:
    """Synthesize prompt phrasings that DESCRIBE the spec exactly."""
    arch = spec["archetype"]
    out = []
    if arch == "castle":
        t = spec.get("towers", 4)
        base = "a castle" if spec.get("ground") != "cliff" else "a castle on a cliff"
        if spec.get("ground") == "hill":
            base = "a castle on a hill"
        variants = [base, "a fortress" + (" on a hill" if spec.get("ground") == "hill" else ""),
                    "a hilltop castle" if spec.get("ground") == "hill" else "a stone castle"]
        if t in (2, 3):
            variants = [f"a castle with {_NUMWORDS[t]} towers",
                        f"a small fort with {_NUMWORDS[t]} towers", base]
        if spec.get("trees", 0) == 0:
            variants = [v + " without trees" for v in variants[:2]] + [base]
        elif spec.get("trees", 0) >= 6:
            variants = [base + " with trees", base + " surrounded by woods", base]
        out = variants
    else:
        elem = spec["element"].replace("_", " ")
        article = "an" if elem[0] in "aeiou" else "a"
        feature = spec.get("feature")
        if feature:
            f = feature.replace("_", " ")
            fart = "an" if f[0] in "aeiou" else "a"
            out = [f"{article} {elem} with {fart} {f}",
                   f"{article} {elem} and {fart} {f}",
                   f"{article} stone {elem} with {fart} {f}"]
        else:
            adj = {"tower": ["stone", "tall", "round"], "wall": ["stone", "long"],
                   "keep": ["stone", "tall"], "woods": ["dense", "small"]}.get(spec["element"], ["stone"])
            a2 = rng.choice(adj)
            art2 = "an" if a2[0] in "aeiou" else "a"
            out = [f"{article} {elem}", f"{art2} {a2} {elem}"]
            if spec["element"] == "woods":
                out = ["woods", "a forest", "a cluster of trees"]
    # dedup, optionally trim
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    k = min(len(uniq), 3) if paraphrase else 1
    return rng.sample(uniq, k) if len(uniq) > k else uniq


# Elements that can stand alone, with whether they take a "ground".
_STANDALONE = ["tower", "wall", "gate", "gatehouse", "keep", "arch", "door",
               "window", "arrow_slit", "tree", "woods", "rock", "cliff"]
_FEATURE_HOSTS = ["tower", "wall", "keep", "gatehouse"]
_FEATURES = ["gate", "door", "window", "arrow_slit"]


def sample_spec(rng: random.Random) -> dict:
    """Sample one SceneSpec across the combinatorial archetype space."""
    arch = rng.choices(
        ["single", "feature", "castle"],
        weights=[0.4, 0.3, 0.3])[0]
    if arch == "castle":
        return {"archetype": "castle",
                "towers": rng.choice([2, 3, 4, 4, 4]),
                "with_gate": rng.random() > 0.15,
                "trees": rng.choice([0, 0, 4, 6, 8, 10]),
                "ground": rng.choice(["hill", "hill", "cliff", "ground"]),
                "tower_shape": rng.choice(["round", "round", "square"])}
    if arch == "feature":
        host = rng.choice(_FEATURE_HOSTS)
        return {"archetype": "feature", "element": host,
                "feature": rng.choice(_FEATURES),
                "ground": rng.choice([None, "ground", "hill"]),
                "tower_shape": rng.choice(["round", "square"])}
    return {"archetype": "single", "element": rng.choice(_STANDALONE),
            "feature": None, "ground": rng.choice([None, "ground", "hill"]),
            "tower_shape": rng.choice(["round", "square"])}


def sample_training_set(n: int, domains: list[str], paraphrase: bool,
                        out_dir: Path, seed: int = 0, conditioned: bool = False):
    """Emit n randomized labeled trees as {prompt, tree} records (skeletons).

    conditioned=True (Raum 1.6): use the SceneSpec system -- structure and
    prompt come from the same spec, covering the full element catalog and
    count/presence grid. This is the accuracy fix.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    over_512 = 0

    if conditioned:
        for i in range(n):
            spec = sample_spec(rng)
            tree = build_from_spec(spec, rng)
            skel = skeleton_dict(tree)
            for p in spec_to_prompts(spec, rng, paraphrase):
                js = json.dumps(skel, separators=(",", ":"))
                if len(js) > 1400:   # ~512 tokens; log+drop, never truncate
                    over_512 += 1
                    continue
                records.append({"prompt": p, "tree": skel})
    else:
        for i in range(n):
            domain = domains[i % len(domains)]
            builder = PRESETS.get(domain, build_castle_on_hill)
            kw = dict(rng=rng)
            if domain == "castle_on_hill" or domain in ("castle", "hill", "village", "keep"):
                builder = build_castle_on_hill
                kw.update(towers=rng.choice([3, 4, 4, 4]),
                          wall_courses=rng.choice([5, 6, 7]),
                          with_gate=rng.random() > 0.15,
                          trees=rng.choice([0, 4, 6, 8, 10]),
                          hill_radius=rng.uniform(1.3, 1.9))
            tree = builder(**kw)
            prompts = PARAPHRASES.get(
                "castle_on_hill" if builder is build_castle_on_hill else domain,
                [domain.replace("_", " ")])
            chosen = rng.sample(prompts, k=min(len(prompts), 3) if paraphrase else 1)
            skel = skeleton_dict(tree)
            for p in chosen:
                records.append({"prompt": p, "tree": skel})

    rng.shuffle(records)
    split = int(len(records) * 0.9)
    (out_dir / "train.json").write_text(json.dumps(records[:split]))
    (out_dir / "val.json").write_text(json.dumps(records[split:]))
    print(f"Wrote {split} train / {len(records) - split} val records to {out_dir}")
    print(f"  conditioned={conditioned} paraphrase={paraphrase} specs={n}")
    if over_512:
        print(f"  dropped {over_512} records over ~512 tokens (logged, not truncated)")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Raum 0.5 castle scene grammar")
    # single-scene mode
    p.add_argument("--preset", choices=list(PRESETS), default="castle_on_hill")
    p.add_argument("--towers", type=int, default=4)
    p.add_argument("--wall-courses", type=int, default=6)
    p.add_argument("--with-gate", action="store_true", default=False)
    p.add_argument("--trees", type=int, default=8)
    p.add_argument("--hill-radius", type=float, default=1.6)
    p.add_argument("--output", default="output/castle_05.json")
    p.add_argument("--seed", type=int, default=0)
    # training-set mode
    p.add_argument("--sample", type=int, default=0,
                   help="If >0, generate this many training scenes instead of one")
    p.add_argument("--domain", default="castle_on_hill",
                   help="Comma-separated domains for --sample")
    p.add_argument("--paraphrase", action="store_true", default=False)
    p.add_argument("--conditioned", action="store_true", default=False,
                   help="Raum 1.6: SceneSpec system (prompt describes structure, "
                        "full element catalog + count/presence grid)")
    p.add_argument("--output-dir", default="data/decomposition_trees/castle_15")
    args = p.parse_args()

    if args.sample > 0:
        sample_training_set(args.sample, [d.strip() for d in args.domain.split(",")],
                            args.paraphrase, Path(args.output_dir), args.seed,
                            conditioned=args.conditioned)
        return

    rng = random.Random(args.seed)
    if args.preset == "castle_on_hill":
        scene = build_castle_on_hill(
            towers=args.towers, wall_courses=args.wall_courses,
            with_gate=args.with_gate, trees=args.trees,
            hill_radius=args.hill_radius, rng=rng)
    else:
        scene = PRESETS[args.preset](rng=rng)

    print_tree(scene)
    tensors = tree_to_tensors(scene)
    print(f"\n  Total Gaussians: {tensors['means'].shape[0]}")
    print(f"  Tree depth: {scene.depth}  |  direct children: {len(scene.children)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tree(scene, out)
    print(f"  Saved scene tree: {out}")


if __name__ == "__main__":
    main()
