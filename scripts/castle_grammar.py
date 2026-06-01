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
    """Log-scale for a single stone roughly stone_size across."""
    v = math.log(max(stone_size * 0.5, 1e-3))
    return [v, v, v]


def stone_course(n_around: int, radius: float, y: float, stone_size: float,
                 color, jitter: float = 0.0) -> list[GaussianParams]:
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
            opacity=2.0, color=_shift(color, 0.04),
        ))
    return out


def stone_wall_face(length: float, height: float, courses: int, stone_size: float,
                    color, gap_center: float = 0.0) -> list[GaussianParams]:
    """
    A flat wall built from courses of stones along X, stacked in Y.

    gap_center > 0 omits stones within that half-width of x=0 (the gate void).
    """
    out = []
    n_along = max(2, int(length / stone_size))
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
            out.append(GaussianParams(
                position=[x, y, 0.0], scale=_stone_log_scale(stone_size),
                opacity=2.0, color=_shift(color, 0.04),
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
            max(6, int(2 * math.pi * radius / STONE)), radius, y, STONE, stone_color))
    tower.children.append(body)
    tower.children.append(CompositionNode(
        name=f"{name}_crenellation", position=[0, height, 0],
        color=stone_color, gaussians=crenellation_ring(radius, STONE, stone_color)))
    if with_roof:
        tower.children.append(CompositionNode(
            name=f"{name}_roof", position=[0, height + STONE, 0],
            color=[0.5, 0.2, 0.12],
            gaussians=cone_roof(radius * 1.1, radius * 1.8, STONE, [0.5, 0.2, 0.12])))
    return tower


def build_wall(name: str, pos, length: float, courses: int, stone_color,
               rot=None, is_gate: bool = False) -> CompositionNode:
    """Wall segment: stone face + crenellation top. Gate omits center stones."""
    height = courses * STONE * 1.6
    wall = CompositionNode(name=name, position=pos, scale=1.0,
                           rotation=rot or [1.0, 0.0, 0.0, 0.0])
    gap = length * 0.22 if is_gate else 0.0
    wall.children.append(CompositionNode(
        name=f"{name}_face", color=stone_color,
        gaussians=stone_wall_face(length, height, courses, STONE, stone_color,
                                  gap_center=gap)))
    wall.children.append(CompositionNode(
        name=f"{name}_crenellation", position=[0, height / 2 + STONE, 0],
        color=stone_color, gaussians=crenellation_strip(length, STONE, stone_color)))
    if is_gate:
        # arch lintel over the void
        arch_color = [0.4, 0.3, 0.2]
        arch = []
        for i in range(7):
            t = i / 6
            ax = (t - 0.5) * gap * 2
            ay = -height * 0.5 + height * 0.18 + math.sin(t * math.pi) * STONE * 2
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
    faces = CompositionNode(name="keep_body", color=stone_color)
    # four faces
    for (px, pz, rot, ln) in [
        (0, side / 2, None, side), (0, -side / 2, None, side),
        (side / 2, 0, [0.7071, 0, 0.7071, 0], side),
        (-side / 2, 0, [0.7071, 0, 0.7071, 0], side),
    ]:
        faces.gaussians.extend(
            _placed(stone_wall_face(ln, height, courses, STONE, stone_color),
                    dx=px, dz=pz, rot=rot))
    keep.children.append(faces)
    keep.children.append(CompositionNode(
        name="keep_roof", position=[0, height / 2 + STONE, 0], color=[0.45, 0.2, 0.13],
        gaussians=cone_roof(side * 0.8, side * 0.9, STONE, [0.45, 0.2, 0.13])))
    return keep


def _placed(gaussians, dx=0.0, dz=0.0, rot=None):
    """Translate (and optionally yaw 90deg) a list of gaussians in-place copy."""
    out = []
    yaw = rot is not None
    for g in gaussians:
        x, y, z = g.position
        if yaw:  # swap x,z for a 90deg face
            x, z = z, x
        out.append(GaussianParams(position=[x + dx, y, z + dz], scale=g.scale,
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
    castle_y = hill_h * hill_radius * 0.8  # sit castle on the dome top

    castle = CompositionNode(name="castle", position=[0, castle_y, 0], scale=0.9)

    # 4 corner towers (or fewer if requested)
    corners = [(-ring, -ring), (ring, -ring), (ring, ring), (-ring, ring)]
    names = ["tower_SW", "tower_SE", "tower_NE", "tower_NW"]
    for i in range(min(towers, 4)):
        cx, cz = corners[i]
        castle.children.append(build_tower(
            names[i], [cx, 0, cz], wall_courses + 2, 0.18, stone))

    # walls between adjacent towers; front (south, -Z toward viewer) is the gate
    wall_specs = [
        ("wall_S", [0, 0, -ring], None, with_gate),          # front -> gate
        ("wall_N", [0, 0, ring], None, False),
        ("wall_E", [ring, 0, 0], [0.7071, 0, 0.7071, 0], False),
        ("wall_W", [-ring, 0, 0], [0.7071, 0, 0.7071, 0], False),
    ]
    for nm, pos, rot, gate in wall_specs:
        castle.children.append(build_wall(
            nm, pos, ring * 2, wall_courses, stone, rot=rot, is_gate=gate))

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
    scene.children.append(build_tower("tower", [0, 0.1, 0], 10, 0.22, stone))
    return scene


PRESETS = {
    "castle_on_hill": build_castle_on_hill,
    "tower": lambda **kw: build_lone_tower(rng=kw.get("rng")),
}


# ── Training-data sampling (for Raum 1.5) ─────────────────────────────

PARAPHRASES = {
    "castle_on_hill": [
        "a castle on a hill", "a fortress atop a hill", "a hilltop castle",
        "a castle on a hill with trees", "a stone castle on a green hill",
        "a medieval fortress on a hill surrounded by trees",
    ],
    "tower": ["a tower", "a stone tower", "a lone watchtower", "a tall tower"],
}


def sample_training_set(n: int, domains: list[str], paraphrase: bool,
                        out_dir: Path, seed: int = 0):
    """Emit n randomized labeled trees as {prompt, tree} records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    for i in range(n):
        domain = domains[i % len(domains)]
        builder = PRESETS.get(domain, build_castle_on_hill)
        # randomized structural params per sample
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
        for p in chosen:
            records.append({"prompt": p, "tree": tree.to_dict()})
    rng.shuffle(records)
    split = int(len(records) * 0.9)
    (out_dir / "train.json").write_text(json.dumps(records[:split]))
    (out_dir / "val.json").write_text(json.dumps(records[split:]))
    print(f"Wrote {split} train / {len(records) - split} val records to {out_dir}")
    print(f"  domains={domains} paraphrase={paraphrase} scenes={n}")


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
    p.add_argument("--output-dir", default="data/decomposition_trees/castle_15")
    args = p.parse_args()

    if args.sample > 0:
        sample_training_set(args.sample, [d.strip() for d in args.domain.split(",")],
                            args.paraphrase, Path(args.output_dir), args.seed)
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
