"""
Generate 100 decomposition tree training samples procedurally.

No API key needed. Creates diverse scenes with realistic decompositions
using the same primitive generators as castle_on_hill.py.

Usage:
    python scripts/generate_training_trees.py
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import CompositionNode, GaussianParams, save_tree


# ── Primitives ────────────────────────────────────────────────────────

def sphere_gaussians(n=30, radius=0.4, color=[0.7, 0.7, 0.7]):
    gs = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = 2 * math.pi * i / golden
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.sin(phi) * math.sin(theta)
        z = radius * math.cos(phi)
        s = math.log(0.05 * radius)
        gs.append(GaussianParams([x, y, z], [s, s, s], 2.0, color))
    return gs


def dome_gaussians(n=30, radius=0.5, color=[0.3, 0.6, 0.2]):
    gs = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = 2 * math.pi * i / golden
        phi = math.acos(1.0 - (i + 0.5) / n)
        if phi > math.pi / 2:
            phi = math.pi - phi
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi) * 0.5
        z = radius * math.sin(phi) * math.sin(theta)
        s = math.log(0.06 * radius)
        gs.append(GaussianParams([x, y, z], [s, s, s], 2.0, color))
    return gs


def cylinder_gaussians(n=40, radius=0.3, height=1.0, color=[0.6, 0.6, 0.6]):
    gs = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        y = (i / n) * height - height / 2
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        s = math.log(0.05 * radius)
        gs.append(GaussianParams([x, y, z], [s, s, s], 2.0, color))
    return gs


def cone_gaussians(n=30, radius=0.3, height=0.5, color=[0.5, 0.2, 0.1]):
    gs = []
    for i in range(n):
        t = i / n
        theta = 2 * math.pi * (i * 7) / n
        r = radius * (1 - t)
        y = t * height
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        s = math.log(0.04 * radius)
        gs.append(GaussianParams([x, y, z], [s, s, s], 2.0, color))
    return gs


def box_gaussians(n=40, sx=1.0, sy=1.0, sz=1.0, color=[0.7, 0.7, 0.7]):
    gs = []
    per_face = n // 6
    half = [sx/2, sy/2, sz/2]
    for axis in range(3):
        for sign in [-1.0, 1.0]:
            for _ in range(per_face):
                pos = [random.uniform(-half[0], half[0]),
                       random.uniform(-half[1], half[1]),
                       random.uniform(-half[2], half[2])]
                pos[axis] = sign * half[axis]
                s = math.log(0.04 * max(sx, sy, sz))
                gs.append(GaussianParams(pos, [s, s, s], 2.0, color))
    return gs


def plane_gaussians(n=30, sx=2.0, sz=2.0, color=[0.4, 0.6, 0.3]):
    gs = []
    side = int(math.sqrt(n))
    for i in range(side):
        for j in range(side):
            x = (i / side - 0.5) * sx
            z = (j / side - 0.5) * sz
            y = random.uniform(-0.02, 0.02)
            s = math.log(0.08)
            gs.append(GaussianParams([x, y, z], [s, s, s], 2.0, color))
    return gs


# ── Scene templates ───────────────────────────────────────────────────

def rand_color_shift(base, var=0.1):
    return [min(1, max(0, c + random.uniform(-var, var))) for c in base]


TEMPLATES = {
    "tower": lambda: CompositionNode("tower", scale=0.3, children=[
        CompositionNode("body", gaussians=cylinder_gaussians(40, 0.25, 1.0, rand_color_shift([0.6, 0.55, 0.5]))),
        CompositionNode("roof", position=[0, 0.6, 0], gaussians=cone_gaussians(25, 0.3, 0.4, rand_color_shift([0.5, 0.2, 0.1]))),
    ]),
    "tree_conifer": lambda: CompositionNode("tree", scale=0.5, children=[
        CompositionNode("trunk", gaussians=cylinder_gaussians(20, 0.08, 0.6, [0.4, 0.25, 0.1])),
        CompositionNode("canopy", position=[0, 0.5, 0], gaussians=cone_gaussians(35, 0.4, 0.7, rand_color_shift([0.1, 0.5, 0.1]))),
    ]),
    "tree_round": lambda: CompositionNode("tree", scale=0.5, children=[
        CompositionNode("trunk", gaussians=cylinder_gaussians(15, 0.06, 0.5, [0.35, 0.2, 0.08])),
        CompositionNode("canopy", position=[0, 0.5, 0], gaussians=sphere_gaussians(30, 0.35, rand_color_shift([0.15, 0.55, 0.1]))),
    ]),
    "house": lambda: CompositionNode("house", scale=0.5, children=[
        CompositionNode("walls", gaussians=box_gaussians(40, 0.8, 0.6, 0.6, rand_color_shift([0.7, 0.65, 0.5]))),
        CompositionNode("roof", position=[0, 0.4, 0], gaussians=cone_gaussians(30, 0.5, 0.3, rand_color_shift([0.6, 0.2, 0.15]))),
    ]),
    "rock": lambda: CompositionNode("rock", gaussians=sphere_gaussians(20, 0.2 + random.random() * 0.3, rand_color_shift([0.5, 0.45, 0.4]))),
    "bush": lambda: CompositionNode("bush", gaussians=sphere_gaussians(20, 0.2, rand_color_shift([0.2, 0.5, 0.15]))),
    "water": lambda: CompositionNode("water", gaussians=plane_gaussians(30, 2.0, 2.0, rand_color_shift([0.1, 0.3, 0.7]))),
    "ground": lambda: CompositionNode("ground", gaussians=plane_gaussians(40, 3.0, 3.0, rand_color_shift([0.3, 0.5, 0.2]))),
    "snow_ground": lambda: CompositionNode("ground", gaussians=plane_gaussians(40, 3.0, 3.0, [0.9, 0.92, 0.95])),
    "sand": lambda: CompositionNode("sand", gaussians=plane_gaussians(35, 2.5, 2.5, rand_color_shift([0.8, 0.7, 0.4]))),
    "boat": lambda: CompositionNode("boat", scale=0.4, children=[
        CompositionNode("hull", gaussians=box_gaussians(30, 1.0, 0.3, 0.4, [0.4, 0.25, 0.1])),
        CompositionNode("mast", position=[0, 0.5, 0], gaussians=cylinder_gaussians(15, 0.03, 0.8, [0.5, 0.35, 0.15])),
        CompositionNode("sail", position=[0.1, 0.6, 0], gaussians=box_gaussians(20, 0.4, 0.5, 0.02, [0.95, 0.92, 0.85])),
    ]),
    "chimney": lambda: CompositionNode("chimney", position=[0.2, 0.5, 0], scale=0.2,
        gaussians=cylinder_gaussians(15, 0.15, 0.5, [0.5, 0.3, 0.2])),
    "sphere_obj": lambda c=[0.8, 0.2, 0.2]: CompositionNode("ball", gaussians=sphere_gaussians(25, 0.25, c)),
    "lamp": lambda: CompositionNode("lamp", scale=0.3, children=[
        CompositionNode("post", gaussians=cylinder_gaussians(15, 0.04, 1.0, [0.3, 0.3, 0.3])),
        CompositionNode("light", position=[0, 0.6, 0], gaussians=sphere_gaussians(10, 0.1, [1.0, 0.9, 0.5])),
    ]),
}


def make_scene(prompt, parts):
    """Build a scene from a list of (template_name, position, scale_override) tuples."""
    scene = CompositionNode("scene")
    for name, pos, scale_ov in parts:
        if name in TEMPLATES:
            node = TEMPLATES[name]()
        else:
            node = CompositionNode(name, gaussians=sphere_gaussians(20, 0.3, [0.5, 0.5, 0.5]))
        node.position = pos
        if scale_ov:
            node.scale = scale_ov
        scene.children.append(node)
    return scene


# ── 100 scenes ────────────────────────────────────────────────────────

SCENES = [
    ("a castle on a hill", [("tower", [-0.8, 0.8, 0.5], 0.3), ("tower", [0.8, 0.8, 0.5], 0.3), ("house", [0, 0.7, 0], 0.8), ("ground", [0, -0.5, 0], 2.0)]),
    ("a red barn in a green field", [("house", [0, 0.3, 0], 0.8), ("ground", [0, -0.5, 0], 2.5), ("tree_round", [1.5, 0, 0.5], 0.5)]),
    ("a lighthouse on rocky cliffs", [("tower", [0, 0.5, 0], 0.6), ("rock", [-0.5, -0.3, 0], 1.0), ("rock", [0.5, -0.4, 0.3], 0.8), ("water", [0, -1.0, 1.5], 1.5)]),
    ("a pirate ship on the ocean", [("boat", [0, 0.1, 0], 1.2), ("water", [0, -0.5, 0], 3.0)]),
    ("a treehouse in a tall oak", [("tree_round", [0, 0, 0], 1.5), ("house", [0.1, 1.2, 0], 0.3)]),
    ("a windmill in a tulip field", [("tower", [0, 0.3, 0], 0.5), ("ground", [0, -0.5, 0], 2.5), ("bush", [1.0, -0.3, 0.5], 0.4), ("bush", [-1.0, -0.3, -0.5], 0.4)]),
    ("a bridge over a river", [("box_gaussians", [0, 0, 0], 0.5), ("water", [0, -0.5, 0], 2.0)]),
    ("a campfire in a forest clearing", [("sphere_obj", [0, 0.1, 0], 0.3), ("tree_conifer", [-1.5, 0, 1], 0.7), ("tree_conifer", [1.5, 0, -0.5], 0.8), ("ground", [0, -0.3, 0], 2.0)]),
    ("a snowman in a winter garden", [("sphere_obj", [0, -0.1, 0], 0.5), ("sphere_obj", [0, 0.3, 0], 0.35), ("sphere_obj", [0, 0.55, 0], 0.25), ("snow_ground", [0, -0.5, 0], 2.0)]),
    ("a robot in a factory", [("box_gaussians", [0, 0.3, 0], 0.4), ("sphere_obj", [0, 0.7, 0], 0.2), ("ground", [0, -0.5, 0], 2.0)]),
    ("a dragon on a mountain peak", [("rock", [0, -0.3, 0], 1.5), ("sphere_obj", [0, 0.5, 0], 0.6)]),
    ("a cottage with a chimney", [("house", [0, 0.2, 0], 0.7), ("chimney", [0.3, 0.6, 0], 0.3), ("ground", [0, -0.5, 0], 2.0)]),
    ("a fountain in a courtyard", [("cylinder_obj", [0, 0.2, 0], 0.3), ("water", [0, -0.1, 0], 0.8), ("ground", [0, -0.5, 0], 2.0)]),
    ("a rocket on a launch pad", [("tower", [0, 0.5, 0], 0.4), ("ground", [0, -0.5, 0], 1.5)]),
    ("a sailboat near an island", [("boat", [-0.8, 0, 0], 0.8), ("rock", [1.2, -0.2, 0], 0.8), ("tree_round", [1.3, 0.3, 0], 0.4), ("water", [0, -0.6, 0], 3.0)]),
    ("a temple in the jungle", [("house", [0, 0.3, 0], 0.8), ("tree_conifer", [-1.5, 0.2, 0], 0.8), ("tree_conifer", [1.5, 0.2, 0.5], 0.7), ("ground", [0, -0.5, 0], 2.5)]),
    ("a helicopter on a rooftop", [("house", [0, -0.2, 0], 1.0), ("sphere_obj", [0, 0.5, 0], 0.3)]),
    ("a tower with a clock", [("tower", [0, 0.3, 0], 0.7), ("sphere_obj", [0, 0.8, 0.25], 0.1), ("ground", [0, -0.5, 0], 1.5)]),
    ("a log cabin by a lake", [("house", [-0.5, 0.2, 0], 0.6), ("water", [1.0, -0.4, 0], 1.5), ("tree_conifer", [-1.5, 0.2, 0.5], 0.6), ("ground", [0, -0.5, 0], 2.5)]),
    ("a crane at a construction site", [("tower", [0, 0.8, 0], 0.3), ("ground", [0, -0.5, 0], 2.0), ("box_gaussians", [1.0, -0.2, 0], 0.4)]),
]

# Generate 80 more procedurally
MORE_PROMPTS = [
    "a church on a village green", "a submarine surfacing", "a hot air balloon over fields",
    "a train at a station", "a ferris wheel by the pier", "a greenhouse with plants",
    "a pagoda in a zen garden", "a waterfall into a pool", "a volcano at sunset",
    "a farm with a silo", "a playground with a slide", "a piano on a stage",
    "a telescope on a cliff", "a throne in a hall", "a well in a square",
    "a tent in the desert", "a birdhouse on a pole", "a lantern on a post",
    "a swing from a tree branch", "a mushroom in the woods",
    "a cactus in the desert", "a crown on a cushion", "a sword in stone",
    "a candle on a table", "a globe on a desk", "a chess set on a board",
    "a guitar on a stand", "a trophy on a shelf", "a ship in a bottle",
    "a flower in a vase", "a flag on a flagpole", "a bell in a tower",
    "a lamp on a nightstand", "a painting on a wall", "a window with shutters",
    "a door with an arch", "a gate with pillars", "a fence around a yard",
    "a path through meadows", "a dock on still water",
    "a bench under a tree", "a statue in a park", "a pyramid in sand",
    "a dome on columns", "an arch over a path", "a staircase in a tower",
    "a balcony with flowers", "a weathervane on a barn", "a sundial on a pedestal",
    "an anchor on a beach", "a drawbridge over water", "a turret on a wall",
    "a catapult on grass", "a wagon on a trail", "a windmill by the sea",
    "a covered bridge", "a stone circle on a plain", "a totem pole in forest",
    "a gazebo in a park", "a obelisk in a plaza", "a minaret at dawn",
    "a water tower on a hill", "a grain elevator by tracks", "a fire truck at a station",
    "a biplane in the sky", "a gondola on a canal", "a rickshaw on a street",
    "a yurt on a steppe", "a igloo under aurora", "a houseboat on a river",
    "a treestump with mushrooms", "a beehive on a branch", "a anthill in grass",
    "a scarecrow in a field", "a mailbox at a road", "a phone booth on a corner",
    "a bus stop with bench", "a streetlight at dusk", "a fire hydrant on sidewalk",
    "a manhole in a road", "a park fountain at night",
]

def generate_random_parts(prompt):
    """Generate scene parts based on prompt keywords."""
    parts = []
    words = prompt.lower().split()

    # Always add ground/base
    if any(w in words for w in ["water", "ocean", "sea", "river", "lake", "canal"]):
        parts.append(("water", [0, -0.6, 0], 2.5))
    elif any(w in words for w in ["desert", "sand", "beach"]):
        parts.append(("sand", [0, -0.5, 0], 2.5))
    elif any(w in words for w in ["snow", "winter", "arctic", "igloo"]):
        parts.append(("snow_ground", [0, -0.5, 0], 2.5))
    else:
        parts.append(("ground", [0, -0.5, 0], 2.0))

    # Main object
    if any(w in words for w in ["tower", "lighthouse", "minaret", "obelisk", "pole", "flagpole"]):
        parts.append(("tower", [0, 0.4, 0], 0.5))
    elif any(w in words for w in ["house", "cabin", "cottage", "church", "temple", "station", "hut", "yurt", "booth"]):
        parts.append(("house", [0, 0.2, 0], 0.6))
    elif any(w in words for w in ["boat", "ship", "gondola", "submarine"]):
        parts.append(("boat", [0, 0.1, 0], 0.8))
    elif any(w in words for w in ["tree", "oak", "branch"]):
        parts.append(("tree_round", [0, 0.2, 0], 1.0))
    else:
        parts.append(("sphere_obj", [0, 0.2, 0], 0.5))

    # Secondary objects
    if any(w in words for w in ["tree", "forest", "woods", "jungle"]):
        parts.append(("tree_conifer", [1.2 + random.random(), 0.1, random.uniform(-0.5, 0.5)], 0.6))
        if random.random() > 0.5:
            parts.append(("tree_round", [-1.3, 0.1, random.uniform(-0.5, 0.5)], 0.5))
    if any(w in words for w in ["rock", "cliff", "stone", "mountain"]):
        parts.append(("rock", [random.uniform(-1, 1), -0.2, random.uniform(-0.5, 0.5)], 0.7))
    if any(w in words for w in ["bush", "flower", "plant", "garden", "mushroom"]):
        parts.append(("bush", [random.uniform(0.5, 1.5), -0.3, random.uniform(-0.5, 0.5)], 0.3))
    if any(w in words for w in ["lamp", "lantern", "light", "streetlight"]):
        parts.append(("lamp", [0.8, 0, 0.3], 0.4))

    return parts


def main():
    random.seed(42)

    all_samples = []

    # First 20 hand-designed
    for prompt, parts in SCENES:
        scene = make_scene(prompt, parts)
        all_samples.append({"prompt": prompt, "tree": scene.to_dict()})

    # Remaining 80 procedural
    for prompt in MORE_PROMPTS:
        parts = generate_random_parts(prompt)
        scene = make_scene(prompt, parts)
        all_samples.append({"prompt": prompt, "tree": scene.to_dict()})

    # Save
    out_path = Path("data/decomposition_trees/train.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_samples, f, indent=2)

    # Stats
    total_gaussians = 0
    for s in all_samples:
        tree = CompositionNode.from_dict(s["tree"])
        total_gaussians += tree.n_gaussians_recursive

    print(f"Generated {len(all_samples)} scenes")
    print(f"Total gaussians across all scenes: {total_gaussians:,}")
    print(f"Avg gaussians per scene: {total_gaussians // len(all_samples)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
