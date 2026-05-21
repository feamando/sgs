"""
Hand-authored decomposition: "a pirate ship on the ocean"

Hierarchy:
    scene
    ├── ocean (flat plane, blue)
    └── ship
        ├── hull (elongated box, curved via scaling)
        ├── deck (flat plane on top of hull)
        ├── mast_main (tall cylinder + rectangular sail)
        ├── mast_fore (shorter cylinder + sail)
        ├── crow_nest (cylinder ring at top of main mast)
        ├── bowsprit (angled cylinder at front)
        └── flag (skull flag at top)

Run standalone:
    python data/scenes/pirate_ship.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.raum.decomposition import CompositionNode, GaussianParams, save_tree, print_tree, tree_to_tensors


# ── Primitive generators ──────────────────────────────────────────────

def make_cylinder(n: int = 80, radius: float = 0.3, height: float = 1.0,
                  color: list[float] = [0.6, 0.6, 0.6]) -> list[GaussianParams]:
    """Cylinder: points on surface, axis along Y."""
    gaussians = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        y = (i / n) * height - height / 2
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        scale_val = math.log(0.05 * radius)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


def make_cone(n: int = 50, radius: float = 0.35, height: float = 0.4,
              color: list[float] = [0.5, 0.2, 0.1]) -> list[GaussianParams]:
    """Cone: points on surface, tip at +Y."""
    gaussians = []
    for i in range(n):
        t = i / n
        theta = 2 * math.pi * (i * 7) / n
        r = radius * (1 - t)
        y = t * height
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        scale_val = math.log(0.04 * radius)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


def make_box(n: int = 100, size_x: float = 1.0, size_y: float = 1.0,
             size_z: float = 1.0, color: list[float] = [0.7, 0.7, 0.7]) -> list[GaussianParams]:
    """Box: points on 6 faces."""
    random.seed(42)
    gaussians = []
    per_face = n // 6
    half = [size_x / 2, size_y / 2, size_z / 2]
    for axis in range(3):
        for sign in [-1.0, 1.0]:
            for _ in range(per_face):
                pos = [random.uniform(-half[0], half[0]),
                       random.uniform(-half[1], half[1]),
                       random.uniform(-half[2], half[2])]
                pos[axis] = sign * half[axis]
                scale_val = math.log(0.04 * max(size_x, size_y, size_z))
                gaussians.append(GaussianParams(
                    position=pos,
                    scale=[scale_val, scale_val, scale_val],
                    opacity=2.0,
                    color=color,
                ))
    return gaussians


def make_plane(n: int = 200, size_x: float = 5.0, size_z: float = 5.0,
               color: list[float] = [0.4, 0.4, 0.4]) -> list[GaussianParams]:
    """Flat ground plane at y=0."""
    random.seed(77)
    gaussians = []
    for _ in range(n):
        x = random.uniform(-size_x / 2, size_x / 2)
        z = random.uniform(-size_z / 2, size_z / 2)
        scale_val = math.log(0.08)
        gaussians.append(GaussianParams(
            position=[x, 0.0, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


def make_flag(n: int = 15, color: list[float] = [0.1, 0.1, 0.1]) -> list[GaussianParams]:
    """Small flag: thin vertical pole + rectangular cloth."""
    gaussians = []
    # Pole
    for i in range(5):
        y = i * 0.06
        gaussians.append(GaussianParams(
            position=[0.0, y, 0.0],
            scale=[-4.0, -3.5, -4.0],
            opacity=2.0,
            color=[0.4, 0.3, 0.2],
        ))
    # Cloth
    for i in range(n - 5):
        x = (i % 4) * 0.03 + 0.02
        y = 0.25 + (i // 4) * 0.03
        gaussians.append(GaussianParams(
            position=[x, y, 0.0],
            scale=[-3.5, -3.8, -4.5],
            opacity=1.5,
            color=color,
        ))
    return gaussians


def make_sail(n: int = 80, width: float = 0.8, height: float = 1.0,
              color: list[float] = [0.92, 0.9, 0.82]) -> list[GaussianParams]:
    """Rectangular sail: slightly billowed plane."""
    random.seed(55)
    gaussians = []
    for i in range(n):
        x = random.uniform(-width / 2, width / 2)
        y = random.uniform(-height / 2, height / 2)
        # Slight billow in z
        z = 0.05 * math.sin(math.pi * (x / width + 0.5))
        scale_val = math.log(0.05)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


# ── Build the pirate ship scene ──────────────────────────────────────

def build_pirate_ship() -> CompositionNode:
    """Construct the full 'pirate ship on the ocean' composition tree."""

    # Ship
    ship = CompositionNode(
        name="ship",
        position=[0.0, 0.3, 0.0],
        scale=1.0,
    )

    # Hull (elongated box with dark wood color)
    hull = CompositionNode(
        name="hull",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    hull.children.append(CompositionNode(
        name="hull_body",
        position=[0.0, 0.0, 0.0],
        color=[0.35, 0.2, 0.1],
        gaussians=make_box(150, size_x=2.5, size_y=0.6, size_z=0.8, color=[0.35, 0.2, 0.1]),
    ))
    # Bow (pointed front, cone tipped sideways approximation)
    hull.children.append(CompositionNode(
        name="hull_bow",
        position=[1.4, 0.0, 0.0],
        color=[0.3, 0.18, 0.08],
        gaussians=make_cone(40, radius=0.4, height=0.5, color=[0.3, 0.18, 0.08]),
    ))
    ship.children.append(hull)

    # Deck
    ship.children.append(CompositionNode(
        name="deck",
        position=[0.0, 0.32, 0.0],
        scale=0.8,
        color=[0.45, 0.3, 0.15],
        gaussians=make_plane(80, size_x=2.2, size_z=0.7, color=[0.45, 0.3, 0.15]),
    ))

    # Main mast
    mast_main = CompositionNode(
        name="mast_main",
        position=[0.0, 0.8, 0.0],
        scale=0.8,
    )
    mast_main.children.append(CompositionNode(
        name="mast_main_pole",
        position=[0.0, 0.0, 0.0],
        color=[0.4, 0.25, 0.12],
        gaussians=make_cylinder(80, radius=0.06, height=2.0, color=[0.4, 0.25, 0.12]),
    ))
    mast_main.children.append(CompositionNode(
        name="mast_main_sail",
        position=[0.0, 0.2, 0.15],
        color=[0.92, 0.9, 0.82],
        gaussians=make_sail(100, width=1.0, height=1.2, color=[0.92, 0.9, 0.82]),
    ))
    ship.children.append(mast_main)

    # Fore mast (shorter)
    mast_fore = CompositionNode(
        name="mast_fore",
        position=[0.7, 0.6, 0.0],
        scale=0.6,
    )
    mast_fore.children.append(CompositionNode(
        name="mast_fore_pole",
        position=[0.0, 0.0, 0.0],
        color=[0.4, 0.25, 0.12],
        gaussians=make_cylinder(50, radius=0.05, height=1.5, color=[0.4, 0.25, 0.12]),
    ))
    mast_fore.children.append(CompositionNode(
        name="mast_fore_sail",
        position=[0.0, 0.1, 0.12],
        color=[0.9, 0.88, 0.8],
        gaussians=make_sail(70, width=0.7, height=0.9, color=[0.9, 0.88, 0.8]),
    ))
    ship.children.append(mast_fore)

    # Crow's nest (ring at top of main mast)
    ship.children.append(CompositionNode(
        name="crow_nest",
        position=[0.0, 1.7, 0.0],
        scale=0.2,
        color=[0.35, 0.22, 0.1],
        gaussians=make_cylinder(30, radius=0.3, height=0.2, color=[0.35, 0.22, 0.1]),
    ))

    # Bowsprit (angled pole at front)
    ship.children.append(CompositionNode(
        name="bowsprit",
        position=[1.5, 0.4, 0.0],
        scale=0.4,
        color=[0.4, 0.25, 0.12],
        gaussians=make_cylinder(25, radius=0.04, height=1.0, color=[0.4, 0.25, 0.12]),
    ))

    # Jolly Roger flag
    ship.children.append(CompositionNode(
        name="flag",
        position=[0.0, 1.9, 0.0],
        scale=0.6,
        gaussians=make_flag(18, color=[0.1, 0.1, 0.1]),
    ))

    # Ocean surface
    ocean = CompositionNode(
        name="ocean",
        position=[0.0, -0.2, 0.0],
        scale=1.0,
        color=[0.1, 0.3, 0.6],
        gaussians=make_plane(300, size_x=8.0, size_z=8.0, color=[0.1, 0.3, 0.6]),
    )

    # Scene root
    scene = CompositionNode(
        name="scene",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    scene.children.append(ship)
    scene.children.append(ocean)

    return scene


if __name__ == "__main__":
    scene = build_pirate_ship()

    print("=== Composition Tree ===")
    print_tree(scene)
    print()

    tensors = tree_to_tensors(scene)
    print(f"=== Flattened ===")
    print(f"  Total Gaussians: {tensors['means'].shape[0]}")
    print(f"  Tree depth: {scene.depth}")
    print(f"  Direct children of scene: {len(scene.children)}")

    # Save
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tree(scene, out_dir / "pirate_ship.json")
    print(f"\n  Saved to {out_dir / 'pirate_ship.json'}")

    import torch
    torch.save(tensors, out_dir / "pirate_ship.pt")
    print(f"  Saved tensors to {out_dir / 'pirate_ship.pt'}")
