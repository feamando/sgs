"""
Hand-authored decomposition: "a castle on a hill"

This is the reference scene for Raum 1.3 Phase 1. The tree is
manually constructed to demonstrate recursive decomposition from
a natural language prompt down to individual Gaussian splats.

Hierarchy:
    scene
    ├── castle (elevated on hill)
    │   ├── tower_NW (cylinder + cone roof + flag)
    │   ├── tower_NE (cylinder + cone roof + flag)
    │   ├── gate (arch shape)
    │   ├── keep (large block + battlements)
    │   └── wall_N (flat slab connecting towers)
    └── hill (dome/mound shape)

Run standalone to generate the scene JSON:
    python data/scenes/castle_on_hill.py
"""

import math
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
        theta = 2 * math.pi * (i * 7) / n  # spiral
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
    import random
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


def make_dome(n: int = 150, radius: float = 1.0,
              color: list[float] = [0.3, 0.6, 0.2]) -> list[GaussianParams]:
    """Dome/hemisphere: points on upper half of sphere."""
    gaussians = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = 2.0 * math.pi * i / golden
        # Only upper hemisphere
        phi = math.acos(1.0 - (i + 0.5) / n)
        if phi > math.pi / 2:
            phi = math.pi - phi  # mirror to upper half
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi) * 0.5  # flatten vertically
        z = radius * math.sin(phi) * math.sin(theta)
        scale_val = math.log(0.06 * radius)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


def make_flag(n: int = 15, color: list[float] = [0.8, 0.1, 0.1]) -> list[GaussianParams]:
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


# ── Build the castle scene ────────────────────────────────────────────

def build_castle_on_hill() -> CompositionNode:
    """Construct the full 'castle on a hill' composition tree."""

    # Tower template (reused for NW and NE)
    def make_tower(name: str, pos: list[float], stone_color: list[float]):
        tower = CompositionNode(
            name=name,
            position=pos,
            scale=0.3,
        )
        # Cylinder body
        tower.children.append(CompositionNode(
            name=f"{name}_body",
            position=[0.0, 0.0, 0.0],
            color=stone_color,
            gaussians=make_cylinder(80, radius=0.3, height=1.2, color=stone_color),
        ))
        # Cone roof
        tower.children.append(CompositionNode(
            name=f"{name}_roof",
            position=[0.0, 0.7, 0.0],
            color=[0.5, 0.2, 0.1],
            gaussians=make_cone(40, radius=0.4, height=0.5, color=[0.5, 0.2, 0.1]),
        ))
        # Flag
        tower.children.append(CompositionNode(
            name=f"{name}_flag",
            position=[0.0, 1.0, 0.0],
            scale=0.5,
            gaussians=make_flag(15, color=[0.8, 0.1, 0.1]),
        ))
        return tower

    # Castle
    castle = CompositionNode(
        name="castle",
        position=[0.0, 0.8, 0.0],
        scale=1.0,
    )

    # Towers
    castle.children.append(make_tower("tower_NW", [-0.8, 0.0, 0.6], [0.65, 0.6, 0.55]))
    castle.children.append(make_tower("tower_NE", [0.8, 0.0, 0.6], [0.65, 0.6, 0.55]))
    castle.children.append(make_tower("tower_SW", [-0.8, 0.0, -0.6], [0.6, 0.55, 0.5]))
    castle.children.append(make_tower("tower_SE", [0.8, 0.0, -0.6], [0.6, 0.55, 0.5]))

    # Gate (arch in the front wall)
    gate = CompositionNode(
        name="gate",
        position=[0.0, -0.2, 0.8],
        scale=0.4,
        color=[0.4, 0.3, 0.2],
        gaussians=make_box(60, size_x=0.6, size_y=0.8, size_z=0.15, color=[0.4, 0.3, 0.2]),
    )
    castle.children.append(gate)

    # Keep (central large building)
    keep = CompositionNode(
        name="keep",
        position=[0.0, 0.1, 0.0],
        scale=0.5,
    )
    keep.children.append(CompositionNode(
        name="keep_body",
        position=[0.0, 0.0, 0.0],
        color=[0.6, 0.58, 0.52],
        gaussians=make_box(120, size_x=1.0, size_y=1.2, size_z=0.8, color=[0.6, 0.58, 0.52]),
    ))
    keep.children.append(CompositionNode(
        name="keep_roof",
        position=[0.0, 0.7, 0.0],
        color=[0.45, 0.2, 0.15],
        gaussians=make_cone(60, radius=0.6, height=0.4, color=[0.45, 0.2, 0.15]),
    ))
    castle.children.append(keep)

    # Walls (connecting towers)
    for wall_name, pos, sx in [
        ("wall_N", [0.0, -0.1, 0.6], 1.4),
        ("wall_S", [0.0, -0.1, -0.6], 1.4),
        ("wall_E", [0.8, -0.1, 0.0], 0.15),
        ("wall_W", [-0.8, -0.1, 0.0], 0.15),
    ]:
        wall = CompositionNode(
            name=wall_name,
            position=pos,
            scale=0.3,
            color=[0.6, 0.55, 0.5],
            gaussians=make_box(40, size_x=sx, size_y=0.6, size_z=0.1, color=[0.6, 0.55, 0.5]),
        )
        castle.children.append(wall)

    # Hill
    hill = CompositionNode(
        name="hill",
        position=[0.0, -0.5, 0.0],
        scale=2.0,
        color=[0.3, 0.6, 0.2],
        gaussians=make_dome(200, radius=1.0, color=[0.3, 0.6, 0.2]),
    )

    # Scene root
    scene = CompositionNode(
        name="scene",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    scene.children.append(castle)
    scene.children.append(hill)

    return scene


if __name__ == "__main__":
    scene = build_castle_on_hill()

    print("=== Composition Tree ===")
    print_tree(scene)
    print()

    tensors = tree_to_tensors(scene)
    print(f"=== Flattened ===")
    print(f"  Total Gaussians: {tensors['means'].shape[0]}")
    print(f"  Tree depth: {scene.depth}")
    print(f"  Direct children of scene: {len(scene.children)}")
    print(f"  Castle sub-parts: {len(scene.children[0].children)}")

    # Save
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tree(scene, out_dir / "castle_on_hill.json")
    print(f"\n  Saved to {out_dir / 'castle_on_hill.json'}")

    # Also save flattened tensors for the renderer
    import torch
    torch.save(tensors, out_dir / "castle_on_hill.pt")
    print(f"  Saved tensors to {out_dir / 'castle_on_hill.pt'}")
