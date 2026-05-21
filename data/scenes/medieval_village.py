"""
Hand-authored decomposition: "a medieval village"

Hierarchy:
    scene
    ├── ground (cobblestone plane)
    ├── houses (3 houses with peaked roofs)
    │   ├── house_1 (box body + cone roof + chimney)
    │   ├── house_2 (box body + cone roof)
    │   └── house_3 (box body + cone roof + chimney)
    ├── church (tall tower with spire + nave)
    │   ├── nave (box body + peaked roof)
    │   └── tower (cylinder + cone spire + bell)
    └── well (cylinder wall + cone roof + bucket)

Run standalone:
    python data/scenes/medieval_village.py
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


def make_dome(n: int = 150, radius: float = 1.0,
              color: list[float] = [0.3, 0.6, 0.2]) -> list[GaussianParams]:
    """Dome/hemisphere: points on upper half of sphere."""
    gaussians = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = 2.0 * math.pi * i / golden
        phi = math.acos(1.0 - (i + 0.5) / n)
        if phi > math.pi / 2:
            phi = math.pi - phi
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi) * 0.5
        z = radius * math.sin(phi) * math.sin(theta)
        scale_val = math.log(0.06 * radius)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


def make_plane(n: int = 200, size_x: float = 5.0, size_z: float = 5.0,
               color: list[float] = [0.4, 0.4, 0.4]) -> list[GaussianParams]:
    """Flat ground plane at y=0."""
    random.seed(99)
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


def make_sphere(n: int = 80, radius: float = 0.3,
                color: list[float] = [0.5, 0.5, 0.5]) -> list[GaussianParams]:
    """Full sphere: Fibonacci lattice."""
    gaussians = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = 2.0 * math.pi * i / golden
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        x = radius * math.sin(phi) * math.cos(theta)
        y = radius * math.cos(phi)
        z = radius * math.sin(phi) * math.sin(theta)
        scale_val = math.log(0.05 * radius)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=2.0,
            color=color,
        ))
    return gaussians


# ── Build the village scene ──────────────────────────────────────────

def build_medieval_village() -> CompositionNode:
    """Construct the full 'medieval village' composition tree."""

    # House template
    def make_house(name: str, pos: list[float], wall_color: list[float],
                   roof_color: list[float], has_chimney: bool = False):
        house = CompositionNode(
            name=name,
            position=pos,
            scale=0.4,
        )
        # Walls
        house.children.append(CompositionNode(
            name=f"{name}_walls",
            position=[0.0, 0.0, 0.0],
            color=wall_color,
            gaussians=make_box(80, size_x=0.8, size_y=0.6, size_z=0.6, color=wall_color),
        ))
        # Roof (cone)
        house.children.append(CompositionNode(
            name=f"{name}_roof",
            position=[0.0, 0.45, 0.0],
            color=roof_color,
            gaussians=make_cone(50, radius=0.55, height=0.4, color=roof_color),
        ))
        # Chimney
        if has_chimney:
            house.children.append(CompositionNode(
                name=f"{name}_chimney",
                position=[0.25, 0.5, 0.0],
                scale=0.3,
                color=[0.35, 0.25, 0.2],
                gaussians=make_cylinder(30, radius=0.1, height=0.4, color=[0.35, 0.25, 0.2]),
            ))
        return house

    # Houses group
    houses = CompositionNode(
        name="houses",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    houses.children.append(make_house(
        "house_1", [-1.2, 0.0, 0.8],
        wall_color=[0.75, 0.68, 0.55], roof_color=[0.55, 0.25, 0.15], has_chimney=True
    ))
    houses.children.append(make_house(
        "house_2", [0.0, 0.0, 1.2],
        wall_color=[0.8, 0.72, 0.6], roof_color=[0.5, 0.22, 0.12]
    ))
    houses.children.append(make_house(
        "house_3", [1.3, 0.0, 0.6],
        wall_color=[0.7, 0.65, 0.52], roof_color=[0.48, 0.2, 0.1], has_chimney=True
    ))

    # Church
    church = CompositionNode(
        name="church",
        position=[-0.5, 0.0, -0.8],
        scale=0.6,
    )
    # Nave
    nave = CompositionNode(
        name="church_nave",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    nave.children.append(CompositionNode(
        name="nave_body",
        position=[0.0, 0.0, 0.0],
        color=[0.72, 0.68, 0.6],
        gaussians=make_box(100, size_x=1.2, size_y=0.8, size_z=0.7, color=[0.72, 0.68, 0.6]),
    ))
    nave.children.append(CompositionNode(
        name="nave_roof",
        position=[0.0, 0.55, 0.0],
        color=[0.4, 0.18, 0.1],
        gaussians=make_cone(60, radius=0.7, height=0.35, color=[0.4, 0.18, 0.1]),
    ))
    church.children.append(nave)

    # Tower
    tower = CompositionNode(
        name="church_tower",
        position=[0.7, 0.2, 0.0],
        scale=0.8,
    )
    tower.children.append(CompositionNode(
        name="tower_body",
        position=[0.0, 0.0, 0.0],
        color=[0.7, 0.65, 0.58],
        gaussians=make_cylinder(100, radius=0.25, height=1.8, color=[0.7, 0.65, 0.58]),
    ))
    tower.children.append(CompositionNode(
        name="tower_spire",
        position=[0.0, 1.1, 0.0],
        color=[0.35, 0.35, 0.4],
        gaussians=make_cone(40, radius=0.3, height=0.6, color=[0.35, 0.35, 0.4]),
    ))
    tower.children.append(CompositionNode(
        name="tower_bell",
        position=[0.0, 0.7, 0.0],
        scale=0.3,
        color=[0.7, 0.6, 0.2],
        gaussians=make_sphere(20, radius=0.1, color=[0.7, 0.6, 0.2]),
    ))
    church.children.append(tower)

    # Well
    well = CompositionNode(
        name="well",
        position=[0.8, 0.0, -0.3],
        scale=0.25,
    )
    well.children.append(CompositionNode(
        name="well_wall",
        position=[0.0, 0.0, 0.0],
        color=[0.5, 0.45, 0.4],
        gaussians=make_cylinder(60, radius=0.4, height=0.5, color=[0.5, 0.45, 0.4]),
    ))
    well.children.append(CompositionNode(
        name="well_roof",
        position=[0.0, 0.5, 0.0],
        color=[0.4, 0.2, 0.1],
        gaussians=make_cone(30, radius=0.5, height=0.3, color=[0.4, 0.2, 0.1]),
    ))
    well.children.append(CompositionNode(
        name="well_bucket",
        position=[0.0, 0.1, 0.0],
        scale=0.3,
        color=[0.4, 0.3, 0.15],
        gaussians=make_cylinder(15, radius=0.15, height=0.2, color=[0.4, 0.3, 0.15]),
    ))

    # Ground (cobblestone)
    ground = CompositionNode(
        name="ground",
        position=[0.0, -0.3, 0.0],
        scale=1.0,
        color=[0.45, 0.4, 0.35],
        gaussians=make_plane(250, size_x=6.0, size_z=6.0, color=[0.45, 0.4, 0.35]),
    )

    # Scene root
    scene = CompositionNode(
        name="scene",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    scene.children.append(houses)
    scene.children.append(church)
    scene.children.append(well)
    scene.children.append(ground)

    return scene


if __name__ == "__main__":
    scene = build_medieval_village()

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
    save_tree(scene, out_dir / "medieval_village.json")
    print(f"\n  Saved to {out_dir / 'medieval_village.json'}")

    import torch
    torch.save(tensors, out_dir / "medieval_village.pt")
    print(f"  Saved tensors to {out_dir / 'medieval_village.pt'}")
