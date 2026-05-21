"""
Hand-authored decomposition: "a space station"

Hierarchy:
    scene
    ├── station
    │   ├── core_module (central cylinder)
    │   ├── hab_module_1 (cylinder, connected via tube)
    │   ├── hab_module_2 (cylinder, opposite side)
    │   ├── lab_module (cylinder, perpendicular axis)
    │   ├── connector_tube_1 (thin cylinder between core and hab_1)
    │   ├── connector_tube_2 (thin cylinder between core and hab_2)
    │   ├── connector_tube_3 (thin cylinder between core and lab)
    │   ├── solar_panel_left (flat plane, blue-black)
    │   └── solar_panel_right (flat plane, blue-black)
    └── stars (sparse small spheres for background)

Run standalone:
    python data/scenes/space_station.py
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
    """Flat panel (solar panel, hull section)."""
    random.seed(88)
    gaussians = []
    for _ in range(n):
        x = random.uniform(-size_x / 2, size_x / 2)
        z = random.uniform(-size_z / 2, size_z / 2)
        scale_val = math.log(0.06)
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


def make_stars(n: int = 60, spread: float = 5.0) -> list[GaussianParams]:
    """Scattered dim points simulating distant stars."""
    random.seed(123)
    gaussians = []
    for _ in range(n):
        x = random.uniform(-spread, spread)
        y = random.uniform(-spread, spread)
        z = random.uniform(-spread, -spread * 0.3)  # behind the station
        brightness = random.uniform(0.6, 1.0)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[-4.5, -4.5, -4.5],
            opacity=1.5,
            color=[brightness, brightness, brightness * 0.95],
        ))
    return gaussians


# ── Build the space station scene ────────────────────────────────────

def build_space_station() -> CompositionNode:
    """Construct the full 'space station' composition tree."""

    station = CompositionNode(
        name="station",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )

    # Core module (central hub)
    station.children.append(CompositionNode(
        name="core_module",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
        color=[0.75, 0.75, 0.78],
        gaussians=make_cylinder(120, radius=0.35, height=1.2, color=[0.75, 0.75, 0.78]),
    ))

    # Habitation module 1 (left side)
    station.children.append(CompositionNode(
        name="hab_module_1",
        position=[-1.5, 0.0, 0.0],
        scale=0.8,
        color=[0.7, 0.72, 0.75],
        gaussians=make_cylinder(100, radius=0.3, height=1.0, color=[0.7, 0.72, 0.75]),
    ))

    # Habitation module 2 (right side)
    station.children.append(CompositionNode(
        name="hab_module_2",
        position=[1.5, 0.0, 0.0],
        scale=0.8,
        color=[0.7, 0.72, 0.75],
        gaussians=make_cylinder(100, radius=0.3, height=1.0, color=[0.7, 0.72, 0.75]),
    ))

    # Lab module (perpendicular, along Z axis)
    station.children.append(CompositionNode(
        name="lab_module",
        position=[0.0, 0.0, 1.3],
        scale=0.7,
        color=[0.72, 0.7, 0.68],
        gaussians=make_cylinder(80, radius=0.25, height=0.9, color=[0.72, 0.7, 0.68]),
    ))

    # Connector tubes
    station.children.append(CompositionNode(
        name="connector_tube_1",
        position=[-0.75, 0.0, 0.0],
        scale=0.5,
        color=[0.6, 0.6, 0.62],
        gaussians=make_cylinder(40, radius=0.08, height=0.6, color=[0.6, 0.6, 0.62]),
    ))
    station.children.append(CompositionNode(
        name="connector_tube_2",
        position=[0.75, 0.0, 0.0],
        scale=0.5,
        color=[0.6, 0.6, 0.62],
        gaussians=make_cylinder(40, radius=0.08, height=0.6, color=[0.6, 0.6, 0.62]),
    ))
    station.children.append(CompositionNode(
        name="connector_tube_3",
        position=[0.0, 0.0, 0.65],
        scale=0.5,
        color=[0.6, 0.6, 0.62],
        gaussians=make_cylinder(40, radius=0.08, height=0.5, color=[0.6, 0.6, 0.62]),
    ))

    # Solar panels (flat planes, dark blue)
    solar_color = [0.12, 0.15, 0.35]

    solar_left = CompositionNode(
        name="solar_panel_left",
        position=[-2.2, 0.5, 0.0],
        scale=1.0,
    )
    solar_left.children.append(CompositionNode(
        name="solar_panel_left_array",
        position=[0.0, 0.0, 0.0],
        color=solar_color,
        gaussians=make_plane(100, size_x=1.2, size_z=0.4, color=solar_color),
    ))
    solar_left.children.append(CompositionNode(
        name="solar_panel_left_strut",
        position=[0.6, -0.2, 0.0],
        scale=0.3,
        color=[0.5, 0.5, 0.52],
        gaussians=make_cylinder(20, radius=0.03, height=0.5, color=[0.5, 0.5, 0.52]),
    ))
    station.children.append(solar_left)

    solar_right = CompositionNode(
        name="solar_panel_right",
        position=[2.2, 0.5, 0.0],
        scale=1.0,
    )
    solar_right.children.append(CompositionNode(
        name="solar_panel_right_array",
        position=[0.0, 0.0, 0.0],
        color=solar_color,
        gaussians=make_plane(100, size_x=1.2, size_z=0.4, color=solar_color),
    ))
    solar_right.children.append(CompositionNode(
        name="solar_panel_right_strut",
        position=[-0.6, -0.2, 0.0],
        scale=0.3,
        color=[0.5, 0.5, 0.52],
        gaussians=make_cylinder(20, radius=0.03, height=0.5, color=[0.5, 0.5, 0.52]),
    ))
    station.children.append(solar_right)

    # Background stars
    stars = CompositionNode(
        name="stars",
        position=[0.0, 0.0, -3.0],
        scale=1.0,
        gaussians=make_stars(80, spread=5.0),
    )

    # Scene root
    scene = CompositionNode(
        name="scene",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    scene.children.append(station)
    scene.children.append(stars)

    return scene


if __name__ == "__main__":
    scene = build_space_station()

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
    save_tree(scene, out_dir / "space_station.json")
    print(f"\n  Saved to {out_dir / 'space_station.json'}")

    import torch
    torch.save(tensors, out_dir / "space_station.pt")
    print(f"  Saved tensors to {out_dir / 'space_station.pt'}")
