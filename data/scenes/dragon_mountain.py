"""
Hand-authored decomposition: "a dragon perched on a mountain"

Hierarchy:
    scene
    ├── mountain
    │   ├── peak (large cone, rocky grey)
    │   ├── base (dome, darker rock)
    │   └── cave_entrance (dark box inset into mountain)
    └── dragon
        ├── body (elongated sphere, dark green/black)
        ├── neck (cylinder, curving upward)
        ├── head (sphere with cone snout)
        ├── wings (two large flat planes, angled)
        │   ├── wing_left
        │   └── wing_right
        ├── tail (tapering cylinder)
        └── legs (4 short cylinders)

Run standalone:
    python data/scenes/dragon_mountain.py
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


def make_plane(n: int = 200, size_x: float = 5.0, size_z: float = 5.0,
               color: list[float] = [0.4, 0.4, 0.4]) -> list[GaussianParams]:
    """Flat plane at y=0."""
    random.seed(33)
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


def make_wing(n: int = 60, span: float = 1.5, chord: float = 0.6,
              color: list[float] = [0.15, 0.2, 0.1]) -> list[GaussianParams]:
    """Triangular wing membrane: tapers from root to tip."""
    random.seed(44)
    gaussians = []
    for _ in range(n):
        # Span position (0 = root, 1 = tip)
        t = random.uniform(0.0, 1.0)
        # Chord narrows toward tip
        local_chord = chord * (1.0 - 0.7 * t)
        x = t * span
        z = random.uniform(-local_chord / 2, local_chord / 2)
        # Slight droop
        y = -0.05 * t * t * span
        scale_val = math.log(0.05)
        gaussians.append(GaussianParams(
            position=[x, y, z],
            scale=[scale_val, scale_val, scale_val],
            opacity=1.8,
            color=color,
        ))
    return gaussians


# ── Build the dragon mountain scene ──────────────────────────────────

def build_dragon_mountain() -> CompositionNode:
    """Construct the full 'dragon perched on a mountain' composition tree."""

    # Mountain
    mountain = CompositionNode(
        name="mountain",
        position=[0.0, -0.5, 0.0],
        scale=1.5,
    )
    # Peak (large cone)
    mountain.children.append(CompositionNode(
        name="peak",
        position=[0.0, 0.3, 0.0],
        scale=1.0,
        color=[0.45, 0.42, 0.4],
        gaussians=make_cone(180, radius=1.2, height=2.0, color=[0.45, 0.42, 0.4]),
    ))
    # Base (broad dome)
    mountain.children.append(CompositionNode(
        name="base",
        position=[0.0, -0.3, 0.0],
        scale=1.2,
        color=[0.35, 0.32, 0.28],
        gaussians=make_dome(200, radius=1.5, color=[0.35, 0.32, 0.28]),
    ))
    # Cave entrance (dark recessed box)
    mountain.children.append(CompositionNode(
        name="cave_entrance",
        position=[0.0, -0.1, 0.8],
        scale=0.3,
        color=[0.08, 0.06, 0.05],
        gaussians=make_box(40, size_x=0.5, size_y=0.4, size_z=0.2, color=[0.08, 0.06, 0.05]),
    ))

    # Dragon
    dragon = CompositionNode(
        name="dragon",
        position=[0.0, 1.2, 0.2],
        scale=0.5,
    )

    # Body (elongated sphere approximation via box)
    dragon.children.append(CompositionNode(
        name="dragon_body",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
        color=[0.12, 0.22, 0.1],
        gaussians=make_sphere(100, radius=0.5, color=[0.12, 0.22, 0.1]),
    ))

    # Neck (cylinder curving upward)
    dragon.children.append(CompositionNode(
        name="dragon_neck",
        position=[0.4, 0.3, 0.0],
        scale=0.6,
        color=[0.14, 0.24, 0.12],
        gaussians=make_cylinder(50, radius=0.15, height=0.7, color=[0.14, 0.24, 0.12]),
    ))

    # Head
    head = CompositionNode(
        name="dragon_head",
        position=[0.6, 0.6, 0.0],
        scale=0.4,
    )
    head.children.append(CompositionNode(
        name="head_skull",
        position=[0.0, 0.0, 0.0],
        color=[0.15, 0.25, 0.12],
        gaussians=make_sphere(40, radius=0.25, color=[0.15, 0.25, 0.12]),
    ))
    head.children.append(CompositionNode(
        name="head_snout",
        position=[0.2, -0.05, 0.0],
        color=[0.13, 0.2, 0.1],
        gaussians=make_cone(25, radius=0.12, height=0.3, color=[0.13, 0.2, 0.1]),
    ))
    # Horns
    head.children.append(CompositionNode(
        name="horn_left",
        position=[-0.08, 0.15, -0.08],
        scale=0.3,
        color=[0.3, 0.25, 0.15],
        gaussians=make_cone(12, radius=0.05, height=0.2, color=[0.3, 0.25, 0.15]),
    ))
    head.children.append(CompositionNode(
        name="horn_right",
        position=[-0.08, 0.15, 0.08],
        scale=0.3,
        color=[0.3, 0.25, 0.15],
        gaussians=make_cone(12, radius=0.05, height=0.2, color=[0.3, 0.25, 0.15]),
    ))
    dragon.children.append(head)

    # Wings
    wings = CompositionNode(
        name="wings",
        position=[0.0, 0.2, 0.0],
        scale=1.0,
    )
    wings.children.append(CompositionNode(
        name="wing_left",
        position=[0.0, 0.0, -0.3],
        scale=1.0,
        color=[0.1, 0.18, 0.08],
        gaussians=make_wing(80, span=1.8, chord=0.7, color=[0.1, 0.18, 0.08]),
    ))
    # Mirror wing for right side (negative x)
    right_wing_gaussians = make_wing(80, span=1.8, chord=0.7, color=[0.1, 0.18, 0.08])
    for g in right_wing_gaussians:
        g.position[0] = -g.position[0]  # mirror
    wings.children.append(CompositionNode(
        name="wing_right",
        position=[0.0, 0.0, 0.3],
        scale=1.0,
        color=[0.1, 0.18, 0.08],
        gaussians=right_wing_gaussians,
    ))
    dragon.children.append(wings)

    # Tail (tapering cylinder)
    dragon.children.append(CompositionNode(
        name="dragon_tail",
        position=[-0.6, -0.1, 0.0],
        scale=0.7,
        color=[0.12, 0.2, 0.09],
        gaussians=make_cylinder(60, radius=0.12, height=1.2, color=[0.12, 0.2, 0.09]),
    ))

    # Legs (4 short cylinders)
    leg_positions = [
        [0.2, -0.3, -0.2],
        [0.2, -0.3, 0.2],
        [-0.2, -0.3, -0.2],
        [-0.2, -0.3, 0.2],
    ]
    for i, lp in enumerate(leg_positions):
        dragon.children.append(CompositionNode(
            name=f"dragon_leg_{i+1}",
            position=lp,
            scale=0.3,
            color=[0.12, 0.2, 0.1],
            gaussians=make_cylinder(20, radius=0.08, height=0.3, color=[0.12, 0.2, 0.1]),
        ))

    # Scene root
    scene = CompositionNode(
        name="scene",
        position=[0.0, 0.0, 0.0],
        scale=1.0,
    )
    scene.children.append(mountain)
    scene.children.append(dragon)

    return scene


if __name__ == "__main__":
    scene = build_dragon_mountain()

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
    save_tree(scene, out_dir / "dragon_mountain.json")
    print(f"\n  Saved to {out_dir / 'dragon_mountain.json'}")

    import torch
    torch.save(tensors, out_dir / "dragon_mountain.pt")
    print(f"  Saved tensors to {out_dir / 'dragon_mountain.pt'}")
