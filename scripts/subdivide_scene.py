"""
Apply trained subdivision MLP to a Raum 1.3 scene, expanding leaf Gaussians
into template-based sub-objects.

Without a trained model/templates, uses a procedural fallback that splits
each Gaussian into a small cluster (useful for testing the pipeline).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import (
    CompositionNode, GaussianParams, load_tree, save_tree, tree_to_tensors
)


def load_template_for_name(name: str, templates_dir: Path) -> list[list[float]] | None:
    """
    Try to find a template matching this node name from the template library.

    Searches category directories for a match (exact or partial).
    Returns positions as list of [x,y,z] or None if not found.
    """
    if not templates_dir or not templates_dir.exists():
        return None

    import torch

    name_lower = name.lower().strip()

    # Direct category match
    for cat_dir in templates_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name.lower()
        if cat_name in name_lower or name_lower in cat_name:
            # Find first model.pt in this category
            for obj_dir in sorted(cat_dir.iterdir()):
                model_path = obj_dir / "model.pt"
                if model_path.exists():
                    data = torch.load(model_path, map_location="cpu", weights_only=True)
                    positions = data["positions"].tolist()
                    return positions

    # Partial keyword match
    keywords = {
        "tower": ["tower", "turret", "spire", "minaret"],
        "wall": ["wall", "fence", "barrier", "fortification"],
        "rock": ["rock", "stone", "boulder", "cliff", "mountain", "hill"],
        "tree": ["tree", "oak", "pine", "forest", "wood"],
        "gate": ["gate", "door", "entrance", "arch", "portal"],
        "roof": ["roof", "top", "chimney"],
        "floor": ["floor", "ground", "path", "road"],
        "bush": ["bush", "shrub", "hedge", "grass", "vegetation"],
        "stairs": ["stairs", "step", "staircase"],
        "column": ["column", "pillar", "post"],
        "brick": ["brick", "block"],
    }

    for cat, words in keywords.items():
        if any(w in name_lower for w in words):
            cat_dir = templates_dir / cat
            if cat_dir.exists():
                for obj_dir in sorted(cat_dir.iterdir()):
                    model_path = obj_dir / "model.pt"
                    if model_path.exists():
                        data = torch.load(model_path, map_location="cpu", weights_only=True)
                        return data["positions"].tolist()

    return None


def template_subdivide(gaussians: list[GaussianParams], template_positions: list[list[float]]) -> list[GaussianParams]:
    """
    Subdivide using a real template shape instead of a sphere.

    Scales and positions the template's point cloud relative to each parent Gaussian.
    """
    import numpy as np

    tpl = np.array(template_positions)
    # Normalize template to unit sphere
    center = tpl.mean(axis=0)
    tpl = tpl - center
    extent = np.abs(tpl).max() + 1e-6
    tpl = tpl / extent

    # Subsample template to reasonable count per parent
    max_per_parent = 20
    if len(tpl) > max_per_parent:
        idx = np.linspace(0, len(tpl) - 1, max_per_parent, dtype=int)
        tpl = tpl[idx]

    result = []
    for g in gaussians:
        parent_scale = sum(math.exp(s) for s in g.scale) / 3.0
        spread = parent_scale * 0.4

        for pos in tpl:
            child = GaussianParams(
                position=[
                    g.position[0] + pos[0] * spread,
                    g.position[1] + pos[1] * spread,
                    g.position[2] + pos[2] * spread,
                ],
                scale=[s - 0.7 for s in g.scale],
                opacity=g.opacity,
                color=g.color,
                rotation=g.rotation,
            )
            result.append(child)

    return result


def procedural_subdivide(gaussians: list[GaussianParams], n_children: int = 12) -> list[GaussianParams]:
    """
    Procedural subdivision fallback: expand each Gaussian into a small cluster.

    Creates n_children Gaussians in a sphere around the parent, with smaller
    scale and inherited color. Used when no template is available.
    """
    result = []
    golden = (1.0 + math.sqrt(5.0)) / 2.0

    for g in gaussians:
        parent_scale = sum(math.exp(s) for s in g.scale) / 3.0

        for i in range(n_children):
            # Fibonacci sphere distribution
            theta = 2.0 * math.pi * i / golden
            phi = math.acos(1.0 - 2.0 * (i + 0.5) / n_children)
            dx = math.sin(phi) * math.cos(theta) * parent_scale * 0.3
            dy = math.sin(phi) * math.sin(theta) * parent_scale * 0.3
            dz = math.cos(phi) * parent_scale * 0.3

            child = GaussianParams(
                position=[g.position[0] + dx, g.position[1] + dy, g.position[2] + dz],
                scale=[s - 0.7 for s in g.scale],  # smaller
                opacity=g.opacity,
                color=g.color,
                rotation=g.rotation,
            )
            result.append(child)

    return result


_templates_dir: Path | None = None


def set_templates_dir(path: Path | None):
    """Set the global templates directory for subdivision."""
    global _templates_dir
    _templates_dir = path


def subdivide_tree(tree: CompositionNode, n_children: int = 12) -> CompositionNode:
    """
    Subdivide all leaf Gaussians in the tree.

    Uses template library if available (matches node name to architecture scans).
    Falls back to procedural sphere distribution for unmatched nodes.
    """
    if tree.is_leaf and tree.gaussians:
        template = load_template_for_name(tree.name, _templates_dir) if _templates_dir else None
        if template:
            tree.gaussians = template_subdivide(tree.gaussians, template)
        else:
            tree.gaussians = procedural_subdivide(tree.gaussians, n_children)
    else:
        for child in tree.children:
            subdivide_tree(child, n_children)
    return tree


def main():
    parser = argparse.ArgumentParser(description="Subdivide a Raum scene")
    parser.add_argument("--input", required=True, help="Input scene JSON")
    parser.add_argument("--output", required=True, help="Output subdivided scene JSON")
    parser.add_argument("--subdivider", default=None,
                        help="Trained subdivider checkpoint (optional, uses procedural if not provided)")
    parser.add_argument("--n-children", type=int, default=12,
                        help="Procedural: children per parent Gaussian (default 12)")
    args = parser.parse_args()

    tree = load_tree(args.input)
    tensors_before = tree_to_tensors(tree)
    n_before = tensors_before["means"].shape[0]

    if args.subdivider and Path(args.subdivider).exists():
        print(f"Using trained subdivider: {args.subdivider}")
        print("(trained model inference not yet implemented, falling back to procedural)")
        tree = subdivide_tree(tree, args.n_children)
    else:
        print(f"Using procedural subdivision (n_children={args.n_children})")
        tree = subdivide_tree(tree, args.n_children)

    save_tree(tree, args.output)
    tensors_after = tree_to_tensors(tree)
    n_after = tensors_after["means"].shape[0]

    print(f"Subdivided: {n_before} -> {n_after} Gaussians ({n_after/n_before:.1f}x)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
