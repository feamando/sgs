"""
Apply trained subdivision MLP to a Raum 1.3 scene, expanding leaf Gaussians
into template-based sub-objects.

Without a trained model/templates, uses a procedural fallback that splits
each Gaussian into a small cluster (useful for testing the pipeline).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import (
    CompositionNode, GaussianParams, load_tree, save_tree, tree_to_tensors
)


def procedural_subdivide(gaussians: list[GaussianParams], n_children: int = 12) -> list[GaussianParams]:
    """
    Procedural subdivision fallback: expand each Gaussian into a small cluster.

    Creates n_children Gaussians in a sphere around the parent, with smaller
    scale and inherited color. Used when no trained model is available.
    """
    import math

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


def subdivide_tree(tree: CompositionNode, n_children: int = 12) -> CompositionNode:
    """
    Subdivide all leaf Gaussians in the tree using procedural expansion.

    Modifies the tree in-place and returns it.
    """
    if tree.is_leaf and tree.gaussians:
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
