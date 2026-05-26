"""
Run gradient-based densification on a Raum scene.

Takes a scene JSON (typically post-subdivision) and increases Gaussian
density in under-represented areas via clone/split/prune iterations.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, save_tree, tree_to_tensors, CompositionNode, GaussianParams
from src.raum.densify import DensifyConfig, densify_loop


def tensors_to_tree(tensors: dict[str, torch.Tensor], name: str = "densified_scene") -> CompositionNode:
    """Convert flat tensors back into a single-node composition tree."""
    gaussians = []
    n = tensors["means"].shape[0]
    for i in range(n):
        gaussians.append(GaussianParams(
            position=tensors["means"][i].tolist(),
            scale=tensors["scales_log"][i].tolist(),
            opacity=tensors["opacities"][i].item(),
            color=tensors["colors"][i].tolist(),
            rotation=tensors["rotations"][i].tolist(),
        ))

    return CompositionNode(name=name, gaussians=gaussians)


def main():
    parser = argparse.ArgumentParser(description="Densify a Raum scene")
    parser.add_argument("--input", required=True, help="Input scene JSON")
    parser.add_argument("--output", required=True, help="Output densified scene JSON")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of densification iterations (default 200)")
    parser.add_argument("--grad-threshold", type=float, default=0.0002,
                        help="Gradient threshold for clone/split (default 0.0002)")
    parser.add_argument("--max-gaussians", type=int, default=50000,
                        help="Maximum Gaussians (default 50000)")
    args = parser.parse_args()

    tree = load_tree(args.input)
    tensors = tree_to_tensors(tree)
    n_before = tensors["means"].shape[0]
    print(f"Input: {n_before} Gaussians")

    config = DensifyConfig(
        grad_threshold=args.grad_threshold,
        max_gaussians=args.max_gaussians,
    )

    print(f"Running {args.iterations} densification iterations...")
    result = densify_loop(tensors, n_iterations=args.iterations, config=config)
    n_after = result["means"].shape[0]

    # Convert back to tree and save
    output_tree = tensors_to_tree(result)
    save_tree(output_tree, args.output)

    print(f"Densified: {n_before} -> {n_after} Gaussians ({n_after/n_before:.1f}x)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
