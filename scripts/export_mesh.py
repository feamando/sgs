"""Export a Raum scene JSON to a triangle mesh (.obj) via Poisson reconstruction."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors
from src.export.mesh import extract_mesh_poisson


def main():
    parser = argparse.ArgumentParser(description="Export scene JSON to mesh")
    parser.add_argument("--input", required=True, help="Input scene JSON path")
    parser.add_argument("--output", required=True, help="Output .obj path")
    parser.add_argument("--depth", type=int, default=8,
                        help="Poisson octree depth (default 8)")
    parser.add_argument("--density-threshold", type=float, default=0.1,
                        help="Low-density vertex pruning threshold (default 0.1)")
    args = parser.parse_args()

    tree = load_tree(args.input)
    tensors = tree_to_tensors(tree)
    n = tensors["means"].shape[0]
    print(f"Scene has {n} Gaussians")

    stats = extract_mesh_poisson(
        tensors, args.output,
        depth=args.depth,
        density_threshold=args.density_threshold,
    )
    print(f"Mesh: {stats['n_vertices']:,} vertices, {stats['n_faces']:,} faces")
    print(f"Written to {args.output} ({stats['file_size_bytes']:,} bytes)")


if __name__ == "__main__":
    main()
