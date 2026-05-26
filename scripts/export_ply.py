"""Export a Raum scene JSON to standard 3DGS .ply format."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors
from src.export.ply import write_ply


def main():
    parser = argparse.ArgumentParser(description="Export scene JSON to .ply")
    parser.add_argument("--input", required=True, help="Input scene JSON path")
    parser.add_argument("--output", required=True, help="Output .ply path")
    parser.add_argument("--sh-degree", type=int, default=0,
                        help="SH band count (0=DC only)")
    args = parser.parse_args()

    tree = load_tree(args.input)
    tensors = tree_to_tensors(tree)
    n = tensors["means"].shape[0]

    write_ply(tensors, args.output, sh_degree=args.sh_degree)
    print(f"Exported {n} Gaussians to {args.output} "
          f"({Path(args.output).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
