"""Export a Raum scene JSON to compressed .splat format for web viewers."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors
from src.export.splat import write_splat


def main():
    parser = argparse.ArgumentParser(description="Export scene JSON to .splat")
    parser.add_argument("--input", required=True, help="Input scene JSON path")
    parser.add_argument("--output", required=True, help="Output .splat path")
    args = parser.parse_args()

    tree = load_tree(args.input)
    tensors = tree_to_tensors(tree)
    n = tensors["means"].shape[0]

    write_splat(tensors, args.output)
    print(f"Exported {n} Gaussians to {args.output} "
          f"({Path(args.output).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
