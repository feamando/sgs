"""
Build all hand-authored demo scenes.

Runs each scene builder, reports Gaussian counts and tree stats.

Usage:
    python data/scenes/build_all.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.raum.decomposition import tree_to_tensors, print_tree


def build_and_report(name: str, builder_func, out_dir: Path):
    """Build a scene, save outputs, and report stats."""
    start = time.time()
    scene = builder_func()
    tensors = tree_to_tensors(scene)
    elapsed = time.time() - start

    n_gaussians = tensors['means'].shape[0]
    depth = scene.depth

    # Count nodes
    def count_nodes(node):
        c = 1
        for child in node.children:
            c += count_nodes(child)
        return c

    n_nodes = count_nodes(scene)

    # Save
    from src.raum.decomposition import save_tree
    import torch

    save_tree(scene, out_dir / f"{name}.json")
    torch.save(tensors, out_dir / f"{name}.pt")

    return {
        "name": name,
        "gaussians": n_gaussians,
        "depth": depth,
        "nodes": n_nodes,
        "time_ms": elapsed * 1000,
    }


def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import all scene builders
    from data.scenes.castle_on_hill import build_castle_on_hill
    from data.scenes.medieval_village import build_medieval_village
    from data.scenes.pirate_ship import build_pirate_ship
    from data.scenes.space_station import build_space_station
    from data.scenes.dragon_mountain import build_dragon_mountain

    scenes = [
        ("castle_on_hill", build_castle_on_hill),
        ("medieval_village", build_medieval_village),
        ("pirate_ship", build_pirate_ship),
        ("space_station", build_space_station),
        ("dragon_mountain", build_dragon_mountain),
    ]

    print("=" * 60)
    print("  SGS Demo Scene Builder")
    print("=" * 60)
    print()

    results = []
    for name, builder in scenes:
        print(f"  Building: {name}...")
        stats = build_and_report(name, builder, out_dir)
        results.append(stats)
        print(f"    -> {stats['gaussians']} Gaussians, "
              f"depth={stats['depth']}, "
              f"nodes={stats['nodes']}, "
              f"{stats['time_ms']:.1f}ms")

    print()
    print("-" * 60)
    total_gaussians = sum(r["gaussians"] for r in results)
    total_nodes = sum(r["nodes"] for r in results)
    print(f"  Total: {len(results)} scenes, "
          f"{total_gaussians} Gaussians, "
          f"{total_nodes} tree nodes")
    print()
    print(f"  Output directory: {out_dir}")
    print(f"  Files per scene: .json (tree) + .pt (tensors)")
    print("=" * 60)


if __name__ == "__main__":
    main()
