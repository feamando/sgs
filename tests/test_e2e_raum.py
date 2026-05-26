"""End-to-end pipeline tests for Raum 1.4."""

import tempfile
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors
from src.raum.densify import DensifyConfig, densify_loop
from src.export.ply import write_ply


def test_full_pipeline_castle():
    """Castle scene -> subdivide -> densify -> export, all under 60s."""
    from scripts.subdivide_scene import subdivide_tree

    scene_path = Path(__file__).parent.parent / "data" / "scenes" / "castle_on_hill.json"
    if not scene_path.exists():
        return

    start = time.time()

    tree = load_tree(str(scene_path))
    t0 = tree_to_tensors(tree)
    n0 = t0["means"].shape[0]

    # Subdivide
    tree = subdivide_tree(tree, n_children=4)
    t1 = tree_to_tensors(tree)
    n1 = t1["means"].shape[0]
    assert n1 > n0

    # Densify (light, for speed)
    config = DensifyConfig(grad_threshold=0.001, max_gaussians=10000)
    t2 = densify_loop(t1, n_iterations=5, config=config)
    n2 = t2["means"].shape[0]
    assert n2 >= n1

    # Export
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ply_path = f.name
    write_ply(t2, ply_path)
    assert Path(ply_path).stat().st_size > 0

    elapsed = time.time() - start
    print(f"Pipeline: {n0} -> {n1} -> {n2} Gaussians in {elapsed:.1f}s")
    assert elapsed < 60, f"Pipeline too slow: {elapsed:.1f}s"

    Path(ply_path).unlink()


def test_pipeline_no_spatial_outliers():
    """After subdivision, no Gaussians should be far from the bounding box."""
    from scripts.subdivide_scene import subdivide_tree

    scene_path = Path(__file__).parent.parent / "data" / "scenes" / "castle_on_hill.json"
    if not scene_path.exists():
        return

    tree = load_tree(str(scene_path))
    t_orig = tree_to_tensors(tree)

    # Original bounding box (with some margin)
    orig_min = t_orig["means"].min(dim=0).values - 2.0
    orig_max = t_orig["means"].max(dim=0).values + 2.0

    tree = load_tree(str(scene_path))
    tree = subdivide_tree(tree, n_children=8)
    t_sub = tree_to_tensors(tree)

    # All subdivided Gaussians should be within expanded bounding box
    within_min = (t_sub["means"] >= orig_min).all()
    within_max = (t_sub["means"] <= orig_max).all()
    assert within_min and within_max, "Spatial outliers detected after subdivision"
