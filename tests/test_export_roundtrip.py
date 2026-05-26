"""End-to-end export round-trip tests."""

import tempfile
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, save_tree, tree_to_tensors
from src.export.ply import write_ply, read_ply
from src.export.splat import write_splat


def test_scene_json_to_ply_roundtrip():
    """JSON scene -> tensors -> .ply -> read back -> positions match."""
    scene_path = Path(__file__).parent.parent / "data" / "scenes" / "castle_on_hill.json"
    if not scene_path.exists():
        return

    tree = load_tree(str(scene_path))
    tensors = tree_to_tensors(tree)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ply_path = f.name

    write_ply(tensors, ply_path)
    loaded = read_ply(ply_path)

    # Positions must match exactly
    pos_diff = (tensors["means"] - loaded["means"]).abs().max().item()
    assert pos_diff < 1e-5, f"Position mismatch: {pos_diff}"

    # Opacities must match exactly
    opa_diff = (tensors["opacities"] - loaded["opacities"]).abs().max().item()
    assert opa_diff < 1e-5, f"Opacity mismatch: {opa_diff}"

    # Colors (through SH DC) should be very close
    color_diff = (tensors["colors"] - loaded["colors"]).abs().max().item()
    assert color_diff < 0.01, f"Color mismatch: {color_diff}"

    Path(ply_path).unlink()


def test_subdivided_scene_export():
    """Subdivided scene -> export -> valid file."""
    from scripts.subdivide_scene import subdivide_tree

    scene_path = Path(__file__).parent.parent / "data" / "scenes" / "castle_on_hill.json"
    if not scene_path.exists():
        return

    tree = load_tree(str(scene_path))
    tree = subdivide_tree(tree, n_children=4)  # small for speed
    tensors = tree_to_tensors(tree)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ply_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        splat_path = f.name

    write_ply(tensors, ply_path)
    write_splat(tensors, splat_path)

    n = tensors["means"].shape[0]
    assert Path(ply_path).stat().st_size > 0
    assert Path(splat_path).stat().st_size == n * 32

    # Read .ply back
    loaded = read_ply(ply_path)
    assert loaded["means"].shape[0] == n

    Path(ply_path).unlink()
    Path(splat_path).unlink()


def test_densified_scene_export():
    """Densified scene -> export -> valid file."""
    from src.raum.densify import DensifyConfig, densify_loop

    tensors = {
        "means": torch.randn(30, 3),
        "scales_log": torch.randn(30, 3) - 2,
        "rotations": torch.tensor([[1, 0, 0, 0]] * 30, dtype=torch.float32),
        "opacities": torch.ones(30),
        "colors": torch.rand(30, 3),
    }

    config = DensifyConfig(grad_threshold=0.0001, max_gaussians=100)
    dense = densify_loop(tensors, n_iterations=5, config=config)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ply_path = f.name

    write_ply(dense, ply_path)
    loaded = read_ply(ply_path)
    assert loaded["means"].shape[0] == dense["means"].shape[0]

    Path(ply_path).unlink()
