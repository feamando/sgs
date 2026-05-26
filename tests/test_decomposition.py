"""Tests for src/raum/decomposition.py"""

import json
import math
import tempfile
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import (
    CompositionNode, GaussianParams, _quat_mul,
    load_tree, save_tree, tree_to_tensors, print_tree,
)


def test_gaussian_params_defaults():
    g = GaussianParams(position=[1, 2, 3], scale=[-1, -1, -1], opacity=0.5, color=[1, 0, 0])
    assert g.rotation == [1.0, 0.0, 0.0, 0.0]
    assert g.sh_degree == 0
    assert g.sh_coeffs is None


def test_composition_node_leaf():
    node = CompositionNode(name="ball")
    node.gaussians.append(
        GaussianParams(position=[0, 0, 0], scale=[-2, -2, -2], opacity=1.0, color=[1, 0, 0])
    )
    assert node.is_leaf
    assert node.n_gaussians_recursive == 1
    assert node.depth == 0


def test_composition_node_internal():
    root = CompositionNode(name="scene")
    child1 = CompositionNode(name="a")
    child1.gaussians.append(
        GaussianParams(position=[1, 0, 0], scale=[0, 0, 0], opacity=0.5, color=[0, 1, 0])
    )
    child2 = CompositionNode(name="b")
    child2.gaussians.append(
        GaussianParams(position=[-1, 0, 0], scale=[0, 0, 0], opacity=0.5, color=[0, 0, 1])
    )
    root.children = [child1, child2]
    assert not root.is_leaf
    assert root.n_gaussians_recursive == 2
    assert root.depth == 1


def test_flatten_gaussians_identity():
    node = CompositionNode(name="test", position=[1, 2, 3], scale=1.0)
    node.gaussians.append(
        GaussianParams(position=[0, 0, 0], scale=[0, 0, 0], opacity=1.0, color=[1, 1, 1])
    )
    flat = node.flatten_gaussians()
    assert len(flat) == 1
    assert flat[0].position == [1.0, 2.0, 3.0]
    assert flat[0].rotation == [1.0, 0.0, 0.0, 0.0]


def test_flatten_gaussians_scale_compounds():
    root = CompositionNode(name="root", scale=2.0)
    child = CompositionNode(name="child", position=[1, 0, 0], scale=0.5)
    child.gaussians.append(
        GaussianParams(position=[0, 0, 0], scale=[0, 0, 0], opacity=1.0, color=[1, 0, 0])
    )
    root.children = [child]
    flat = root.flatten_gaussians()
    assert len(flat) == 1
    # Child position = parent_pos + child_pos * parent_scale = [0,0,0] + [1,0,0]*2 = [2,0,0]
    assert abs(flat[0].position[0] - 2.0) < 1e-6
    # Scale compounds: log(parent_scale * child_scale) = log(2*0.5) = log(1) = 0
    expected_scale = 0.0 + math.log(2.0 * 0.5)
    assert abs(flat[0].scale[0] - expected_scale) < 1e-6


def test_quat_mul_identity():
    identity = [1.0, 0.0, 0.0, 0.0]
    q = [0.7071, 0.7071, 0.0, 0.0]
    result = _quat_mul(identity, q)
    for i in range(4):
        assert abs(result[i] - q[i]) < 1e-4


def test_quat_mul_90_rotations():
    # 90 deg around Z, then 90 deg around Z = 180 deg around Z
    q_90z = [math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)]
    q_180z = _quat_mul(q_90z, q_90z)
    # Expected: [cos(pi/2), 0, 0, sin(pi/2)] = [0, 0, 0, 1]
    assert abs(q_180z[0] - 0.0) < 1e-6
    assert abs(q_180z[3] - 1.0) < 1e-6


def test_json_round_trip():
    root = CompositionNode(name="scene", position=[0, 1, 0])
    child = CompositionNode(name="obj", position=[1, 0, 0], scale=0.5)
    child.gaussians.append(
        GaussianParams(
            position=[0.1, 0.2, 0.3], scale=[-1, -1, -1],
            opacity=0.8, color=[1, 0.5, 0],
            rotation=[0.7071, 0.7071, 0, 0],
            sh_degree=0,
        )
    )
    root.children = [child]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    save_tree(root, tmp_path)
    loaded = load_tree(tmp_path)

    assert loaded.name == "scene"
    assert loaded.children[0].name == "obj"
    g = loaded.children[0].gaussians[0]
    assert abs(g.rotation[0] - 0.7071) < 1e-4
    assert abs(g.rotation[1] - 0.7071) < 1e-4

    Path(tmp_path).unlink()


def test_json_backward_compat_euler():
    """Legacy scenes with 3-value Euler rotation should load as identity quaternion."""
    legacy_data = {
        "name": "test",
        "position": [0, 0, 0],
        "scale": 1.0,
        "rotation": [0, 0, 0],
        "gaussians": [
            {"position": [0, 0, 0], "scale": [0, 0, 0], "opacity": 1.0, "color": [1, 1, 1]}
        ]
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(legacy_data, f)
        tmp_path = f.name

    tree = load_tree(tmp_path)
    assert tree.rotation == [1.0, 0.0, 0.0, 0.0]
    assert tree.gaussians[0].rotation == [1.0, 0.0, 0.0, 0.0]

    Path(tmp_path).unlink()


def test_tree_to_tensors_shapes():
    root = CompositionNode(name="scene")
    for i in range(5):
        child = CompositionNode(name=f"obj_{i}")
        child.gaussians.append(
            GaussianParams(position=[i, 0, 0], scale=[0, 0, 0], opacity=0.5, color=[1, 0, 0])
        )
        root.children.append(child)

    t = tree_to_tensors(root)
    assert t["means"].shape == (5, 3)
    assert t["scales_log"].shape == (5, 3)
    assert t["rotations"].shape == (5, 4)
    assert t["opacities"].shape == (5,)
    assert t["colors"].shape == (5, 3)


def test_tree_to_tensors_empty():
    root = CompositionNode(name="empty")
    t = tree_to_tensors(root)
    assert t["means"].shape == (0, 3)
    assert t["rotations"].shape == (0, 4)


def test_castle_scene_loads():
    """Smoke test: the existing castle scene loads and flattens."""
    scene_path = Path(__file__).parent.parent / "data" / "scenes" / "castle_on_hill.json"
    if not scene_path.exists():
        return  # skip if scene not available

    tree = load_tree(str(scene_path))
    t = tree_to_tensors(tree)
    assert t["means"].shape[0] > 100
    assert t["rotations"].shape[0] == t["means"].shape[0]
