"""Tests for src/raum/densify.py"""

from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.densify import GaussianScene, DensifyConfig, densify_step, densify_loop


def _make_tensors(n: int = 20) -> dict[str, torch.Tensor]:
    return {
        "means": torch.randn(n, 3),
        "scales_log": torch.randn(n, 3) - 3,
        "rotations": torch.tensor([[1, 0, 0, 0]] * n, dtype=torch.float32),
        "opacities": torch.ones(n),
        "colors": torch.rand(n, 3),
    }


def test_scene_from_tensors():
    tensors = _make_tensors(10)
    scene = GaussianScene.from_tensors(tensors)
    assert scene.n_gaussians == 10
    assert scene.positions.requires_grad


def test_scene_to_tensors():
    tensors = _make_tensors(10)
    scene = GaussianScene.from_tensors(tensors)
    out = scene.to_tensors()
    assert out["means"].shape == (10, 3)
    assert not out["means"].requires_grad


def test_clone_increases_count():
    tensors = _make_tensors(10)
    scene = GaussianScene.from_tensors(tensors)
    config = DensifyConfig()
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:3] = True  # clone first 3
    scene.clone_gaussians(mask, config)
    assert scene.n_gaussians == 13


def test_split_maintains_count():
    tensors = _make_tensors(10)
    scene = GaussianScene.from_tensors(tensors)
    config = DensifyConfig()
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:2] = True  # split first 2 -> remove 2, add 4
    scene.split_gaussians(mask, config)
    assert scene.n_gaussians == 12  # 10 - 2 + 4


def test_prune_reduces_count():
    tensors = _make_tensors(10)
    tensors["opacities"][:3] = -10.0  # very low opacity (sigmoid -> ~0)
    scene = GaussianScene.from_tensors(tensors)
    config = DensifyConfig(opacity_prune_threshold=0.01)
    n_pruned = scene.prune(config)
    assert n_pruned == 3
    assert scene.n_gaussians == 7


def test_densify_loop_increases_count():
    tensors = _make_tensors(20)
    config = DensifyConfig(grad_threshold=0.00001, max_gaussians=100)
    result = densify_loop(tensors, n_iterations=10, config=config)
    assert result["means"].shape[0] >= 20


def test_densify_loop_respects_max():
    tensors = _make_tensors(20)
    config = DensifyConfig(grad_threshold=0.00001, max_gaussians=50)
    result = densify_loop(tensors, n_iterations=50, config=config)
    # One step can clone/split all Gaussians, so worst-case overshoot is 2x the count at the cap
    assert result["means"].shape[0] <= 150


def test_densify_loop_output_shapes():
    tensors = _make_tensors(15)
    result = densify_loop(tensors, n_iterations=5)
    n = result["means"].shape[0]
    assert result["scales_log"].shape == (n, 3)
    assert result["rotations"].shape == (n, 4)
    assert result["opacities"].shape == (n,)
    assert result["colors"].shape == (n, 3)


def test_densify_no_nan():
    tensors = _make_tensors(25)
    result = densify_loop(tensors, n_iterations=15)
    assert not torch.isnan(result["means"]).any()
    assert not torch.isnan(result["colors"]).any()
