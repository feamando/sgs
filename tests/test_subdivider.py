"""Tests for src/raum/subdivider.py"""

from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.subdivider import SubdivisionMLP, Template, apply_deformation


def test_mlp_output_shapes():
    mlp = SubdivisionMLP(embed_dim=300, n_categories=50, max_templates_per_category=5)
    B = 8
    logits, deform = mlp(
        torch.randn(B, 3),
        torch.randn(B, 1),
        torch.rand(B, 3),
        torch.randn(B, 300),
        torch.randn(B, 300),
    )
    assert logits.shape == (B, 5)
    assert deform.shape == (B, 13)


def test_mlp_no_nan():
    mlp = SubdivisionMLP()
    logits, deform = mlp(
        torch.randn(4, 3),
        torch.randn(4, 1),
        torch.rand(4, 3),
        torch.randn(4, 300),
        torch.randn(4, 300),
    )
    assert not torch.isnan(logits).any()
    assert not torch.isnan(deform).any()


def test_mlp_gradient_flows():
    mlp = SubdivisionMLP()
    pos = torch.randn(2, 3, requires_grad=True)
    logits, deform = mlp(
        pos,
        torch.randn(2, 1),
        torch.rand(2, 3),
        torch.randn(2, 300),
        torch.randn(2, 300),
    )
    loss = logits.sum() + deform.sum()
    loss.backward()
    assert pos.grad is not None
    assert not torch.isnan(pos.grad).any()


def test_template_creation():
    tpl = Template(
        category="tower",
        template_id=0,
        positions=torch.randn(50, 3),
        scales=torch.randn(50, 3),
        rotations=torch.tensor([[1, 0, 0, 0]] * 50, dtype=torch.float32),
        opacities=torch.zeros(50),
        colors=torch.rand(50, 3),
    )
    assert tpl.n_gaussians == 50


def test_apply_deformation_shapes():
    tpl = Template(
        category="wall",
        template_id=0,
        positions=torch.randn(30, 3),
        scales=torch.randn(30, 3),
        rotations=torch.tensor([[1, 0, 0, 0]] * 30, dtype=torch.float32),
        opacities=torch.zeros(30),
        colors=torch.rand(30, 3),
    )
    deform = torch.randn(13)
    parent_pos = torch.tensor([1.0, 2.0, 3.0])

    result = apply_deformation(tpl, deform, parent_pos, parent_scale=1.0)
    assert result["positions"].shape == (30, 3)
    assert result["scales"].shape == (30, 3)
    assert result["rotations"].shape == (30, 4)
    assert result["opacities"].shape == (30,)
    assert result["colors"].shape == (30, 3)


def test_apply_deformation_colors_clamped():
    tpl = Template(
        category="test",
        template_id=0,
        positions=torch.zeros(5, 3),
        scales=torch.zeros(5, 3),
        rotations=torch.tensor([[1, 0, 0, 0]] * 5, dtype=torch.float32),
        opacities=torch.zeros(5),
        colors=torch.ones(5, 3) * 0.9,
    )
    # Large positive color shift should still clamp to [0, 1]
    deform = torch.zeros(13)
    deform[6:9] = 5.0  # large color shift

    result = apply_deformation(tpl, deform, torch.zeros(3), parent_scale=1.0)
    assert result["colors"].max() <= 1.0
    assert result["colors"].min() >= 0.0


def test_mlp_param_count():
    mlp = SubdivisionMLP(embed_dim=300, n_categories=50, max_templates_per_category=5)
    n_params = sum(p.numel() for p in mlp.parameters())
    # Should be roughly 400-500K (reasonable for fast inference)
    assert 100_000 < n_params < 1_000_000
