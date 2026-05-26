"""
Gradient-based Gaussian densification for Raum 1.4.

Implements the clone/split/prune strategy from Kerbl et al. 2023:
- Clone small Gaussians in under-reconstructed areas (high gradient, small scale)
- Split large Gaussians covering too much area (high gradient, large scale)
- Prune low-opacity Gaussians that contribute nothing visible

This operates on a flat list of Gaussians (post-subdivision) and iteratively
increases density where the scene needs more detail.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DensifyConfig:
    """Configuration for the densification loop."""
    grad_threshold: float = 0.0002
    scale_threshold: float = 0.01
    opacity_prune_threshold: float = 0.005
    max_gaussians: int = 50000
    clone_scale_factor: float = 0.8
    split_scale_factor: float = 0.6


class GaussianScene:
    """
    Mutable scene of Gaussians that supports densification operations.

    All parameters are stored as tensors with gradients enabled for
    position (the gradient magnitude drives clone/split decisions).
    """

    def __init__(
        self,
        positions: torch.Tensor,     # [N, 3]
        scales: torch.Tensor,        # [N, 3] log-scale
        rotations: torch.Tensor,     # [N, 4] quaternion
        opacities: torch.Tensor,     # [N] logit
        colors: torch.Tensor,        # [N, 3]
    ):
        self.positions = positions.clone().requires_grad_(True)
        self.scales = scales.clone().requires_grad_(True)
        self.rotations = rotations.clone()
        self.opacities = opacities.clone().requires_grad_(True)
        self.colors = colors.clone().requires_grad_(True)

    @property
    def n_gaussians(self) -> int:
        return self.positions.shape[0]

    def to_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "means": self.positions.detach(),
            "scales_log": self.scales.detach(),
            "rotations": self.rotations.detach(),
            "opacities": self.opacities.detach(),
            "colors": self.colors.detach(),
        }

    @classmethod
    def from_tensors(cls, tensors: dict[str, torch.Tensor]) -> GaussianScene:
        return cls(
            positions=tensors["means"],
            scales=tensors["scales_log"],
            rotations=tensors["rotations"],
            opacities=tensors["opacities"],
            colors=tensors["colors"],
        )

    def clone_gaussians(self, mask: torch.Tensor, config: DensifyConfig):
        """Clone (duplicate) Gaussians at indices where mask is True."""
        if not mask.any():
            return

        new_positions = self.positions[mask].detach().clone()
        # Add small random offset to avoid exact overlap
        new_positions += torch.randn_like(new_positions) * 0.01

        new_scales = self.scales[mask].detach().clone()
        new_scales += torch.log(torch.tensor(config.clone_scale_factor))

        new_rotations = self.rotations[mask].clone()
        new_opacities = self.opacities[mask].detach().clone()
        new_colors = self.colors[mask].detach().clone()

        self.positions = torch.cat([
            self.positions.detach(), new_positions
        ]).requires_grad_(True)
        self.scales = torch.cat([
            self.scales.detach(), new_scales
        ]).requires_grad_(True)
        self.rotations = torch.cat([self.rotations.detach(), new_rotations])
        self.opacities = torch.cat([
            self.opacities.detach(), new_opacities
        ]).requires_grad_(True)
        self.colors = torch.cat([
            self.colors.detach(), new_colors
        ]).requires_grad_(True)

    def split_gaussians(self, mask: torch.Tensor, config: DensifyConfig):
        """Split large Gaussians into two smaller ones."""
        if not mask.any():
            return

        # Create two children from each parent
        parent_pos = self.positions[mask].detach()
        parent_scales = self.scales[mask].detach()
        parent_rot = self.rotations[mask]
        parent_opa = self.opacities[mask].detach()
        parent_col = self.colors[mask].detach()

        # Offset children along the longest axis
        scale_exp = torch.exp(parent_scales)
        longest_axis = scale_exp.argmax(dim=1)
        offset = torch.zeros_like(parent_pos)
        for i in range(len(offset)):
            axis = longest_axis[i].item()
            offset[i, axis] = scale_exp[i, axis] * 0.5

        child1_pos = parent_pos + offset
        child2_pos = parent_pos - offset
        child_scales = parent_scales + torch.log(torch.tensor(config.split_scale_factor))

        # Remove parents, add children
        keep_mask = ~mask
        self.positions = torch.cat([
            self.positions[keep_mask].detach(), child1_pos, child2_pos
        ]).requires_grad_(True)
        self.scales = torch.cat([
            self.scales[keep_mask].detach(), child_scales, child_scales
        ]).requires_grad_(True)
        self.rotations = torch.cat([
            self.rotations[keep_mask], parent_rot, parent_rot
        ])
        self.opacities = torch.cat([
            self.opacities[keep_mask].detach(), parent_opa, parent_opa
        ]).requires_grad_(True)
        self.colors = torch.cat([
            self.colors[keep_mask].detach(), parent_col, parent_col
        ]).requires_grad_(True)

    def prune(self, config: DensifyConfig):
        """Remove Gaussians with very low opacity."""
        opacity_sigmoid = torch.sigmoid(self.opacities.detach())
        keep = opacity_sigmoid > config.opacity_prune_threshold

        if keep.all():
            return 0

        n_pruned = (~keep).sum().item()
        self.positions = self.positions[keep].detach().requires_grad_(True)
        self.scales = self.scales[keep].detach().requires_grad_(True)
        self.rotations = self.rotations[keep]
        self.opacities = self.opacities[keep].detach().requires_grad_(True)
        self.colors = self.colors[keep].detach().requires_grad_(True)
        return n_pruned


def densify_step(
    scene: GaussianScene,
    config: DensifyConfig,
) -> dict[str, int]:
    """
    Run one densification step based on accumulated position gradients.

    Requires that scene.positions.grad has been populated by a backward pass.

    Returns stats: n_cloned, n_split, n_pruned
    """
    stats = {"n_cloned": 0, "n_split": 0, "n_pruned": 0}

    if scene.positions.grad is None:
        return stats

    if scene.n_gaussians >= config.max_gaussians:
        stats["n_pruned"] = scene.prune(config)
        return stats

    grad_norm = scene.positions.grad.norm(dim=1)
    scale_max = torch.exp(scene.scales.detach()).max(dim=1).values

    high_grad = grad_norm > config.grad_threshold
    small_scale = scale_max < config.scale_threshold
    large_scale = ~small_scale

    clone_mask = high_grad & small_scale
    split_mask = high_grad & large_scale

    stats["n_cloned"] = clone_mask.sum().item()
    stats["n_split"] = split_mask.sum().item()

    # Split first (removes parents), then clone
    scene.split_gaussians(split_mask, config)

    # Recompute clone mask after split changed indices
    # For simplicity, skip clone if split happened (they can interleave next iteration)
    if stats["n_split"] == 0:
        scene.clone_gaussians(clone_mask, config)

    stats["n_pruned"] = scene.prune(config)

    return stats


def densify_loop(
    tensors: dict[str, torch.Tensor],
    n_iterations: int = 200,
    config: DensifyConfig | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run the full densification loop on a set of Gaussians.

    Uses a simple self-supervised loss: penalizes Gaussians that are isolated
    (far from neighbors) and rewards uniform coverage of the bounding volume.

    Args:
        tensors: initial Gaussian parameters
        n_iterations: number of densification steps
        config: densification configuration

    Returns:
        densified tensors dict
    """
    if config is None:
        config = DensifyConfig()

    scene = GaussianScene.from_tensors(tensors)
    initial_n = scene.n_gaussians

    for i in range(n_iterations):
        # Compute a coverage-encouraging loss:
        # For each Gaussian, penalize distance to nearest neighbor
        # This encourages cloning in sparse areas
        if scene.n_gaussians < 2:
            break

        pos = scene.positions
        # Pairwise distances (compute on small subset if too many)
        n = min(scene.n_gaussians, 2000)
        subset = pos[:n]
        dists = torch.cdist(subset, subset)
        # Mask diagonal without in-place op
        diag_mask = torch.eye(n, dtype=torch.bool, device=dists.device)
        dists = dists + diag_mask.float() * 1e10
        nn_dist = dists.min(dim=1).values
        loss = nn_dist.mean()

        loss.backward()

        densify_step(scene, config)

        # Zero gradients for next iteration
        if scene.positions.grad is not None:
            scene.positions.grad = None
        if scene.scales.grad is not None:
            scene.scales.grad = None
        if scene.opacities.grad is not None:
            scene.opacities.grad = None
        if scene.colors.grad is not None:
            scene.colors.grad = None

        if scene.n_gaussians >= config.max_gaussians:
            break

    return scene.to_tensors()
