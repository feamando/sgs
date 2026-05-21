"""
Recursive semantic-to-geometric decomposition (Raum 1.3).

A composition tree maps a text prompt to a hierarchy of sub-concepts,
each with spatial relations, terminating at individual Gaussian splat
parameters. The tree is the intermediate representation between
language (Planck decomposer) and geometry (renderer).

Example:
    "castle on a hill"
    ├── castle (position=[0, 1, 0], scale=2.0)
    │   ├── tower_NW (rel_pos=[-1, 0, 1], terminal Gaussians)
    │   ├── tower_NE (rel_pos=[1, 0, 1], terminal Gaussians)
    │   ├── gate (rel_pos=[0, 0, 1.5], terminal Gaussians)
    │   └── keep (rel_pos=[0, 0, 0], terminal Gaussians)
    └── hill (position=[0, -0.5, 0], scale=3.0, terminal Gaussians)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class GaussianParams:
    """Terminal Gaussian splat parameters."""
    position: list[float]       # [x, y, z]
    scale: list[float]          # [sx, sy, sz] log-scale
    opacity: float              # logit (pre-sigmoid)
    color: list[float]          # [r, g, b] in [0, 1]


@dataclass
class CompositionNode:
    """
    One node in the composition tree.

    Internal nodes have children (sub-concepts).
    Leaf nodes have gaussians (terminal primitives).
    """
    name: str
    position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    color: list[float] | None = None
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Internal node: has children
    children: list[CompositionNode] = field(default_factory=list)

    # Leaf node: has terminal Gaussians
    gaussians: list[GaussianParams] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def n_gaussians_recursive(self) -> int:
        if self.is_leaf:
            return len(self.gaussians)
        return sum(c.n_gaussians_recursive for c in self.children)

    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.depth for c in self.children)

    def flatten_gaussians(self, parent_pos: list[float] | None = None,
                          parent_scale: float = 1.0) -> list[GaussianParams]:
        """
        Recursively flatten the tree into a list of world-space Gaussians.

        Each node's position is relative to its parent. Scale compounds
        multiplicatively down the tree.
        """
        # Compute world position and scale for this node
        world_pos = [0.0, 0.0, 0.0]
        if parent_pos:
            for i in range(3):
                world_pos[i] = parent_pos[i] + self.position[i] * parent_scale
        else:
            world_pos = list(self.position)

        world_scale = parent_scale * self.scale

        if self.is_leaf:
            # Transform leaf Gaussians to world space
            result = []
            for g in self.gaussians:
                world_g = GaussianParams(
                    position=[world_pos[i] + g.position[i] * world_scale for i in range(3)],
                    scale=[g.scale[i] + math.log(max(world_scale, 1e-6)) for i in range(3)],
                    opacity=g.opacity,
                    color=g.color if self.color is None else self.color,
                )
                result.append(world_g)
            return result
        else:
            # Recurse into children
            result = []
            for child in self.children:
                result.extend(child.flatten_gaussians(world_pos, world_scale))
            return result

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d = {
            "name": self.name,
            "position": self.position,
            "scale": self.scale,
        }
        if self.color:
            d["color"] = self.color
        if self.rotation != [0.0, 0.0, 0.0]:
            d["rotation"] = self.rotation
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.gaussians:
            d["gaussians"] = [
                {"position": g.position, "scale": g.scale,
                 "opacity": g.opacity, "color": g.color}
                for g in self.gaussians
            ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CompositionNode:
        """Deserialize from JSON-compatible dict."""
        node = cls(
            name=d["name"],
            position=d.get("position", [0, 0, 0]),
            scale=d.get("scale", 1.0),
            color=d.get("color"),
            rotation=d.get("rotation", [0, 0, 0]),
        )
        for child_d in d.get("children", []):
            node.children.append(cls.from_dict(child_d))
        for g_d in d.get("gaussians", []):
            node.gaussians.append(GaussianParams(
                position=g_d["position"],
                scale=g_d["scale"],
                opacity=g_d["opacity"],
                color=g_d["color"],
            ))
        return node


def save_tree(tree: CompositionNode, path: str | Path):
    """Save a composition tree to JSON."""
    with open(path, "w") as f:
        json.dump(tree.to_dict(), f, indent=2)


def load_tree(path: str | Path) -> CompositionNode:
    """Load a composition tree from JSON."""
    with open(path) as f:
        return CompositionNode.from_dict(json.load(f))


def tree_to_tensors(tree: CompositionNode) -> dict[str, torch.Tensor]:
    """
    Flatten a composition tree into renderer-ready tensors.

    Returns:
        means: [N, 3] world-space positions
        scales_log: [N, 3] log-scales
        opacities: [N] logit opacities
        colors: [N, 3] RGB colors
    """
    gaussians = tree.flatten_gaussians()
    if not gaussians:
        return {
            "means": torch.zeros(0, 3),
            "scales_log": torch.zeros(0, 3),
            "opacities": torch.zeros(0),
            "colors": torch.zeros(0, 3),
        }

    means = torch.tensor([g.position for g in gaussians], dtype=torch.float32)
    scales_log = torch.tensor([g.scale for g in gaussians], dtype=torch.float32)
    opacities = torch.tensor([g.opacity for g in gaussians], dtype=torch.float32)
    colors = torch.tensor([g.color for g in gaussians], dtype=torch.float32)

    return {
        "means": means,
        "scales_log": scales_log,
        "opacities": opacities,
        "colors": colors,
    }


def print_tree(node: CompositionNode, indent: int = 0):
    """Pretty-print a composition tree."""
    prefix = "  " * indent
    leaf_info = f" [{len(node.gaussians)} gaussians]" if node.is_leaf else ""
    pos = f"pos=({node.position[0]:.1f}, {node.position[1]:.1f}, {node.position[2]:.1f})"
    print(f"{prefix}{node.name} ({pos}, scale={node.scale:.1f}){leaf_info}")
    for child in node.children:
        print_tree(child, indent + 1)
