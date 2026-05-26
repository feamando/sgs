"""
Template-based subdivision for Raum 1.4.

Each leaf Gaussian in a composition tree gets replaced by a deformed template
(a learned set of Gaussians representing the object's shape). The subdivision
MLP selects a template and outputs deformation parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Template:
    """A GS shape template: a set of Gaussians representing one object class."""
    category: str
    template_id: int
    positions: torch.Tensor   # [M, 3]
    scales: torch.Tensor      # [M, 3]
    rotations: torch.Tensor   # [M, 4]
    opacities: torch.Tensor   # [M]
    colors: torch.Tensor      # [M, 3]

    @property
    def n_gaussians(self) -> int:
        return self.positions.shape[0]

    def to(self, device: torch.device) -> Template:
        return Template(
            category=self.category,
            template_id=self.template_id,
            positions=self.positions.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            opacities=self.opacities.to(device),
            colors=self.colors.to(device),
        )


class TemplateLibrary:
    """Collection of templates indexed by category."""

    def __init__(self, templates_dir: str | Path):
        self.templates_dir = Path(templates_dir)
        self.templates: dict[str, list[Template]] = {}
        self.all_categories: list[str] = []
        self._load()

    def _load(self):
        if not self.templates_dir.exists():
            return
        for path in sorted(self.templates_dir.glob("*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=True)
            cat = data["category"]
            tpl = Template(
                category=cat,
                template_id=data["template_id"],
                positions=data["positions"],
                scales=data["scales"],
                rotations=data["rotations"],
                opacities=data["opacities"],
                colors=data["colors"],
            )
            if cat not in self.templates:
                self.templates[cat] = []
                self.all_categories.append(cat)
            self.templates[cat].append(tpl)

    def get(self, category: str, template_id: int = 0) -> Template | None:
        templates = self.templates.get(category, [])
        for t in templates:
            if t.template_id == template_id:
                return t
        return templates[0] if templates else None

    def n_templates(self, category: str) -> int:
        return len(self.templates.get(category, []))

    @property
    def categories(self) -> list[str]:
        return self.all_categories


class SubdivisionMLP(nn.Module):
    """
    Predicts template selection and deformation parameters for a leaf Gaussian.

    Input: parent Gaussian features + semantic embedding
    Output: template logits + deformation (position offset, scale factor, color shift, rotation delta)
    """

    def __init__(
        self,
        embed_dim: int = 300,
        n_categories: int = 50,
        max_templates_per_category: int = 5,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.n_categories = n_categories
        self.max_templates = max_templates_per_category

        # Input: position(3) + scale(1) + color(3) + embedding(300) + context(300) = 607
        input_dim = 3 + 1 + 3 + embed_dim + embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Template selection head
        self.template_head = nn.Linear(hidden_dim // 2, max_templates_per_category)

        # Deformation head: position_offset(3) + scale_factor(3) + color_shift(3) + rotation_delta(4) = 13
        self.deform_head = nn.Linear(hidden_dim // 2, 13)

    def forward(
        self,
        position: torch.Tensor,      # [B, 3]
        scale: torch.Tensor,          # [B, 1]
        color: torch.Tensor,          # [B, 3]
        embedding: torch.Tensor,      # [B, 300]
        context: torch.Tensor,        # [B, 300]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            template_logits: [B, max_templates]
            deformation: [B, 13] (pos_offset[3], scale_factor[3], color_shift[3], rot_delta[4])
        """
        x = torch.cat([position, scale, color, embedding, context], dim=-1)
        features = self.net(x)
        template_logits = self.template_head(features)
        deformation = self.deform_head(features)
        return template_logits, deformation


def apply_deformation(
    template: Template,
    deformation: torch.Tensor,
    parent_position: torch.Tensor,
    parent_scale: float,
) -> dict[str, torch.Tensor]:
    """
    Apply deformation parameters to a template, positioning it relative to the parent.

    Args:
        template: the GS template to deform
        deformation: [13] tensor (pos_offset[3], scale_factor[3], color_shift[3], rot_delta[4])
        parent_position: [3] world position of the parent Gaussian
        parent_scale: scalar scale of the parent

    Returns:
        dict with positions [M,3], scales [M,3], rotations [M,4], opacities [M], colors [M,3]
    """
    pos_offset = deformation[:3]
    scale_factor = deformation[3:6]
    color_shift = deformation[6:9]
    rot_delta = deformation[9:13]

    # Position: template positions scaled and offset
    positions = template.positions * parent_scale + parent_position + pos_offset * parent_scale

    # Scale: template scales + learned factor
    scales = template.scales + scale_factor.unsqueeze(0)

    # Color: template colors + shift, clamped to [0, 1]
    colors = (template.colors + color_shift.unsqueeze(0)).clamp(0, 1)

    # Rotation: template rotations (no learned rotation composition for now)
    rotations = template.rotations.clone()

    # Opacity: unchanged
    opacities = template.opacities.clone()

    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "colors": colors,
    }
