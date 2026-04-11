"""
PoC-D: Compositional Scene Graph model.

Frozen SGS/GloVe features → per-word role classifier + attribute heads.
~50K trainable parameters.
"""

import torch
import torch.nn as nn

from .vocab import N_OBJECTS, N_ROLES


class RaumCompositional(nn.Module):
    """
    Text → scene decomposition via per-word classification.

    Takes per-word features from SGS vocab or GloVe, predicts:
      - role (object/color/size/relation/other)
      - object type (for object-role words)
      - color RGB (for color-role words)
      - size scale (for size-role words)
      - spatial offset (for relation-role words, conditioned on context)
    """

    def __init__(self, d_f: int = 300):
        super().__init__()

        # Role classifier
        self.role_head = nn.Sequential(
            nn.Linear(d_f, 64),
            nn.ReLU(),
            nn.Linear(64, N_ROLES),
        )

        # Object type classifier
        self.object_head = nn.Sequential(
            nn.Linear(d_f, 64),
            nn.ReLU(),
            nn.Linear(64, N_OBJECTS),
        )

        # Color predictor (RGB)
        self.color_head = nn.Sequential(
            nn.Linear(d_f, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        # Size predictor (positive scalar)
        self.size_head = nn.Sequential(
            nn.Linear(d_f, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        # Relation head: concat(relation_features, mean_context) → xyz offset
        self.relation_head = nn.Sequential(
            nn.Linear(d_f * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        Args:
            features: [B, L, d_f] — per-word features (from SGS vocab or GloVe)
            mask: [B, L] — 1 for real tokens, 0 for pad

        Returns:
            dict with predictions for each head.
        """
        B, L, _ = features.shape

        role_logits = self.role_head(features)         # [B, L, N_ROLES]
        obj_logits = self.object_head(features)        # [B, L, N_OBJECTS]
        color_pred = self.color_head(features)         # [B, L, 3]
        size_pred = self.size_head(features).squeeze(-1)  # [B, L]

        # Context for relation: mean-pool all features in the sentence
        masked_feat = features * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        context = masked_feat.sum(dim=1) / lengths     # [B, d_f]
        context_broadcast = context.unsqueeze(1).expand_as(features)
        rel_input = torch.cat([features, context_broadcast], dim=-1)
        relation_pred = self.relation_head(rel_input)  # [B, L, 3]

        return {
            "role_logits": role_logits,
            "obj_logits": obj_logits,
            "color_pred": color_pred,
            "size_pred": size_pred,
            "relation_pred": relation_pred,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_loss(preds: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    """
    Compute multi-task loss.

    Each attribute loss is masked to only apply to words with that role.
    """
    B, L = batch["role_labels"].shape
    mask = batch["mask"]  # [B, L]

    # Role classification (all words)
    role_loss = nn.functional.cross_entropy(
        preds["role_logits"].view(-1, N_ROLES),
        batch["role_labels"].view(-1),
        ignore_index=-1,
        reduction="mean",
    )

    # Object classification (only object words: obj_labels != -1)
    obj_mask = (batch["obj_labels"] != -1) & (mask > 0)
    if obj_mask.any():
        obj_loss = nn.functional.cross_entropy(
            preds["obj_logits"][obj_mask],
            batch["obj_labels"][obj_mask],
            reduction="mean",
        )
    else:
        obj_loss = torch.tensor(0.0, device=mask.device)

    # Color (only color words: color_labels[:,:,0] != -1)
    color_mask = (batch["color_labels"][:, :, 0] != -1) & (mask > 0)
    if color_mask.any():
        color_loss = nn.functional.mse_loss(
            preds["color_pred"][color_mask],
            batch["color_labels"][color_mask],
            reduction="mean",
        )
    else:
        color_loss = torch.tensor(0.0, device=mask.device)

    # Size (only size words: size_labels != -1)
    size_mask = (batch["size_labels"] != -1) & (mask > 0)
    if size_mask.any():
        size_loss = nn.functional.mse_loss(
            preds["size_pred"][size_mask],
            batch["size_labels"][size_mask],
            reduction="mean",
        )
    else:
        size_loss = torch.tensor(0.0, device=mask.device)

    # Relation offset (only relation words)
    from .vocab import ROLE_RELATION
    rel_mask = (batch["role_labels"] == ROLE_RELATION) & (mask > 0)
    if rel_mask.any():
        # Target: the scene's relation_label, broadcast to all relation words
        target = batch["relation_label"]  # [B, 3]
        target_expanded = target.unsqueeze(1).expand(B, L, 3)
        rel_loss = nn.functional.mse_loss(
            preds["relation_pred"][rel_mask],
            target_expanded[rel_mask],
            reduction="mean",
        )
    else:
        rel_loss = torch.tensor(0.0, device=mask.device)

    total = role_loss + obj_loss + color_loss + size_loss + rel_loss

    metrics = {
        "loss": total.item(),
        "role_loss": role_loss.item(),
        "obj_loss": obj_loss.item(),
        "color_loss": color_loss.item(),
        "size_loss": size_loss.item(),
        "rel_loss": rel_loss.item(),
    }
    return total, metrics
