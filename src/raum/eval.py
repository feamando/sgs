"""
Evaluation metrics for Raum PoC.
"""

import torch
import numpy as np
from .vocab import ROLE_OBJECT, ROLE_COLOR, ROLE_SIZE, ROLE_RELATION


@torch.no_grad()
def compute_metrics(preds: dict, batch: dict) -> dict:
    """
    Compute all evaluation metrics for a batch.

    Returns dict of metric_name → value.
    """
    mask = batch["mask"]
    metrics = {}

    # ── Role accuracy ──
    role_pred = preds["role_logits"].argmax(dim=-1)  # [B, L]
    role_correct = (role_pred == batch["role_labels"]) & (mask > 0)
    metrics["role_acc"] = role_correct.float().sum().item() / mask.sum().item()

    # ── Object accuracy (only object words) ──
    obj_mask = (batch["obj_labels"] != -1) & (mask > 0)
    if obj_mask.any():
        obj_pred = preds["obj_logits"][obj_mask].argmax(dim=-1)
        obj_correct = (obj_pred == batch["obj_labels"][obj_mask])
        metrics["obj_acc"] = obj_correct.float().mean().item()
    else:
        metrics["obj_acc"] = -1.0

    # ── Color L2 (only color words) ──
    color_mask = (batch["color_labels"][:, :, 0] != -1) & (mask > 0)
    if color_mask.any():
        diff = preds["color_pred"][color_mask] - batch["color_labels"][color_mask]
        metrics["color_l2"] = diff.norm(dim=-1).mean().item()
    else:
        metrics["color_l2"] = -1.0

    # ── Size MAE (only size words) ──
    size_mask = (batch["size_labels"] != -1) & (mask > 0)
    if size_mask.any():
        diff = (preds["size_pred"][size_mask] - batch["size_labels"][size_mask]).abs()
        metrics["size_mae"] = diff.mean().item()
    else:
        metrics["size_mae"] = -1.0

    # ── Relation position MAE (only relation words) ──
    B, L = batch["role_labels"].shape
    rel_mask = (batch["role_labels"] == ROLE_RELATION) & (mask > 0)
    if rel_mask.any():
        target = batch["relation_label"].unsqueeze(1).expand(B, L, 3)
        diff = (preds["relation_pred"][rel_mask] - target[rel_mask]).abs()
        metrics["rel_mae"] = diff.mean().item()
    else:
        metrics["rel_mae"] = -1.0

    return metrics


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Average metrics across batches (skip -1 entries)."""
    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if m[key] != -1.0]
        agg[key] = np.mean(vals) if vals else -1.0
    return agg


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print metrics."""
    parts = []
    for k, v in metrics.items():
        if v == -1.0:
            continue
        if "acc" in k:
            parts.append(f"{k}={v:.1%}")
        elif "l2" in k or "mae" in k:
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v:.4f}")
    print(f"  {prefix}{' | '.join(parts)}")
