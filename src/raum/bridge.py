"""
PoC-C: Shared-Equation Bridge model.

Learns a space transform from SGS semantic Gaussians to 3DGS spatial Gaussians.
~335K trainable parameters (SGS encoder frozen).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaumBridge(nn.Module):
    """
    Text → 3D Gaussian scene via learned space transform + upsampler.

    Flow:
      SGS vocab → per-word semantic Gaussians (μ_s, Σ_s, α_s, f_s)
          ↓ mu_proj
      coarse xyz positions (one per word)
          ↓ color_head, scale_head, opacity_head
      per-word 3D attributes
          ↓ upsampler
      dense Gaussian cloud (K per word)
    """

    def __init__(
        self,
        d_s: int = 64,
        d_f: int = 300,
        K: int = 64,
    ):
        super().__init__()
        self.K = K

        # ── Space transform: semantic position → xyz ──
        self.mu_proj = nn.Sequential(
            nn.Linear(d_s, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        # ── Attribute heads ──
        self.color_head = nn.Sequential(
            nn.Linear(d_f, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        self.scale_head = nn.Sequential(
            nn.Linear(d_f, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        self.opacity_head = nn.Sequential(
            nn.Linear(d_f, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # ── Gaussian upsampler ──
        # Input: per-word features + coarse xyz
        # Output: K × (offset[3] + scale[3] + opacity[1] + color[3]) = K × 10
        self.upsampler = nn.Sequential(
            nn.Linear(d_f + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, K * 10),
        )

        self._init_weights()

    def _init_weights(self):
        """Small init for smooth training start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # mu_proj output layer: very small init so positions start near origin
        nn.init.normal_(self.mu_proj[-1].weight, std=0.01)

        # Upsampler final layer: small so offsets start near zero
        nn.init.normal_(self.upsampler[-1].weight, std=0.005)

    def forward(
        self,
        mu_s: torch.Tensor,         # [B, N, d_s] semantic positions
        features: torch.Tensor,      # [B, N, d_f] features
        mask: torch.Tensor,          # [B, N] 1=real, 0=pad
    ) -> dict:
        """
        Map semantic Gaussians to spatial Gaussians.

        Returns dict with all Gaussian params for rendering:
            means:     [B, N*K, 3]
            scales:    [B, N*K, 3]  (log-scale)
            opacities: [B, N*K]     (logit, pre-sigmoid)
            colors:    [B, N*K, 3]  (RGB, 0-1)
            coarse_means: [B, N, 3] (for analysis — word-level positions)
        """
        B, N, _ = mu_s.shape
        K = self.K

        # 1. Space transform → coarse positions
        mu_xyz = self.mu_proj(mu_s)                        # [B, N, 3]

        # 2. Per-word attributes
        color_coarse = self.color_head(features)           # [B, N, 3]
        scale_coarse = self.scale_head(features)           # [B, N, 3]
        opacity_coarse = self.opacity_head(features)       # [B, N, 1]

        # 3. Upsample
        up_input = torch.cat([features, mu_xyz], dim=-1)   # [B, N, d_f+3]
        up_raw = self.upsampler(up_input)                  # [B, N, K*10]
        up = up_raw.view(B, N, K, 10)

        offsets  = up[..., 0:3] * 0.3                      # local offsets, bounded
        d_scale  = up[..., 3:6]                             # delta log-scale
        d_opac   = up[..., 6:7]                             # delta opacity logit
        d_color  = torch.sigmoid(up[..., 7:10])            # local color modulation

        # 4. Assemble dense Gaussians
        # Position: coarse center + local offset
        means = mu_xyz.unsqueeze(2) + offsets               # [B, N, K, 3]
        means = means.view(B, N * K, 3)

        # Scale: base + per-splat delta
        scales = scale_coarse.unsqueeze(2) + d_scale        # [B, N, K, 3]
        scales = scales.view(B, N * K, 3)

        # Opacity: base + delta
        opacities = opacity_coarse.unsqueeze(2) + d_opac    # [B, N, K, 1]
        opacities = opacities.view(B, N * K)

        # Color: blend coarse color with per-splat modulation
        colors = color_coarse.unsqueeze(2) * d_color         # [B, N, K, 3]
        colors = colors.view(B, N * K, 3)

        # 5. Mask out Gaussians from padding words
        if mask is not None:
            word_mask = mask.unsqueeze(2).expand(B, N, K).reshape(B, N * K)
            opacities = opacities - 100.0 * (1.0 - word_mask)  # sigmoid(-100) ≈ 0

        return {
            "means": means,
            "scales": scales,
            "opacities": opacities,
            "colors": colors,
            "coarse_means": mu_xyz,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def compute_bridge_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    scene: dict,
    lambda_scale: float = 0.01,
    lambda_center: float = 0.001,
    lambda_spread: float = 0.0,
    target_spread: float = 1.0,
    coarse_means_batch: torch.Tensor | None = None,
    mask_batch: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Rendering loss + regularization.

    Args:
        rendered: [3, H, W] predicted image
        target: [3, H, W] ground truth image
        scene: output dict from RaumBridge (this sample only)
        lambda_scale: weight for scale regularization
        lambda_center: weight for centering regularization
        lambda_spread: weight for spread regulariser (Fix 1). 0 disables it.
        target_spread: desired per-axis std of coarse means across the batch.
        coarse_means_batch: [B, N, 3] coarse means for the full batch (only
            needed when lambda_spread > 0).
        mask_batch: [B, N] mask for the full batch.
    """
    # MSE on pixels
    mse = F.mse_loss(rendered, target)

    # PSNR (for logging)
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))

    # Regularization: prevent Gaussians from exploding
    scale_reg = scene["scales"].exp().mean()
    center_reg = scene["means"].norm(dim=-1).mean()

    loss = mse + lambda_scale * scale_reg + lambda_center * center_reg

    # ── Spread regulariser (Fix 1) ──
    # Pushes the coarse means' per-axis std toward `target_spread`. Computed
    # on the full batch, not just the current sample, so the gradient is
    # independent of per-sample rendering. Caller should only pass this on
    # one render pass per batch to avoid double-counting.
    spread_val = 0.0
    if lambda_spread > 0.0 and coarse_means_batch is not None:
        flat = coarse_means_batch.reshape(-1, 3)
        if mask_batch is not None:
            m = mask_batch.reshape(-1).bool()
            flat = flat[m]
        # Per-axis std, then mean over axes. log-barrier style so a
        # collapsed bridge (std → 0) pays a large price.
        std = flat.std(dim=0)                       # [3]
        spread_penalty = ((std - target_spread) ** 2).mean()
        loss = loss + lambda_spread * spread_penalty
        spread_val = std.mean().item()

    metrics = {
        "loss": loss.item(),
        "mse": mse.item(),
        "psnr": psnr.item(),
        "scale_mean": scale_reg.item(),
        "center_mean": center_reg.item(),
        "spread": spread_val,
    }
    return loss, metrics


# ── Fix 2: direct positional supervision ──

def compute_position_loss(
    coarse_means: torch.Tensor,     # [B, N, 3] bridge output
    obj_labels: torch.Tensor,        # [B, N] long, object index or -1
    object_positions: torch.Tensor,  # [B, max_obj, 3] GT positions
    mask: torch.Tensor,              # [B, N] 1=real, 0=pad
    margin: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Direct supervision that ties coarse means to the ground-truth object
    positions they should be rendering.

    For each object-bearing word in each sample, find its GT position (by
    insertion order within the sentence) and compute an MSE against the
    bridge's coarse mean at that token. Non-object tokens are ignored.

    Also adds a pairwise margin loss for 2-object scenes so that the axis
    with the relevant relation ("above" → y, "left" → x, ...) is pushed
    in the right *direction*, not just to the right magnitude. The sign
    comes from the GT positions themselves — no extra bookkeeping.

    Returns (loss, metrics).
    """
    B, N, _ = coarse_means.shape
    device = coarse_means.device

    # Regression term: for each token whose obj_labels[b, n] >= 0 and which
    # is the first (or second) object of that sample, snap to GT.
    reg_loss = torch.zeros((), device=device)
    pair_loss = torch.zeros((), device=device)
    n_reg = 0
    n_pairs = 0

    for b in range(B):
        # Find indices of object-bearing words in order
        is_obj = (obj_labels[b] >= 0) & (mask[b] > 0.5)
        idxs = torch.nonzero(is_obj, as_tuple=False).flatten().tolist()

        # Regression: token i in idxs → object i in object_positions
        gt_positions = []
        pred_positions = []
        for i, tok_i in enumerate(idxs[:object_positions.shape[1]]):
            gt_positions.append(object_positions[b, i])
            pred_positions.append(coarse_means[b, tok_i])
        if pred_positions:
            gt_stack = torch.stack(gt_positions)
            pred_stack = torch.stack(pred_positions)
            reg_loss = reg_loss + F.mse_loss(pred_stack, gt_stack, reduction="sum")
            n_reg += gt_stack.shape[0]

        # Pairwise margin: only for 2-object scenes
        if len(idxs) >= 2:
            gt_a = object_positions[b, 0]
            gt_b = object_positions[b, 1]
            pr_a = coarse_means[b, idxs[0]]
            pr_b = coarse_means[b, idxs[1]]
            # For each axis where the GT objects differ, enforce the sign
            # of the difference with a margin.
            diff_gt = gt_b - gt_a          # [3]
            diff_pr = pr_b - pr_a          # [3]
            # sign(gt) * diff_pr should be >= margin where |gt| is large.
            active = diff_gt.abs() > 1e-3
            if active.any():
                direction = torch.sign(diff_gt[active])
                signed = direction * diff_pr[active]
                pair_loss = pair_loss + F.relu(margin - signed).sum()
                n_pairs += int(active.sum().item())

    reg_loss = reg_loss / max(n_reg, 1)
    pair_loss = pair_loss / max(n_pairs, 1)
    total = reg_loss + pair_loss

    metrics = {
        "pos_mse": reg_loss.item(),
        "pos_pair_margin": pair_loss.item(),
        "n_pos_tokens": n_reg,
    }
    return total, metrics
