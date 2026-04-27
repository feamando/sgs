"""
Raum 1.0: template-routing bridge.

Maps a token sequence to a 3D scene by routing each object-role token
to an object template from `templates.build_template_library(...)` and
stamping it at a predicted position with a predicted colour and scale.

Architecture:

    tokens (mu_s [B,N,d_s] + features [B,N,d_f])
      + learned positional embedding
      → Linear fusion to d_model
      → TransformerEncoder (2 layers, 4 heads, d_model)
      → per-token heads:
          position_head  → xyz                (every token)
          template_head  → 6-way logits       (object tokens)
          color_head     → RGB                (object tokens, sigmoid)
          scale_head     → scalar size        (object tokens)
          role_head      → N_ROLES logits     (every token)

Assembly is a separate step (`assemble_scene`) so training can use
just the head outputs (fast, analytic supervision) while inference
and demo rendering stamp templates into a flat Gaussian cloud.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import N_OBJECTS, N_ROLES


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class RaumBridge(nn.Module):
    """Context-aware text → object-list bridge."""

    def __init__(
        self,
        d_s: int = 64,
        d_f: int = 300,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 32,
        n_templates: int = N_OBJECTS,
        n_roles: int = N_ROLES,
        # Kept for CLI compatibility with the old bridge; unused at this
        # layer since we no longer upsample per-word Gaussians. The demo
        # stamps full templates instead.
        K: int = 32,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_f = d_f
        self.d_model = d_model
        self.max_len = max_len
        self.n_templates = n_templates
        self.K = K  # retained for checkpoint metadata only

        # Fuse semantic position + feature into d_model. Position
        # channel carries SGS "where", feature channel carries "what".
        self.input_proj = nn.Linear(d_s + d_f, d_model)

        # Learned absolute positional embedding. Token order encodes role
        # binding ("first sphere" vs "second sphere") in our templates.
        self.pos_emb = nn.Parameter(torch.zeros(max_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Per-token heads ────────────────────────────────────────
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.template_head = nn.Linear(d_model, n_templates)
        self.role_head = nn.Linear(d_model, n_roles)
        self.color_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_emb, std=0.02)
        # Keep initial positions near the origin so the network does
        # not start in a wildly off-scale regime.
        nn.init.normal_(self.position_head[-1].weight, std=0.02)

    def forward(
        self,
        mu_s: torch.Tensor,      # [B, N, d_s]
        features: torch.Tensor,  # [B, N, d_f]
        mask: torch.Tensor,      # [B, N] 1=real, 0=pad
    ) -> dict:
        B, N, _ = mu_s.shape
        if N > self.max_len:
            raise ValueError(
                f"Input length {N} exceeds model max_len {self.max_len}"
            )

        x = torch.cat([mu_s, features], dim=-1)          # [B, N, d_s+d_f]
        x = self.input_proj(x)                            # [B, N, d_model]
        x = x + self.pos_emb[:N].unsqueeze(0)             # broadcast

        # TransformerEncoder treats src_key_padding_mask as True=pad.
        pad_mask = mask < 0.5
        h = self.encoder(x, src_key_padding_mask=pad_mask)

        positions = self.position_head(h)                 # [B, N, 3]
        template_logits = self.template_head(h)           # [B, N, T]
        role_logits = self.role_head(h)                   # [B, N, R]
        colors = torch.sigmoid(self.color_head(h))        # [B, N, 3] in [0,1]
        # Scale is a multiplier on the template's intrinsic size.
        # Softplus keeps it strictly positive; bias toward ~1.0 at init.
        scales = F.softplus(self.scale_head(h)).squeeze(-1) + 0.1  # [B, N]

        return {
            "positions": positions,
            "template_logits": template_logits,
            "role_logits": role_logits,
            "colors": colors,
            "scales": scales,
            "coarse_means": positions,  # alias kept for analyzer compat
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

def compute_routing_loss(
    out: dict,
    batch: dict,
    *,
    lambda_pos: float = 1.0,
    lambda_dir: float = 0.5,
    lambda_tpl: float = 1.0,
    lambda_col: float = 1.0,
    lambda_scl: float = 0.5,
    lambda_rol: float = 0.5,
    pair_margin: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """
    Analytic multi-task loss. All terms are supervised by the fields
    already produced by `src.raum.data.tokenize_scene`:

      position_MSE  — every object-role token → its GT xyz
      direction     — pairwise margin in sign-of-difference for 2-object scenes
      template_CE   — every object-role token → its GT obj_type
      color_MSE     — every colour-role token → RGB (and object tokens inherit
                     the colour from the neighbouring/attached colour token)
      scale_MSE     — every size-role token → scale multiplier
      role_CE       — every token → role class

    All masks respect padding.
    """
    device = out["positions"].device

    positions = out["positions"]              # [B, N, 3]
    tpl_logits = out["template_logits"]       # [B, N, T]
    role_logits = out["role_logits"]          # [B, N, R]
    pred_colors = out["colors"]               # [B, N, 3]
    pred_scales = out["scales"]               # [B, N]

    mask = batch["mask"].to(device)                          # [B, N]
    role_labels = batch["role_labels"].to(device)            # [B, N] (-1 pad)
    obj_labels = batch["obj_labels"].to(device)              # [B, N] (-1 non-obj)
    color_labels = batch["color_labels"].to(device)          # [B, N, 3] (-1 pad)
    size_labels = batch["size_labels"].to(device)            # [B, N]   (-1 pad)
    object_positions = batch["object_positions"].to(device)  # [B, max_obj, 3]
    object_types = batch["object_types"].to(device)          # [B, max_obj] (-1 pad)
    object_colors = batch["object_colors"].to(device)        # [B, max_obj, 3]
    object_scales = batch["object_scales"].to(device)        # [B, max_obj]

    B, N = mask.shape

    # ── Role classification: every real token ──
    m_tok = mask > 0.5
    if m_tok.any():
        role_loss = F.cross_entropy(
            role_logits[m_tok], role_labels[m_tok].clamp(min=0),
        )
    else:
        role_loss = positions.new_zeros(())

    # ── Per-object supervision: iterate over samples ──
    # (B is small, this is cheap and reads clearly.)
    pos_mse = positions.new_zeros(())
    tpl_ce = positions.new_zeros(())
    col_mse = positions.new_zeros(())
    scl_mse = positions.new_zeros(())
    pair_loss = positions.new_zeros(())
    n_pos = 0
    n_pair = 0

    tpl_correct = 0
    dir_correct = 0
    dir_total = 0

    for b in range(B):
        # Indices of object-bearing tokens, in sentence order.
        is_obj = (obj_labels[b] >= 0) & (mask[b] > 0.5)
        tok_idxs = torch.nonzero(is_obj, as_tuple=False).flatten().tolist()

        # GT slot order = sentence order of object tokens.
        for slot, tok_i in enumerate(tok_idxs[: object_positions.shape[1]]):
            if object_types[b, slot] < 0:
                continue

            gt_pos = object_positions[b, slot]              # [3]
            gt_tpl = object_types[b, slot]                  # []
            gt_col = object_colors[b, slot]                 # [3]
            gt_scl = object_scales[b, slot]                 # []

            pos_mse = pos_mse + F.mse_loss(positions[b, tok_i], gt_pos, reduction="sum")
            tpl_ce = tpl_ce + F.cross_entropy(
                tpl_logits[b, tok_i].unsqueeze(0), gt_tpl.unsqueeze(0),
            )
            col_mse = col_mse + F.mse_loss(pred_colors[b, tok_i], gt_col, reduction="sum")
            scl_mse = scl_mse + (pred_scales[b, tok_i] - gt_scl).pow(2)
            n_pos += 1

            if tpl_logits[b, tok_i].argmax().item() == int(gt_tpl.item()):
                tpl_correct += 1

        # Pairwise direction margin for 2-object scenes.
        if len(tok_idxs) >= 2:
            gt_a = object_positions[b, 0]
            gt_b = object_positions[b, 1]
            pr_a = positions[b, tok_idxs[0]]
            pr_b = positions[b, tok_idxs[1]]
            diff_gt = gt_b - gt_a
            diff_pr = pr_b - pr_a
            active = diff_gt.abs() > 1e-3
            if active.any():
                direction = torch.sign(diff_gt[active])
                signed = direction * diff_pr[active]
                pair_loss = pair_loss + F.relu(pair_margin - signed).sum()
                n_pair += int(active.sum().item())
                dir_correct += int(((direction * diff_pr[active]) > 0).sum().item())
                dir_total += int(active.sum().item())

    pos_mse = pos_mse / max(n_pos, 1)
    tpl_ce = tpl_ce / max(n_pos, 1)
    col_mse = col_mse / max(n_pos, 1)
    scl_mse = scl_mse / max(n_pos, 1)
    pair_loss = pair_loss / max(n_pair, 1)

    total = (
        lambda_pos * pos_mse
        + lambda_dir * pair_loss
        + lambda_tpl * tpl_ce
        + lambda_col * col_mse
        + lambda_scl * scl_mse
        + lambda_rol * role_loss
    )

    metrics = {
        "loss": float(total.item()),
        "pos_mse": float(pos_mse.item()),
        "pair_margin": float(pair_loss.item()),
        "tpl_ce": float(tpl_ce.item()),
        "col_mse": float(col_mse.item()),
        "scl_mse": float(scl_mse.item()),
        "role_ce": float(role_loss.item()),
        "tpl_acc": tpl_correct / max(n_pos, 1),
        "dir_acc": dir_correct / max(dir_total, 1),
    }
    return total, metrics


# ──────────────────────────────────────────────────────────────────────
# Assembly: head outputs → flat Gaussian cloud
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PredictedObject:
    """Materialised object for a single sample; used by the demo."""
    word_index: int
    template_name: str
    template_id: int
    template_confidence: float
    position: list[float]
    color: list[float]
    scale: float


def assemble_scene(
    out: dict,
    template_lib: dict,
    template_names: list[str],
    mask: torch.Tensor,
    *,
    sample_index: int = 0,
    role_threshold: float = 0.0,
    object_role_id: int | None = None,
    soft_mix: bool = False,
    top_k_templates: int = 2,
) -> tuple[dict, list[PredictedObject]]:
    """
    Materialise the predicted scene for one sample into a flat Gaussian
    cloud that the viewer can render.

    If `soft_mix` is True the returned splats are a weighted blend of
    the top-k template shapes (anchored to a common Fibonacci count);
    otherwise we pick the argmax template per object token.
    """
    b = sample_index
    positions = out["positions"][b].detach().cpu()       # [N, 3]
    tpl_logits = out["template_logits"][b].detach().cpu()
    role_logits = out["role_logits"][b].detach().cpu()
    colors = out["colors"][b].detach().cpu()             # [N, 3]
    scales = out["scales"][b].detach().cpu()             # [N]
    m = mask[b].detach().cpu()                           # [N]

    role_pred = role_logits.argmax(dim=-1).tolist()

    # Which tokens are objects? Prefer the model's role prediction when
    # a specific role id is given, else fall back to argmax template
    # confidence over a threshold. Both still require mask==1.
    object_tokens = []
    for i in range(positions.shape[0]):
        if m[i] < 0.5:
            continue
        if object_role_id is not None:
            if role_pred[i] == object_role_id:
                object_tokens.append(i)
        else:
            # Fallback: any token with high template-head confidence.
            probs = torch.softmax(tpl_logits[i], dim=-1)
            if probs.max().item() > 0.5:
                object_tokens.append(i)

    # Build flat Gaussian cloud.
    all_means = []
    all_scales_log = []
    all_opacities = []
    all_colors = []
    objects: list[PredictedObject] = []

    for i in object_tokens:
        probs = torch.softmax(tpl_logits[i], dim=-1)
        top = probs.topk(min(top_k_templates, probs.shape[-1]))
        chosen_idx = int(top.indices[0].item())
        chosen_name = template_names[chosen_idx]
        tpl = template_lib[chosen_name]

        pos = positions[i]
        col = colors[i].clamp(0.0, 1.0)
        scl = float(scales[i].item())

        # Stamp: transform template means by scale + translation.
        means = tpl.means * scl + pos.unsqueeze(0)
        sc_log = tpl.scales.clone()
        # Shift log-scale by log(scl) so the individual splats grow/shrink.
        sc_log = sc_log + torch.log(torch.tensor(max(scl, 1e-3)))
        opac = tpl.opacities.clone()

        all_means.append(means)
        all_scales_log.append(sc_log)
        all_opacities.append(opac)
        all_colors.append(col.unsqueeze(0).expand(means.shape[0], 3).clone())

        objects.append(PredictedObject(
            word_index=i,
            template_name=chosen_name,
            template_id=chosen_idx,
            template_confidence=float(top.values[0].item()),
            position=pos.tolist(),
            color=col.tolist(),
            scale=scl,
        ))

    if not all_means:
        empty = torch.zeros(0, 3)
        splats = {
            "means": empty,
            "scales_log": empty,
            "opacities": torch.zeros(0),
            "colors": empty,
        }
        return splats, objects

    splats = {
        "means": torch.cat(all_means, dim=0),
        "scales_log": torch.cat(all_scales_log, dim=0),
        "opacities": torch.cat(all_opacities, dim=0),
        "colors": torch.cat(all_colors, dim=0),
    }
    return splats, objects
