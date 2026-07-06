"""
Raum Path A: train the learned FILL model (part-token + pose -> Gaussians).

Replaces the hand-built grammar fill (expand_part) with a learned conditional
SET generator. This is "decompose to particles, reconstruct in hundreds of
thousands of splats" -- the upgrade that lifts the grammar ceiling (the
decomposer was only ever as expressive as expand_part).

Architecture (DETR-style set prediction, mirrors RaumBridge's transformer):
  condition = part-token embedding + pose MLP(courses, color)
  N learnable query slots -> TransformerDecoder attends to the condition
  per slot: an "active" logit + a Gaussian (mu[3], log-scale[3], quat[4],
            opacity[1], color[3])
  N capped at --max-gaussians (default 512; the grammar's biggest part is the
  504-stone keep).

Two-stage supervision:
  Stage A (this file, default): SET-MATCHING reconstruction. Chamfer-style
    nearest-neighbour match between predicted and target Gaussians (order-free,
    since a part's stones have no canonical order) + BCE on the active mask.
    Pure geometry/appearance regression against the grammar's own clouds. CPU-
    trainable on the ~480-example dataset. Proves the model can REPRODUCE the
    grammar fill.
  Stage B (--render-supervision sds, 4090 / .venv-sds): add a render-score
    (SDS) term so parts LOOK right under a diffusion prior, not just match the
    template. This is where it can BEAT the grammar. NOT template Chamfer-to-
    scan (Raum 0.6 proved that distorts clean geometry).

Usage:
  python scripts/train_fill.py --data data/fill/path1_fill.json --selftest
  python scripts/train_fill.py --data data/fill/path1_fill.json --epochs 30 \
    --out checkpoints/fill_model
  python scripts/train_fill.py --data data/fill/path1_fill.json --epochs 30 \
    --render-supervision sds --out checkpoints/fill_model    # 4090 / .venv-sds
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# canonical part-token vocabulary (must match build_fill_dataset PART_SPECS kinds)
PART_KINDS = ["tower", "wall", "keep", "gatehouse", "gate", "tree", "door",
              "window", "arrow_slit", "arch", "cliff", "rock"]
PART_TO_ID = {k: i for i, k in enumerate(PART_KINDS)}


# ── model ──────────────────────────────────────────────────────────────

class FillModel(nn.Module):
    """Conditional Gaussian-set generator. part-token + pose -> {Gaussian}."""

    def __init__(self, n_parts=len(PART_KINDS), d_model=128, n_layers=3,
                 n_heads=4, max_gaussians=512):
        super().__init__()
        self.max_g = max_gaussians
        self.d_model = d_model
        self.part_emb = nn.Embedding(n_parts, d_model)
        # pose: courses (scalar) + color (3) -> d_model
        self.pose_mlp = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.queries = nn.Parameter(torch.randn(max_gaussians, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        # per-slot heads
        self.head_active = nn.Linear(d_model, 1)
        self.head_mean = nn.Linear(d_model, 3)
        self.head_scale = nn.Linear(d_model, 3)   # log-scale
        self.head_quat = nn.Linear(d_model, 4)
        self.head_opacity = nn.Linear(d_model, 1)
        self.head_color = nn.Linear(d_model, 3)

    def forward(self, part_ids, pose):
        """part_ids [B], pose [B,4] -> dict of per-slot tensors [B, max_g, *]."""
        B = part_ids.shape[0]
        cond = self.part_emb(part_ids) + self.pose_mlp(pose)   # [B, d_model]
        memory = cond.unsqueeze(1)                              # [B, 1, d_model]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)        # [B, max_g, d_model]
        h = self.decoder(q, memory)                            # [B, max_g, d_model]
        quat = self.head_quat(h)
        quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-6)
        return {
            "active": self.head_active(h).squeeze(-1),         # [B, max_g] logit
            "means": self.head_mean(h),
            "scales_log": self.head_scale(h),
            "rotations": quat,
            "opacities": self.head_opacity(h).squeeze(-1),
            "colors": torch.sigmoid(self.head_color(h)),
        }


# ── order-free set loss (chamfer on positions + matched attributes) ────

def hungarian_set_loss(pred, target_means, target_attrs, n_target):
    """DETR-style one-to-one set loss via optimal bipartite matching.

    Each target stone is matched to a UNIQUE predicted slot (Hungarian on the
    position cost matrix), so slots can't pile onto the same stone -- the blob
    failure the Chamfer loss couldn't prevent. Matched slots are supervised on
    position + attributes; the active mask fires exactly the matched slots.

    Needs scipy; caller falls back to chamfer_set_loss if unavailable.
    """
    from scipy.optimize import linear_sum_assignment
    max_g = pred["means"].shape[0]
    pm = pred["means"]                                  # [max_g,3]
    M = target_means.shape[0]
    d = torch.cdist(pm, target_means)                   # [max_g, M]
    # optimal assignment on the detached cost; supervise with live tensors.
    row, col = linear_sum_assignment(d.detach().cpu().numpy())  # len min(max_g,M)=M
    row = torch.as_tensor(row, device=pm.device, dtype=torch.long)
    col = torch.as_tensor(col, device=pm.device, dtype=torch.long)

    active_tgt = torch.zeros(max_g, device=pm.device)
    active_tgt[row] = 1.0
    loss_active = F.binary_cross_entropy_with_logits(pred["active"], active_tgt)

    loss_pos = F.mse_loss(pm[row], target_means[col])
    loss_scale = F.mse_loss(pred["scales_log"][row], target_attrs["scales_log"][col])
    loss_color = F.mse_loss(pred["colors"][row], target_attrs["colors"][col])
    loss_opac = F.mse_loss(pred["opacities"][row], target_attrs["opacities"][col])
    return loss_pos + 0.5 * loss_scale + 0.5 * loss_color + 0.2 * loss_opac + loss_active


def chamfer_set_loss(pred, target_means, target_attrs, n_target):
    """pred: model output dict (single example, [max_g, *]).
    target_means [M,3], target_attrs dict of [M,*], n_target=M.

    Truly bidirectional (symmetric) Chamfer:
      - BACKWARD (coverage): each target is pulled onto its nearest pred slot.
      - FORWARD (anti-collapse): each pred slot is pulled onto its nearest
        target, weighted by how "active" the slot wants to be. Without this
        term most of the 512 slots get no geometry gradient and collapse to a
        few averaged points (the "blobs" failure). This is the key fix.
    Attributes are matched on the coverage direction; BCE trains the active mask.
    """
    max_g = pred["means"].shape[0]
    pm = pred["means"]                                  # [max_g,3]
    d = torch.cdist(pm, target_means)                   # [max_g, M]

    # backward / coverage: target -> nearest pred slot
    nn_p = d.argmin(0)                                  # [M] pred slot per target
    # forward: pred slot -> nearest target (distance kept, not just index)
    fwd_min, _ = d.min(1)                               # [max_g] euclidean dist

    # active mask: slots that cover some target should fire
    active_tgt = torch.zeros(max_g, device=pm.device)
    active_tgt[nn_p] = 1.0
    loss_active = F.binary_cross_entropy_with_logits(pred["active"], active_tgt)

    # geometry — coverage (backward)
    loss_cov = F.mse_loss(pm[nn_p], target_means)
    # geometry — forward, weighted by (detached) activeness so inactive slots
    # aren't forced onto the shape once the mask has settled.
    w = torch.sigmoid(pred["active"]).detach()          # [max_g]
    loss_fwd = (w * fwd_min.pow(2)).sum() / (w.sum() + 1e-6)

    # attributes on the coverage match
    idx = nn_p
    loss_scale = F.mse_loss(pred["scales_log"][idx], target_attrs["scales_log"])
    loss_color = F.mse_loss(pred["colors"][idx], target_attrs["colors"])
    loss_opac = F.mse_loss(pred["opacities"][idx], target_attrs["opacities"])
    return (loss_cov + loss_fwd + 0.5 * loss_scale + 0.5 * loss_color
            + 0.2 * loss_opac + loss_active)


# ── data ───────────────────────────────────────────────────────────────

def load_examples(path, device, max_g):
    raw = json.load(open(path))
    out = []
    for e in raw:
        g = e["gaussians"]
        m = torch.tensor(g["means"], dtype=torch.float32, device=device)
        if m.shape[0] > max_g:
            m = m[:max_g]
        attrs = {
            "scales_log": torch.tensor(g["scales_log"], dtype=torch.float32, device=device)[:max_g],
            "colors": torch.tensor(g["colors"], dtype=torch.float32, device=device)[:max_g],
            "opacities": torch.tensor(g["opacities"], dtype=torch.float32, device=device)[:max_g],
        }
        pose = torch.tensor([e["params"]["courses"] / 10.0] + e["params"]["color"],
                            dtype=torch.float32, device=device)
        out.append({"part_id": PART_TO_ID[e["part"]], "pose": pose,
                    "means": m, "attrs": attrs, "n": m.shape[0]})
    return out


# ── train ──────────────────────────────────────────────────────────────

def selftest(device):
    """One forward+backward on random data: the set loss is differentiable and
    decreases on a single overfit example. CPU ok."""
    print("[selftest] FillModel forward/backward + chamfer set loss")
    torch.manual_seed(0)
    model = FillModel(max_gaussians=64).to(device)
    pid = torch.tensor([0], device=device)
    pose = torch.zeros(1, 4, device=device)
    tgt_m = torch.randn(20, 3, device=device)
    tgt_a = {"scales_log": torch.full((20, 3), -2.5, device=device),
             "colors": torch.rand(20, 3, device=device),
             "opacities": torch.full((20,), 2.0, device=device)}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Exercise the default (Hungarian) loss if scipy is present, else Chamfer.
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
        loss_fn, which = hungarian_set_loss, "hungarian"
    except ImportError:
        loss_fn, which = chamfer_set_loss, "chamfer"
    losses = []
    for _ in range(50):
        opt.zero_grad()
        out = model(pid, pose)
        single = {k: v[0] for k, v in out.items()}
        loss = loss_fn(single, tgt_m, tgt_a, 20)
        loss.backward(); opt.step()
        losses.append(loss.item())
    ok = losses[-1] < losses[0]
    print(f"[selftest] {which} loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({'DECREASING, grads flow' if ok else 'NOT decreasing -- bug'})")
    return ok


def train(args, device):
    data = load_examples(args.data, device, args.max_gaussians)
    print(f"[fill] {len(data)} examples, max_g={args.max_gaussians}, device={device}")
    model = FillModel(max_gaussians=args.max_gaussians).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    render_sup = args.render_supervision == "sds"
    guide = None
    if render_sup:
        # Stage B is NOT implemented: the render term below (line ~203) is still
        # a TODO, so loading SDSGuidance would pull in diffusers/gsplat (needs
        # .venv-sds) and then contribute nothing to the loss. Don't pay that
        # cost for a no-op. Warn and train Stage A (Chamfer) — identical result.
        print("[fill] WARNING: --render-supervision sds requested, but the Stage B "
              "render loss is not implemented yet (train_fill.py:~203). Training "
              "Stage A (Chamfer set-matching) instead. This runs in the main .venv "
              "(no diffusers/SDS needed). Implement the render term before Stage B "
              "can beat the grammar.")
        render_sup = False

    # Pick the set loss. Hungarian (one-to-one) prevents the slot-pileup blobs
    # that Chamfer allows; fall back to Chamfer if scipy isn't installed.
    loss_fn = chamfer_set_loss
    if args.match == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment  # noqa: F401
            loss_fn = hungarian_set_loss
            print("[fill] set loss: Hungarian (one-to-one bipartite match)")
        except ImportError:
            print("[fill] WARNING: --match hungarian needs scipy (pip install scipy); "
                  "falling back to Chamfer.")
    else:
        print("[fill] set loss: Chamfer (bidirectional)")

    import random
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(data)
        ep_loss = 0.0
        for ex in data:
            opt.zero_grad()
            pid = torch.tensor([ex["part_id"]], device=device)
            out = model(pid, ex["pose"].unsqueeze(0))
            single = {k: v[0] for k, v in out.items()}
            loss = loss_fn(single, ex["means"], ex["attrs"], ex["n"])
            # TODO Stage B: add an SDS render-score term here (against `guide`)
            # so parts LOOK right, not just match the grammar template. Until
            # then --render-supervision sds is a no-op (see the warning above).
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  ep {epoch:3d}  loss {ep_loss/len(data):.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.out) / "best.pt"
    torch.save({"model": model.state_dict(), "max_gaussians": args.max_gaussians,
                "part_kinds": PART_KINDS}, ckpt)
    print(f"[fill] saved -> {ckpt}")


def main():
    p = argparse.ArgumentParser(description="Train the learned fill model")
    p.add_argument("--data", default="data/fill/path1_fill.json")
    p.add_argument("--out", default="checkpoints/fill_model")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-gaussians", type=int, default=512)
    p.add_argument("--match", choices=["hungarian", "chamfer"], default="hungarian",
                   help="Set-matching loss. hungarian = one-to-one (best, needs "
                        "scipy); chamfer = bidirectional nearest-neighbour.")
    p.add_argument("--render-supervision", choices=["none", "sds"], default="none")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.selftest:
        sys.exit(0 if selftest(device) else 1)
    train(args, device)


if __name__ == "__main__":
    main()
