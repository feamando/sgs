"""VSP v1, phase 3: train Planck 2.0 -- the first VSP-bundled SGS LM.

The novelty vs a normal SGS LM: a token's embedding is INITIALIZED from its
cached V/S/P bundle (Visual + Semantic + Physical), not a random lookup row. The
grounded meaning is injected before training, so the model spends capacity USING
the representation, not learning to disambiguate from scratch. That's the whole
efficiency bet (smaller model, multimodal understanding built in).

Wiring (honest + minimal): SGSLanguageModel already embeds each token via
tok_mu[d_s], tok_features[d_f] (+ log_var, raw_alpha). We initialize tok_mu and
tok_features from a linear projection of the per-token [V|S|P] bundle
(vsps/vocab.json). Grounded/abstract/subword tiers all have an S; only grounded
carry V+P. The bundle-projected init is the warm start; --freeze-vp-steps keeps
the projected part fixed early (mirrors Raum freezing the Planck encoder) so the
model learns to use it before it can corrupt it.

Baseline for the publishable comparison: train the SAME SGSLanguageModel at the
same params/tokens on a SentencePiece-tokenized corpus (random init). Planck 2.0
must beat it on the disambiguation benchmark (phase 4), not on val loss.

Lessons carried in: plain AdamW + stdout (accel shelved, wandb paid); hard
opt-step budget so a resumed run can't replay the epoch
([[project_hertz_resume_epoch_restart]]).

Usage:
  python scripts/train_planck2.py --tokens data/wiki_vsps \
    --vocab data/vsps/vocab.json --d-f 1000 --freeze-vp-steps 2000 \
    --save-dir checkpoints/planck2 --opt-steps 40000
  python scripts/train_planck2.py --selftest
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel


# ── VSPS data ────────────────────────────────────────────────────────────

def load_vocab_bundles(vocab_path):
    """Return (n_tokens, V, S, P) stacked arrays [n, dim] from vsps/vocab.json.
    Missing blocks are zero (abstract: no V/P; subword: no V/P)."""
    vj = json.load(open(vocab_path))
    dims = vj["dims"]
    toks = sorted(vj["tokens"], key=lambda t: t["id"])
    n = len(toks)
    V = np.zeros((n, dims["V"]), np.float32)
    S = np.zeros((n, dims["S"]), np.float32)
    P = np.zeros((n, dims["P"]), np.float32)
    for t in toks:
        i = t["id"]
        if t.get("V"): V[i] = t["V"]
        if t.get("S"): S[i] = t["S"]
        if t.get("P"): P[i] = t["P"]
    return n, V, S, P


class TokenStream(torch.utils.data.Dataset):
    """MEMMAP the tokens.bin (uint32/uint16 from tokenize_vsps) into non-
    overlapping context_len windows. Never loads the 2.1B-token stream into RAM
    (a JSON of that size needs ~60GB; the .bin memmaps at ~0 RAM)."""

    def __init__(self, tokens_dir, context_len):
        tdir = Path(tokens_dir)
        bin_path = tdir / "tokens.bin" if tdir.is_dir() else tdir
        meta_path = (tdir / "tokens_meta.json") if tdir.is_dir() else \
            bin_path.with_name("tokens_meta.json")
        dtype = "uint32"
        if meta_path.exists():
            dtype = json.load(open(meta_path)).get("dtype", "uint32")
        self.data = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
        self.ctx = context_len
        self.n = (len(self.data) - 1) // context_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.ctx
        # cast to int64 for the embedding lookup; copy out of the memmap
        x = torch.from_numpy(self.data[s:s + self.ctx].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1:s + 1 + self.ctx].astype(np.int64))
        return x, y


# ── VSP bundle init ───────────────────────────────────────────────────────

def init_from_bundles(model, V, S, P, device):
    """Initialize tok_mu (d_s) and tok_features (d_f) from a linear projection of
    the [V|S|P] bundle -- the grounded warm start (meaning injected before
    training). The projection is a throwaway seeding transform (not kept).

    Freezing is handled in the train loop by ZEROING these two tables' grads
    (NOT requires_grad toggling + add_param_group, which desyncs the LR
    scheduler). So all params stay in the optimizer from step 0."""
    dV, dS, dP = V.shape[1], S.shape[1], P.shape[1]
    d_s = model.tok_mu.embedding_dim
    d_f = model.tok_features.embedding_dim
    bundle = torch.tensor(np.concatenate([V, S, P], axis=1), device=device)  # [n, dV+dS+dP]
    with torch.no_grad():
        proj_mu = torch.nn.Linear(dV + dS + dP, d_s).to(device)
        proj_feat = torch.nn.Linear(dV + dS + dP, d_f).to(device)
        model.tok_mu.weight.copy_(proj_mu(bundle))
        model.tok_features.weight.copy_(proj_feat(bundle))


def selftest():
    """Tiny end-to-end: 40-token vocab-like bundles -> model -> one train step."""
    print("[selftest] train_planck2")
    torch.manual_seed(0)
    n, dV, dS, dP = 40, 8, 6, 4
    V = np.random.randn(n, dV).astype(np.float32)
    S = np.random.randn(n, dS).astype(np.float32)
    P = np.random.randn(n, dP).astype(np.float32)
    model = SGSLanguageModel(vocab_size=n, d_s=16, d_f=32, n_passes=2, n_heads=2, max_len=32)
    before = model.tok_mu.weight.detach().clone()
    init_from_bundles(model, V, S, P, "cpu")
    assert not torch.equal(before, model.tok_mu.weight), "bundle init did not change tok_mu"
    x = torch.randint(0, n, (2, 16))
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, n), x.view(-1))
    loss.backward()
    ok = torch.isfinite(loss) and logits.shape == (2, 16, n)
    print(f"[selftest] loss {loss.item():.3f} logits {tuple(logits.shape)} "
          f"| {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def parse_args():
    p = argparse.ArgumentParser(description="Train Planck 2.0 (VSP-bundled SGS LM)")
    p.add_argument("--tokens", default="data/wiki_vsps", help="dir with tokens.bin + tokens_meta.json")
    p.add_argument("--vocab", default="data/vsps/vocab.json")
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--opt-steps", type=int, default=40000, help="hard budget")
    p.add_argument("--freeze-vp-steps", type=int, default=2000,
                   help="keep the bundle-seeded embeddings frozen for N opt-steps")
    p.add_argument("--random-init", action="store_true",
                   help="SKIP the VSP bundle init (random embeddings) -- the "
                        "matched-compute BASELINE for the disambiguation gate")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--save-dir", default="checkpoints/planck2")
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_tokens, V, S, P = load_vocab_bundles(args.vocab)
    print(f"[p2] vocab: {n_tokens:,} tokens (V{V.shape[1]}/S{S.shape[1]}/P{P.shape[1]})")

    ds = TokenStream(args.tokens, args.context_len)
    print(f"[p2] corpus: {len(ds):,} windows of {args.context_len} "
          f"({len(ds)*args.context_len:,} tokens)")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=(device.type == "cuda"))

    model = SGSLanguageModel(
        vocab_size=n_tokens, d_s=args.d_s, d_f=args.d_f, n_passes=args.n_passes,
        n_heads=args.n_heads, max_len=args.context_len, ffn_mult=args.ffn_mult).to(device)
    print(f"[p2] params: {model.count_parameters()/1e6:.1f}M")

    if not args.random_init:
        init_from_bundles(model, V, S, P, device)
        print(f"[p2] VSP bundle init ON (freeze-vp {args.freeze_vp_steps} steps)")
    else:
        print("[p2] RANDOM init (baseline mode)")

    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.mixed_precision]
    # ALL params in the optimizer from step 0 (never add/remove groups mid-run --
    # that desyncs SequentialLR). The grounded warm start is implemented by
    # ZEROING the bundle-seeded embeddings' grads while opt_step < freeze_vp_steps.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, args.warmup_steps)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.opt_steps - args.warmup_steps, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [args.warmup_steps])
    frozen_tables = [] if args.random_init else [model.tok_mu.weight, model.tok_features.weight]

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    opt_step, micro, t0 = 0, 0, time.time()
    announced_unfreeze = False
    opt.zero_grad(set_to_none=True)
    done = False
    while not done:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device.type, dtype=amp, enabled=amp != torch.float32):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, n_tokens), y.view(-1)) / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum == 0:
                # grounded warm start: kill the seeded tables' grads while frozen
                if opt_step < args.freeze_vp_steps:
                    for w in frozen_tables:
                        if w.grad is not None:
                            w.grad.zero_()
                elif frozen_tables and not announced_unfreeze:
                    print(f"[p2] bundle embeddings now training (step {opt_step})", flush=True)
                    announced_unfreeze = True
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.log_interval == 0:
                    rate = opt_step / max(time.time() - t0, 1e-6)
                    print(f"[p2] step {opt_step}/{args.opt_steps} | loss "
                          f"{loss.item()*args.grad_accum:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                          f"| {rate:.2f} step/s", flush=True)
                if opt_step % args.save_interval == 0:
                    _save(model, opt_step, save_dir / f"step_{opt_step}.pt", args)
                if opt_step >= args.opt_steps:
                    done = True
                    break
    _save(model, opt_step, save_dir / "final.pt", args)
    print(f"[p2] done. checkpoints in {save_dir}")


def _save(model, step, path, args):
    torch.save({"model": model.state_dict(), "opt_step": step,
                "arch": {"d_s": args.d_s, "d_f": args.d_f, "n_passes": args.n_passes,
                         "n_heads": args.n_heads, "max_len": args.context_len,
                         "ffn_mult": args.ffn_mult}}, path)
    print(f"[p2] saved {path}", flush=True)


if __name__ == "__main__":
    main()
