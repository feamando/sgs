"""
Training script for Radiance Planck — SGS causal language model.

Usage:
    python scripts/train_lm.py --data-dir data/tinystories
    python scripts/train_lm.py --data-dir data/tinystories --wandb

Designed for RTX 4090 (24GB VRAM) on Windows + CUDA.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel
from src.tinystories import prepare_data, get_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train Radiance Planck")

    # Data
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--vocab-size", type=int, default=32000)

    # Architecture
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop training after this many global steps (0 = no cap)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "fp32"])

    # Logging & checkpointing
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="radiance-planck")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--save-dir", default="checkpoints/planck")

    # Resume
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")

    # Workers
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)")

    # ── Planck 1.2 accel flags (§2.1-§2.4; see docs/plans/planck_12_plan.md) ──
    # §2.1 Transmittance-weighted loss
    p.add_argument("--transmittance-loss", action="store_true",
                   help="§2.1: weight CE by (1-T[t,t])^gamma + floor regularizer")
    p.add_argument("--tl-gamma", type=float, default=1.5)
    p.add_argument("--tl-lambda", type=float, default=0.01)
    p.add_argument("--tl-tmax", type=float, default=0.3)

    # §2.2 Adaptive pass count
    p.add_argument("--adaptive-passes", action="store_true",
                   help="§2.2: exit multi-pass loop early when transmittance stabilises")
    p.add_argument("--ap-eps", type=float, default=0.02)
    p.add_argument("--ap-min-step", type=int, default=2000)

    # §2.3 Kernel top-k sparsity (Tier B)
    p.add_argument("--sparse-k", type=int, default=0,
                   help="§2.3: top-k keys per query; 0 disables")
    p.add_argument("--sparse-warmup-steps", type=int, default=5000)
    p.add_argument("--sparse-tau-gate", type=float, default=30.0)

    # §2.4 Shared kernel across passes
    p.add_argument("--shared-kernel", action="store_true",
                   help="§2.4: reuse pass-1 kernel in subsequent passes")

    # Profiling (bakes into stdout per plan; see planck_12_plan.md §resolved)
    p.add_argument("--log-profile", action="store_true",
                   help="Print per-substep wall times every log-interval steps")
    p.add_argument("--profile-step", type=int, default=-1,
                   help="torch.profiler wrap step N and dump top ops (-1 disables)")

    return p.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device, amp_dtype):
    """Run validation and return average loss + perplexity.

    Eval always uses plain CE regardless of §2.1 — transmittance-weighted
    loss is a training-time confidence signal, not the metric we want to
    compare across ablations. Keeps val numbers directly comparable.
    """
    was_returning = getattr(model, "return_accel_state", False)
    # Swap off accel-tuple return for the duration of eval so the CE call
    # sees plain logits without branching.
    if was_returning:
        model.return_accel_state = False
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    if was_returning:
        model.return_accel_state = True
    avg_loss = total_loss / max(n, 1)
    return avg_loss, math.exp(min(avg_loss, 20))  # cap perplexity to avoid overflow


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──
    print("\n=== Preparing data ===")
    data = prepare_data(args.data_dir, args.vocab_size, args.context_len)
    actual_vocab = data["vocab_size"]

    train_loader = get_dataloader(
        data["train_bin"], args.context_len, args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        data["val_bin"], args.context_len, args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")

    # ── Model ──
    print("\n=== Creating model ===")
    accel_active = (
        args.transmittance_loss
        or args.adaptive_passes
        or args.sparse_k > 0
        or args.shared_kernel
    )
    model = SGSLanguageModel(
        vocab_size=actual_vocab,
        d_s=args.d_s,
        d_f=args.d_f,
        n_passes=args.n_passes,
        n_heads=args.n_heads,
        max_len=args.context_len,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        return_accel_state=accel_active,
        adaptive_passes=args.adaptive_passes,
        ap_eps=args.ap_eps,
        ap_min_step=args.ap_min_step,
        sparse_k=args.sparse_k,
        sparse_warmup_steps=args.sparse_warmup_steps,
        sparse_tau_gate=args.sparse_tau_gate,
        shared_kernel=args.shared_kernel,
    ).to(device)
    if accel_active:
        flags_on = ", ".join([
            n for n, v in [
                ("transmittance-loss", args.transmittance_loss),
                ("adaptive-passes", args.adaptive_passes),
                (f"sparse-k={args.sparse_k}", args.sparse_k > 0),
                ("shared-kernel", args.shared_kernel),
            ] if v
        ])
        print(f"  Accel flags: {flags_on}")

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    breakdown = model.param_breakdown()
    for k, v in breakdown.items():
        if k != "total":
            print(f"    {k}: {v:,} ({v/n_params*100:.1f}%)")

    # ── Optimizer ──
    # Separate weight decay for embeddings vs other params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ln_f" in name or "bias" in name or "log_tau" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    # ── LR Schedule ──
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - args.warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])

    # ── Mixed precision ──
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.mixed_precision]
    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision == "fp16" else None
    print(f"  Mixed precision: {args.mixed_precision}")

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

    # ── Wandb ──
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args) | {"n_params": n_params, "actual_vocab": actual_vocab},
            )
            wandb.watch(model, log_freq=1000)
        except ImportError:
            print("  wandb not installed, logging disabled")
            args.wandb = False

    # ── Checkpoints ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print(f"\n=== Training ({args.epochs} epochs) ===")
    model.train()
    best_val_loss = float("inf")
    passes_run_ema = float(args.n_passes)

    def _compute_loss(x_batch, y_batch):
        """Forward + loss. Returns (loss, T_diag_mean_or_None, passes_run)."""
        model.opt_step = global_step
        out = model(x_batch)
        if accel_active:
            logits_, T_diag_, passes_ = out
        else:
            logits_, T_diag_, passes_ = out, None, args.n_passes

        flat_logits = logits_.view(-1, logits_.size(-1))
        flat_y = y_batch.view(-1)

        if args.transmittance_loss and T_diag_ is not None:
            # Per-token CE, reshape to [B, L], apply §2.1 weighting.
            ce_flat = F.cross_entropy(flat_logits, flat_y, reduction="none")
            ce = ce_flat.view_as(T_diag_)                      # [B, L]
            T_clamped = T_diag_.clamp(0.0, 1.0)
            weight = (1.0 - T_clamped).pow(args.tl_gamma)
            ce_w = (weight * ce).mean()
            reg = F.relu(T_clamped - args.tl_tmax).pow(2).mean() * args.tl_lambda
            l = ce_w + reg
        else:
            l = F.cross_entropy(flat_logits, flat_y)
        mean_T = T_diag_.mean().item() if T_diag_ is not None else None
        return l, mean_T, passes_

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # §2.3 / plan note: one-shot torch.profiler dump on the
            # requested step. Fires once per run.
            use_profiler = (
                args.profile_step >= 0 and global_step == args.profile_step
            )
            if use_profiler:
                prof_ctx = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                )
                prof_ctx.__enter__()

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                loss, T_mean, passes_run = _compute_loss(x, y)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if use_profiler:
                prof_ctx.__exit__(None, None, None)
                print("\n=== torch.profiler top ops (CUDA time) ===")
                print(prof_ctx.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20,
                ))

            batch_tokens = x.numel()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens
            # Smoothed passes_run so the log isn't noisy when adaptive-exit
            # flips between 2 and 3 across batches.
            passes_run_ema = 0.9 * passes_run_ema + 0.1 * float(passes_run)

            # ── Log ──
            if global_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg = epoch_loss / epoch_tokens
                elapsed = time.time() - t0
                tok_per_sec = epoch_tokens / elapsed
                extra = ""
                if accel_active:
                    parts = [f"passes {passes_run_ema:.2f}"]
                    if T_mean is not None:
                        parts.append(f"T {T_mean:.3f}")
                    extra = " | " + " ".join(parts)
                print(
                    f"  epoch {epoch+1} step {global_step:>6d} | "
                    f"loss {loss.item():.4f} avg {avg:.4f} | "
                    f"lr {lr:.2e} gnorm {grad_norm:.2f} | "
                    f"tau {model.tau.item():.1f} | "
                    f"{tok_per_sec:.0f} tok/s" + extra
                )
                if args.wandb:
                    import wandb
                    log_payload = {
                        "train/loss": loss.item(),
                        "train/avg_loss": avg,
                        "train/perplexity": math.exp(min(loss.item(), 20)),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/tau": model.tau.item(),
                        "train/tokens_per_sec": tok_per_sec,
                        "train/epoch": epoch + step / steps_per_epoch,
                    }
                    if accel_active:
                        log_payload["train/passes_run_ema"] = passes_run_ema
                        if T_mean is not None:
                            log_payload["train/T_mean"] = T_mean
                    wandb.log(log_payload, step=global_step)

            # ── Eval ──
            if global_step % args.eval_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, args.eval_steps, device, amp_dtype)
                print(f"  >>> val loss {val_loss:.4f} ppl {val_ppl:.1f}")
                if args.wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save(model, optimizer, scheduler, scaler, epoch, global_step, save_dir / "best.pt")
                model.train()

            # ── Checkpoint ──
            if global_step % args.save_interval == 0:
                _save(model, optimizer, scheduler, scaler, epoch, global_step,
                      save_dir / f"step_{global_step}.pt")

            if args.max_steps and global_step >= args.max_steps:
                print(f"  Reached --max-steps {args.max_steps}; stopping.")
                break

        # End of epoch
        epoch_avg = epoch_loss / max(epoch_tokens, 1)
        print(f"  Epoch {epoch+1} done | avg loss {epoch_avg:.4f} | ppl {math.exp(min(epoch_avg, 20)):.1f}")
        _save(model, optimizer, scheduler, scaler, epoch + 1, global_step,
              save_dir / f"epoch_{epoch+1}.pt")

        if args.max_steps and global_step >= args.max_steps:
            break

    # ── Final eval ──
    print("\n=== Final evaluation ===")
    val_loss, val_ppl = evaluate(model, val_loader, len(val_loader), device, amp_dtype)
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Final val ppl:  {val_ppl:.1f}")
    _save(model, optimizer, scheduler, scaler, args.epochs, global_step, save_dir / "final.pt")
    print(f"\nDone. Checkpoints in {save_dir}")

    if args.wandb:
        import wandb
        wandb.finish()


def _save(model, optimizer, scheduler, scaler, epoch, global_step, path):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if scaler:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
