"""
Radiance Hertz — end-to-end: download FineWeb-Edu + train 1B SGS language model.

Single script so you can kick it off and walk away.

Usage (Windows, RTX 4090):
    python scripts/train_hertz.py
    python scripts/train_hertz.py --max-tokens 1B --epochs 1      # quick test
    python scripts/train_hertz.py --resume checkpoints/hertz/best.pt
    python scripts/train_hertz.py --wandb
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
from src.tinystories import prepare_fineweb, get_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train Radiance Hertz (~640M-1B)")

    # Data
    p.add_argument("--data-dir", default="data/fineweb")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-tokens", default="2B",
                   help="Max tokens to download (2B default — feasible on single GPU)")

    # Architecture — Hertz 1B defaults
    p.add_argument("--d-s", type=int, default=256)
    p.add_argument("--d-f", type=int, default=5000,
                   help="Feature dim (5000 with 4 heads/3 passes → 1.04B params)")
    p.add_argument("--n-passes", type=int, default=3,
                   help="Rendering passes (3 not 5 — 40%% faster, fits VRAM)")
    p.add_argument("--n-heads", type=int, default=4,
                   help="Attention heads (4 not 8 — halves kernel compute)")
    p.add_argument("--context-len", type=int, default=512,
                   help="Context length (512 not 1024 — 4x smaller kernel matrices)")
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad-checkpoint", action="store_true", default=False,
                   help="Gradient checkpointing. Off by default: at B=2, L=512, H=4, "
                        "d_f=5000 activations fit in ~12GB of the 24GB 4090. "
                        "Enabling it blocks CUDA graphs (RNG state capture clash, "
                        "pytorch#162504) and costs ~30-40%% throughput.")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")

    # Training
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-GPU micro-batch (2 for 1B on 24GB VRAM)")
    p.add_argument("--grad-accum", type=int, default=32,
                   help="Gradient accumulation steps (effective batch = 2 * 32 = 64)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1,
                   help="Epochs (1 is standard for 10B tokens)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--mixed-precision", default="bf16",
                   choices=["bf16", "fp16", "fp32"])

    # Logging & checkpointing
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="radiance-hertz")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=5000)
    p.add_argument("--save-dir", default="checkpoints/hertz")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--warm-start", type=str, default=None,
                   help="Load model weights only (no optimizer, scheduler, "
                        "or step counters). Use when --resume throughput "
                        "collapses from Adam state reload. Step counters "
                        "restart at 0; LR re-warms up.")

    # Workers
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 for Windows)")

    # Diagnostics (throughput debugging)
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile. Useful on Windows where "
                        "triton-windows may add overhead without the Linux "
                        "Inductor speedups.")
    p.add_argument("--profile-steps", type=int, default=0,
                   help="If > 0, run torch.profiler for this many optimizer "
                        "steps at the start of training and dump a Chrome "
                        "trace to <save_dir>/trace.json, then exit. Use to "
                        "identify whether the bottleneck is kernel compute, "
                        "dataloader starvation, or a CPU/GPU sync.")

    return p.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device, amp_dtype):
    """Run validation and return average loss + perplexity."""
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
    avg_loss = total_loss / max(n, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("RADIANCE HERTZ — 1B SGS Language Model")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram_gb:.1f} GB")

        # RTX 4090 supports bf16, but check anyway
        if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("  bf16 not supported, falling back to fp16")
            args.mixed_precision = "fp16"

        # bf16 reduction in cuBLAS picks tensor-core paths for small-K bmms
        # (pytorch#120750). The rendering kernel does [B*H, L, d_s] x [B*H, d_s, L]
        # with d_s=256, L=512, which benefits from this.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        # cuBLASLt often selects better algos for bf16 bmm on Ada than cuBLAS.
        # Note: on Windows this setter emits a warning and is a no-op
        # (torch issue: unsupported on Windows), so we only claim it's on
        # when we're actually on Linux.
        import sys as _sys
        if hasattr(torch.backends.cuda, "preferred_blas_library") and _sys.platform != "win32":
            try:
                torch.backends.cuda.preferred_blas_library(backend="cublaslt")
                print("  BLAS: cuBLASLt")
            except Exception:
                pass
        elif _sys.platform == "win32":
            print("  BLAS: default (cuBLASLt preference unsupported on Windows)")

    # ── Parse max-tokens ──
    max_tok_str = args.max_tokens.upper().replace("B", "000000000").replace("M", "000000")
    max_tokens = int(max_tok_str)

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Data — download + tokenize (skips if already done)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Data preparation")
    print(f"{'='*60}")

    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    tok_model = data_dir / "tokenizer.model"

    if train_bin.exists() and val_bin.exists() and tok_model.exists():
        # Data already prepared — load vocab size from tokenizer
        print(f"  Data already prepared in {data_dir}")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(tok_model))
        actual_vocab = sp.get_piece_size()
        n_train = os.path.getsize(train_bin) // 2  # uint16
        n_val = os.path.getsize(val_bin) // 2
        print(f"  Train: {n_train:,} tokens, Val: {n_val:,} tokens")
        print(f"  Vocab: {actual_vocab}")
    else:
        print(f"  Downloading and preparing FineWeb-Edu ({args.max_tokens} tokens)...")
        data = prepare_fineweb(
            str(data_dir), args.vocab_size, args.context_len, max_tokens
        )
        train_bin = data["train_bin"]
        val_bin = data["val_bin"]
        actual_vocab = data["vocab_size"]

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Model + Training
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Training")
    print(f"{'='*60}")

    train_loader = get_dataloader(
        str(train_bin), args.context_len, args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        str(val_bin), args.context_len, args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    effective_batch = args.batch_size * args.grad_accum
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} accum = {effective_batch} effective")
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Optimizer steps: {total_steps // args.grad_accum:,}")

    # ── Model ──
    print("\n  Creating model...")
    model = SGSLanguageModel(
        vocab_size=actual_vocab,
        d_s=args.d_s,
        d_f=args.d_f,
        n_passes=args.n_passes,
        n_heads=args.n_heads,
        max_len=args.context_len,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        use_checkpoint=args.grad_checkpoint,
    ).to(device)
    if args.grad_checkpoint:
        print("  Gradient checkpointing: ON (saves VRAM, recomputes during backward)")

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M / {n_params/1e9:.2f}B)")
    breakdown = model.param_breakdown()
    for k, v in breakdown.items():
        if k != "total":
            print(f"    {k}: {v:,} ({v/n_params*100:.1f}%)")

    # ── Optimizer ──
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
        fused=(device.type == "cuda"),
    )

    # ── LR Schedule (based on optimizer steps, not micro-steps) ──
    opt_steps = total_steps // args.grad_accum
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=args.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(opt_steps - args.warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             milestones=[args.warmup_steps])

    # ── Mixed precision ──
    amp_dtype = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
    }[args.mixed_precision]
    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision == "fp16" else None
    print(f"  Mixed precision: {args.mixed_precision}")

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    opt_step = 0
    if args.resume and args.warm_start:
        raise SystemExit("Pass either --resume or --warm-start, not both.")
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        opt_step = ckpt.get("opt_step", global_step // args.grad_accum)
        if scaler and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
    elif args.warm_start:
        # Weights only. Fresh optimizer state (momentum re-builds in ~200
        # opt steps, cheap) and fresh LR schedule (re-warms up). Sidesteps
        # the Adam-state-reload throughput collapse observed on Windows.
        print(f"  Warm start from {args.warm_start} (weights only)")
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)

    # ── torch.compile (after resume so state_dict keys match) ──
    # We use mode="default" (Inductor kernel fusion, no CUDA graphs). CUDA
    # graphs (reduce-overhead / max-autotune) are fundamentally incompatible
    # with gradient accumulation in this script: we queue grad_accum=32
    # forwards' activations before the matching backwards run, and CUDA graph
    # buffer reuse overwrites still-live saved tensors. cudagraph_mark_step_begin
    # does not reliably fix this for multi-forward accumulation patterns, and
    # additionally clashes with gradient checkpointing (pytorch#162504 RNG
    # state capture). The Apr-15 e2956ff baseline that hit ~10k tok/s used
    # default mode (no CUDA graphs), so this restores that configuration.
    use_compile = hasattr(torch, "compile") and device.type == "cuda"
    if args.no_compile:
        use_compile = False
        print("  torch.compile: OFF (--no-compile)")
    if use_compile:
        try:
            import triton  # noqa: F401
        except ImportError:
            use_compile = False
            print("  torch.compile: OFF (Triton not available)")
    if use_compile:
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile: ON (mode=default, kernel fusion, no CUDA graphs)")
        except Exception as e:
            print(f"  torch.compile: FAILED ({e}), continuing without")

    # ── Wandb ──
    # We do NOT call wandb.watch(): it registers per-submodule forward/backward
    # hooks which both add overhead and break torch.compile fullgraph tracing.
    # tau and grad_norm are logged manually below (search for wandb.log).
    # _disable_stats=True kills the system-stats daemon thread.
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args) | {"n_params": n_params, "actual_vocab": actual_vocab},
                settings=wandb.Settings(_disable_stats=True),
            )
        except ImportError:
            print("  wandb not installed, logging disabled")
            args.wandb = False

    # ── Checkpoints ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    # Throughput measurement:
    #   epoch_loss_sum / epoch_tokens = cumulative loss (for end-of-epoch summary)
    #   window_tokens / window_elapsed = rolling tok/s over the last log_interval
    #     optimizer steps (surfaces mid-epoch regressions that a cumulative
    #     average smears out).
    # Loss handling:
    #   We accumulate loss as a GPU tensor (loss_accum) and sync only at
    #   log_interval, avoiding a per-micro-step loss.item() that forces CPU↔GPU
    #   syncs and stalls the pipeline.
    print(f"\n  Starting training ({args.epochs} epochs)...")
    model.train()
    best_val_loss = float("inf")

    # ── Profiler (diagnostic mode, exits after N opt steps) ──
    # When --profile-steps > 0, wrap the training loop in torch.profiler and
    # dump a Chrome trace. Purpose: figure out where the ~5x throughput gap
    # between e2956ff (~10k tok/s, d_f=3700) and today (~2k tok/s, d_f=5000)
    # is coming from. The d_f bump explains ~1.8x; this surfaces the rest.
    profiler_ctx = None
    profile_micro_steps = args.profile_steps * args.grad_accum if args.profile_steps else 0
    if args.profile_steps > 0:
        trace_path = str(save_dir / "trace.json")
        print(f"  Profiler: ON ({args.profile_steps} opt steps "
              f"= {profile_micro_steps} micro-steps, trace → {trace_path})")
        profiler_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=2, active=max(profile_micro_steps - 3, 1), repeat=1,
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
            record_shapes=False,
            with_stack=False,
        )
        profiler_ctx.__enter__()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        window_tokens = 0
        window_t0 = time.time()
        loss_accum = torch.zeros((), device=device)
        loss_accum_tokens = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                    enabled=amp_dtype != torch.float32):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss_scaled = loss / args.grad_accum

            # Backward
            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            global_step += 1
            batch_tokens = x.numel()
            # Keep the running sum on-device; no .item() here.
            loss_accum = loss_accum + loss.detach() * batch_tokens
            loss_accum_tokens += batch_tokens
            epoch_tokens += batch_tokens
            window_tokens += batch_tokens

            if profiler_ctx is not None:
                profiler_ctx.step()
                if global_step >= profile_micro_steps:
                    profiler_ctx.__exit__(None, None, None)
                    print(f"\n  Profiler: wrote trace to {save_dir / 'trace.json'}")
                    print(f"  View at chrome://tracing or ui.perfetto.dev")
                    return

            # Optimizer step every grad_accum micro-steps
            if global_step % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                # ── Log ──
                if opt_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    # Single device sync for the whole log interval.
                    interval_loss_sum = loss_accum.item()
                    interval_avg = interval_loss_sum / max(loss_accum_tokens, 1)
                    epoch_loss_sum += interval_loss_sum
                    loss_accum = torch.zeros((), device=device)
                    loss_accum_tokens = 0

                    window_elapsed = time.time() - window_t0
                    tok_per_sec = window_tokens / max(window_elapsed, 1e-9)
                    window_tokens = 0
                    window_t0 = time.time()

                    epoch_avg = epoch_loss_sum / max(epoch_tokens, 1)
                    print(
                        f"  epoch {epoch+1} opt_step {opt_step:>7d} | "
                        f"loss {interval_avg:.4f} avg {epoch_avg:.4f} | "
                        f"lr {lr:.2e} gnorm {grad_norm:.2f} | "
                        f"tau {model.tau.item():.1f} | "
                        f"{tok_per_sec:.0f} tok/s"
                    )
                    if args.wandb:
                        import wandb
                        wandb.log({
                            "train/loss": interval_avg,
                            "train/avg_loss": epoch_avg,
                            "train/perplexity": math.exp(min(interval_avg, 20)),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/tau": model.tau.item(),
                            "train/tokens_per_sec": tok_per_sec,
                            "train/epoch": epoch + step / steps_per_epoch,
                        }, step=opt_step)

                # ── Eval ──
                if opt_step % args.eval_interval == 0:
                    val_loss, val_ppl = evaluate(
                        model, val_loader, args.eval_steps, device, amp_dtype)
                    print(f"  >>> val loss {val_loss:.4f} ppl {val_ppl:.1f}")
                    if args.wandb:
                        import wandb
                        wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl},
                                  step=opt_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save(model, optimizer, scheduler, scaler,
                              epoch, global_step, opt_step, save_dir / "best.pt")
                    model.train()

                # ── Checkpoint ──
                if opt_step % args.save_interval == 0:
                    _save(model, optimizer, scheduler, scaler,
                          epoch, global_step, opt_step,
                          save_dir / f"step_{opt_step}.pt")

        # End of epoch
        # Fold any remaining (sub-log-interval) loss into the epoch total.
        if loss_accum_tokens > 0:
            epoch_loss_sum += loss_accum.item()
            loss_accum = torch.zeros((), device=device)
            loss_accum_tokens = 0
        epoch_avg = epoch_loss_sum / max(epoch_tokens, 1)
        print(f"  Epoch {epoch+1} done | avg loss {epoch_avg:.4f} | "
              f"ppl {math.exp(min(epoch_avg, 20)):.1f}")
        _save(model, optimizer, scheduler, scaler,
              epoch + 1, global_step, opt_step,
              save_dir / f"epoch_{epoch+1}.pt")

    # ── Final eval ──
    print(f"\n{'='*60}")
    print("Final evaluation")
    print(f"{'='*60}")
    val_loss, val_ppl = evaluate(
        model, val_loader, len(val_loader), device, amp_dtype)
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Final val ppl:  {val_ppl:.1f}")
    _save(model, optimizer, scheduler, scaler,
          args.epochs, global_step, opt_step, save_dir / "final.pt")
    print(f"\nDone. Checkpoints in {save_dir}")

    if args.wandb:
        import wandb
        wandb.finish()


def _save(model, optimizer, scheduler, scaler, epoch, global_step, opt_step, path):
    # Unwrap torch.compile wrapper so checkpoints are always portable
    raw_model = getattr(model, "_orig_mod", model)
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "opt_step": opt_step,
    }
    if scaler:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
