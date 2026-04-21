"""
Train Radiance Planck 1.1 — Hierarchical SGS with Knowledge Blobs.

End-to-end: build blobs (if needed) + train with two-pass rendering.

Usage (Windows, RTX 4090):
    python scripts/train_planck11.py
    python scripts/train_planck11.py --base-checkpoint checkpoints/planck/best.pt
    python scripts/train_planck11.py --freeze-base --epochs 1   # blob-only warmup
    python scripts/train_planck11.py --t-max 0.0                # ablation: blobs disabled
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel
from src.blob_store import BlobStore
from src.sgs_lm_hsgs import HSGSLanguageModel
from src.tinystories import prepare_data, get_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Train Radiance Planck 1.1 (H-SGS)")

    # Data
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--vocab-size", type=int, default=32000)

    # Base model (Planck 1.0)
    p.add_argument("--base-checkpoint", default="checkpoints/planck/best.pt")
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # Blob store
    p.add_argument("--n-blobs", type=int, default=50000)
    p.add_argument("--blob-k", type=int, default=8, help="Top-k blobs to retrieve")
    p.add_argument("--t-max", type=float, default=0.3, help="Max blob transmittance budget")
    p.add_argument("--blob-dir", default="data/blobs/tinystories",
                   help="Pre-built blob store (or will be built)")
    p.add_argument("--blob-chunk-size", type=int, default=128)
    p.add_argument("--max-chunks", type=int, default=500000)

    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Lower than Planck 1.0 since we're fine-tuning")
    p.add_argument("--blob-lr", type=float, default=1e-3,
                   help="Separate (higher) LR for blob parameters")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--mixed-precision", default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--freeze-base", action="store_true",
                   help="Freeze base model, train only blob params (warmup phase)")

    # Logging & checkpointing
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="radiance-planck-11")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--save-dir", default="checkpoints/planck11")

    # Resume
    p.add_argument("--resume", type=str, default=None)

    # Workers
    p.add_argument("--num-workers", type=int, default=0)

    return p.parse_args()


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device, amp_dtype):
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
    print("RADIANCE PLANCK 1.1 — Hierarchical SGS (Knowledge Splatting)")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
        if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("  bf16 not supported, falling back to fp16")
            args.mixed_precision = "fp16"

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Data
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 1: Data preparation")
    print(f"{'='*60}")

    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"

    if train_bin.exists() and val_bin.exists():
        print(f"  Data already prepared in {data_dir}")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(data_dir / "tokenizer.model"))
        actual_vocab = sp.get_piece_size()
    else:
        print("  Preparing TinyStories data...")
        data = prepare_data(str(data_dir), args.vocab_size, args.context_len)
        actual_vocab = data["vocab_size"]

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Build blobs (if needed)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 2: Knowledge blob construction")
    print(f"{'='*60}")

    blob_path = Path(args.blob_dir) / "blobs.pt"
    if blob_path.exists():
        print(f"  Loading pre-built blobs from {blob_path}")
        blob_data = torch.load(blob_path, map_location="cpu", weights_only=False)
        n_blobs = blob_data["mu"].shape[0]
        print(f"  {n_blobs:,} blobs loaded")
    else:
        print("  Building blobs from training data...")
        import subprocess
        cmd = [
            sys.executable, "scripts/build_blobs.py",
            "--data-dir", str(args.data_dir),
            "--checkpoint", args.base_checkpoint,
            "--n-blobs", str(args.n_blobs),
            "--chunk-size", str(args.blob_chunk_size),
            "--output", args.blob_dir,
            "--max-chunks", str(args.max_chunks),
            "--d-s", str(args.d_s),
            "--d-f", str(args.d_f),
            "--n-passes", str(args.n_passes),
            "--n-heads", str(args.n_heads),
            "--context-len", str(args.context_len),
        ]
        subprocess.run(cmd, check=True)
        blob_data = torch.load(blob_path, map_location="cpu", weights_only=False)
        n_blobs = blob_data["mu"].shape[0]

    # ══════════════════════════════════════════════════════════
    # PHASE 3: Build model
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("PHASE 3: Model construction")
    print(f"{'='*60}")

    # Create blob store
    blob_store = BlobStore(
        n_blobs=n_blobs,
        d_s=args.d_s,
        d_f=args.d_f,
        k=args.blob_k,
        t_max=args.t_max,
    )
    blob_store.init_from_clusters(
        mu=blob_data["mu"],
        log_var=blob_data["log_var"],
        alpha=blob_data["raw_alpha"],
        features=blob_data["features"],
    )

    # Load base model and wrap
    model = HSGSLanguageModel.from_pretrained(
        args.base_checkpoint,
        blob_store,
        device=device,
        d_s=args.d_s, d_f=args.d_f,
        n_passes=args.n_passes, n_heads=args.n_heads,
        max_len=args.context_len, ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    )

    n_params = model.count_parameters()
    base_params = model.base.count_parameters()
    blob_params = model.blobs.count_parameters()
    print(f"  Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"    Base (Planck 1.0): {base_params:,} ({base_params/1e6:.1f}M)")
    print(f"    Blobs: {blob_params:,} ({blob_params/1e6:.1f}M, {blob_params/n_params*100:.0f}%)")
    print(f"  Blob config: k={args.blob_k}, t_max={args.t_max}, n_blobs={n_blobs:,}")

    if args.freeze_base:
        print("  FREEZING base model (blob-only training)")
        for param in model.base.parameters():
            param.requires_grad = False

    # ── Optimizer: separate LR for base vs blob params ──
    base_decay, base_nodecay, blob_all = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("blobs.") or name.startswith("blob_"):
            blob_all.append(param)
        elif "ln_f" in name or "bias" in name or "log_tau" in name:
            base_nodecay.append(param)
        else:
            base_decay.append(param)

    param_groups = []
    if base_decay:
        param_groups.append({"params": base_decay, "lr": args.lr, "weight_decay": args.weight_decay})
    if base_nodecay:
        param_groups.append({"params": base_nodecay, "lr": args.lr, "weight_decay": 0.0})
    if blob_all:
        param_groups.append({"params": blob_all, "lr": args.blob_lr, "weight_decay": 0.01})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    # ── Data loaders ──
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
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total steps: {total_steps:,}")

    # ── LR schedule ──
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=args.warmup_steps)
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

        # Optimizer/scheduler state only transfers when the param-group layout
        # matches. The common mismatch: checkpoint was saved with
        # --freeze-base (one or two groups) but the resume run trains all
        # params (three groups), or vice versa. In that case, fall back to
        # weights-only resume with a fresh optimizer+scheduler.
        ckpt_groups = len(ckpt["optimizer"]["param_groups"])
        curr_groups = len(optimizer.param_groups)
        if ckpt_groups == curr_groups:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0)
            global_step = ckpt.get("global_step", 0)
            print(f"  Restored optimizer + scheduler (epoch {start_epoch}, step {global_step})")
        else:
            print(
                f"  WARN: param-group count differs "
                f"(checkpoint={ckpt_groups}, current={curr_groups}). "
                f"Falling back to weights-only resume: optimizer and scheduler reset, "
                f"epoch/global_step reset to 0. Pass matching --freeze-base to preserve."
            )

    # ── Wandb ──
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args) | {"n_params": n_params, "actual_vocab": actual_vocab},
            )
        except ImportError:
            args.wandb = False

    # ── Checkpoints ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # PHASE 4: Training
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"PHASE 4: Training ({args.epochs} epochs)")
    print(f"{'='*60}")

    model.train()
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

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

            batch_tokens = x.numel()
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens

            # ── Log ──
            if global_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                blob_lr = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else lr
                avg = epoch_loss / epoch_tokens
                elapsed = time.time() - t0
                tok_per_sec = epoch_tokens / elapsed
                print(
                    f"  epoch {epoch+1} step {global_step:>6d} | "
                    f"loss {loss.item():.4f} avg {avg:.4f} | "
                    f"lr {lr:.2e} blob_lr {blob_lr:.2e} | "
                    f"gnorm {grad_norm:.2f} | {tok_per_sec:.0f} tok/s"
                )
                if args.wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg,
                        "train/lr": lr,
                        "train/blob_lr": blob_lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tok_per_sec,
                    }, step=global_step)

            # ── Eval ──
            if global_step % args.eval_interval == 0:
                val_loss, val_ppl = evaluate(model, val_loader, args.eval_steps, device, amp_dtype)
                print(f"  >>> val loss {val_loss:.4f} ppl {val_ppl:.1f}")
                if args.wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save(model, optimizer, scheduler, scaler, epoch, global_step,
                          save_dir / "best.pt")
                model.train()

            # ── Checkpoint ──
            if global_step % args.save_interval == 0:
                _save(model, optimizer, scheduler, scaler, epoch, global_step,
                      save_dir / f"step_{global_step}.pt")

        epoch_avg = epoch_loss / max(epoch_tokens, 1)
        print(f"  Epoch {epoch+1} done | avg loss {epoch_avg:.4f} | ppl {math.exp(min(epoch_avg, 20)):.1f}")
        _save(model, optimizer, scheduler, scaler, epoch + 1, global_step,
              save_dir / f"epoch_{epoch+1}.pt")

    # ── Final ──
    print(f"\n{'='*60}")
    print("Final evaluation")
    print(f"{'='*60}")
    val_loss, val_ppl = evaluate(model, val_loader, len(val_loader), device, amp_dtype)
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Final val ppl:  {val_ppl:.1f}")
    _save(model, optimizer, scheduler, scaler, args.epochs, global_step,
          save_dir / "final.pt")
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
