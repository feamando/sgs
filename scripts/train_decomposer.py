"""
Train Planck as a decomposition predictor (Raum 1.3 Phase 2).

Fine-tunes the Planck LM on (prompt, tree_json) pairs so it learns to
generate composition trees from text descriptions. The model learns to
output valid JSON trees that can be parsed by src/raum/decomposition.py.

Training format per sample:
    input:  "DECOMPOSE: a castle on a hill\nTREE:"
    target: '{"name": "scene", "children": [...]}'

Usage:
    python scripts/train_decomposer.py `
      --data data/decomposition_trees/train.json `
      --checkpoint checkpoints/planck13/best.pt `
      --tokenizer data/wikipedia/tokenizer.model `
      --save-dir checkpoints/planck_decomposer `
      --epochs 50

The fine-tuning keeps the base model frozen and trains a small adapter
head (LoRA-style projection on tok_features) to shift outputs toward
JSON tree generation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Train Planck decomposer")
    p.add_argument("--data", required=True, help="Path to train.json")
    p.add_argument("--checkpoint", required=True, help="Planck base checkpoint")
    p.add_argument("--tokenizer", required=True, help="SentencePiece model")
    p.add_argument("--save-dir", default="checkpoints/planck_decomposer")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=10)
    return p.parse_args()


def format_sample(prompt: str, tree: dict) -> str:
    """Format a training sample as input+target string."""
    tree_json = json.dumps(tree, separators=(",", ":"))
    return f"DECOMPOSE: {prompt}\nTREE: {tree_json}"


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()
    print(f"Tokenizer: {vocab_size} tokens")

    # Load training data
    with open(args.data) as f:
        data = json.load(f)
    print(f"Training data: {len(data)} samples")

    # Format and tokenize
    samples = []
    for item in data:
        text = format_sample(item["prompt"], item["tree"])
        ids = sp.encode(text, out_type=int)
        if len(ids) <= args.max_len:
            samples.append(ids)
        else:
            samples.append(ids[:args.max_len])

    # Train/val split
    n_val = max(1, int(len(samples) * args.val_split))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  Train: {len(train_samples)}, Val: {n_val}")
    print(f"  Avg length: {sum(len(s) for s in samples) / len(samples):.0f} tokens")

    # Load model
    print(f"\nLoading Planck: {args.checkpoint}")
    from src.sgs_lm import SGSLanguageModel, migrate_state_dict

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = migrate_state_dict(state)

    model = SGSLanguageModel(vocab_size=vocab_size)
    model.load_state_dict(state)
    model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Fine-tune: unfreeze all parameters (small dataset, full fine-tune is fine)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * (len(train_samples) // args.batch_size + 1),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n=== Training ({args.epochs} epochs) ===")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle
        import random
        random.shuffle(train_samples)

        # Batch
        for batch_start in range(0, len(train_samples), args.batch_size):
            batch_ids = train_samples[batch_start:batch_start + args.batch_size]
            if not batch_ids:
                continue

            # Pad to same length
            max_len = min(max(len(s) for s in batch_ids), args.max_len)
            padded = []
            for ids in batch_ids:
                if len(ids) < max_len:
                    padded.append(ids + [0] * (max_len - len(ids)))
                else:
                    padded.append(ids[:max_len])

            input_ids = torch.tensor(padded, dtype=torch.long, device=device)

            # Causal LM: predict next token
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]

            # Mask padding
            mask = (targets != 0).float()
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="none",
            )
            loss = (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % args.log_interval == 0:
                print(f"  ep {epoch+1} step {n_batches} | loss {loss.item():.3f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ids in val_samples:
                ids_t = torch.tensor([ids[:args.max_len]], dtype=torch.long, device=device)
                logits = model(ids_t[:, :-1])
                targets = ids_t[:, 1:]
                mask = (targets != 0).float()
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
                )
                val_loss += (loss * mask.reshape(-1)).sum().item() / mask.sum().clamp(min=1).item()
        val_loss /= max(len(val_samples), 1)

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) | "
              f"train_loss={avg_loss:.3f} | val_loss={val_loss:.3f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "args": vars(args),
            }, save_dir / "best.pt")
            print(f"  ** new best (val_loss={val_loss:.3f}), saved best.pt")

    # Save final
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
        "val_loss": val_loss,
        "args": vars(args),
    }, save_dir / "final.pt")
    print(f"\nDone. Best val_loss={best_val_loss:.3f}")
    print(f"Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
