"""
Training script for Raum bridge (1.0 and 1.1).

Raum 1.0 (GloVe path):
    python scripts/train_raum_bridge.py ^
      --glove data/glove.6B.300d.txt ^
      --save-dir checkpoints/raum_10

Raum 1.1 (frozen Planck encoder):
    python scripts/train_raum_bridge.py ^
      --glove data/glove.6B.300d.txt ^
      --encoder-checkpoint checkpoints/planck13/best.pt --freeze-encoder ^
      --n-objects-max 3 ^
      --d-model 256 --n-layers 6 --n-heads 8 ^
      --save-dir checkpoints/raum_11_n3

Resume after a Ctrl-C:
    python scripts/train_raum_bridge.py ... --resume
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_glove
from src.gaussian import SemanticGaussianVocab
from src.raum.data import generate_comp_gen_split, RaumDataset, collate_raum
from src.raum.bridge import RaumBridge, compute_routing_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train Raum bridge (1.0 / 1.1)")
    p.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")

    # Raum 1.1: frozen encoder
    p.add_argument("--encoder-checkpoint", type=str, default=None,
                   help="Path to Planck checkpoint (enables 1.1 mode)")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze encoder weights (default for 1.1)")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="SentencePiece model for encoder mode (auto-detected from data dir)")

    # Raum 1.1: blob library
    p.add_argument("--blobs-dir", type=str, default=None,
                   help="Path to blob library dir (data/blobs/)")
    p.add_argument("--n-blobs-max", type=int, default=None,
                   help="Cap on blob library size for ablations")

    # Data
    p.add_argument("--n-objects-max", type=int, default=2,
                   help="Max objects per scene (2 for 1.0, 3 for 1.1)")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--K", type=int, default=32,
                   help="Retained for demo/analyser compat; unused by training.")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--save-dir", default="checkpoints/raum_10",
                   help="Checkpoint dir. Use a new dir per experiment.")
    p.add_argument("--seed", type=int, default=42)

    # Loss weights
    p.add_argument("--lambda-pos", type=float, default=1.0)
    p.add_argument("--lambda-dir", type=float, default=0.5)
    p.add_argument("--lambda-tpl", type=float, default=1.0)
    p.add_argument("--lambda-col", type=float, default=1.0)
    p.add_argument("--lambda-scl", type=float, default=0.5)
    p.add_argument("--lambda-rol", type=float, default=0.5)
    p.add_argument("--pair-margin", type=float, default=0.3)

    # Resume
    p.add_argument("--resume", action="store_true",
                   help="Resume from <save-dir>/last.pt if it exists.")
    p.add_argument("--ckpt-interval", type=int, default=0,
                   help="Save last.pt every N steps. 0 means eval + "
                        "end-of-epoch only.")
    return p.parse_args()


def _load_blob_library(blobs_dir: str, n_blobs_max: int | None) -> tuple[int, list[str]]:
    """Load blob index and return (n_blobs, blob_names)."""
    index_path = Path(blobs_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Blob index not found at {index_path}. "
            f"Run scripts/build_blobs_shapenet.py first."
        )
    with open(index_path) as f:
        index = json.load(f)

    blob_names = index if isinstance(index, list) else list(index.keys())
    if n_blobs_max is not None:
        blob_names = blob_names[:n_blobs_max]
    return len(blob_names), blob_names


@torch.no_grad()
def evaluate(model, get_features_fn, loader, device, args, max_batches=None) -> dict:
    """Run the validation loader and return averaged metrics."""
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        token_ids = batch["token_ids"].to(device)
        mask = batch["mask"].to(device)
        mu_s, features = get_features_fn(token_ids)
        out = model(mu_s, features, mask)
        _, metrics = compute_routing_loss(
            out, batch,
            lambda_pos=args.lambda_pos, lambda_dir=args.lambda_dir,
            lambda_tpl=args.lambda_tpl, lambda_col=args.lambda_col,
            lambda_scl=args.lambda_scl, lambda_rol=args.lambda_rol,
            pair_margin=args.pair_margin,
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    use_encoder = args.encoder_checkpoint is not None
    if use_encoder:
        print(f"\n=== Raum 1.1 mode (frozen Planck encoder) ===")
        print(f"  Encoder: {args.encoder_checkpoint}")
    else:
        print(f"\n=== Raum 1.0 mode (GloVe + SemanticGaussianVocab) ===")

    # ── GloVe (always loaded for 1.0; also for 1.1 as fallback word2idx) ──
    print("\n=== Loading GloVe ===")
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)

    # ── Feature extraction function ──
    if use_encoder:
        from src.raum.encoder import FrozenPlanckEncoder, build_sp_word2idx
        encoder = FrozenPlanckEncoder(args.encoder_checkpoint, device=device)
        encoder.to(device)
        encoder.eval()
        d_s = encoder.d_s
        d_f = encoder.d_f
        print(f"  Encoder dims: d_s={d_s}, d_f={d_f}, vocab={encoder.vocab_size}")

        # Build word2idx from SentencePiece so token_ids match the encoder
        tokenizer_path = args.tokenizer
        if tokenizer_path is None:
            # Auto-detect from checkpoint dir sibling data dir
            ckpt_dir = Path(args.encoder_checkpoint).parent
            candidates = [
                ckpt_dir.parent / "data" / "wikipedia" / "tokenizer.model",
                ckpt_dir / "tokenizer.model",
                Path("data/wikipedia/tokenizer.model"),
                Path("data/tinystories/tokenizer.model"),
            ]
            for c in candidates:
                if c.exists():
                    tokenizer_path = str(c)
                    break
        if tokenizer_path and Path(tokenizer_path).exists():
            print(f"  Tokenizer: {tokenizer_path}")
            word2idx = build_sp_word2idx(tokenizer_path)
            print(f"  SP word2idx: {len(word2idx)} scene words mapped")
        else:
            print(f"  WARN: No SP tokenizer found, using GloVe word2idx.")
            print(f"  Token IDs may not match encoder embeddings!")

        def get_features(token_ids):
            return encoder.encode(token_ids)
    else:
        d_f = vectors.shape[1]
        d_s = args.d_s
        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab.to(device)
        vocab.eval()

        def get_features(token_ids):
            mu_s, _, _, features = vocab.get_params(token_ids)
            return mu_s, features

    # ── Determine n_blobs + scene object vocabulary ──
    from src.raum.vocab import N_OBJECTS
    from src.raum.data import load_blob_object_names
    if args.blobs_dir:
        n_blobs, blob_names = _load_blob_library(args.blobs_dir, args.n_blobs_max)
        scene_objects = load_blob_object_names(args.blobs_dir)
        if args.n_blobs_max:
            scene_objects = {k: v for k, v in scene_objects.items() if v < args.n_blobs_max}
        print(f"  Blob library: {n_blobs} classes from {args.blobs_dir}")
    else:
        n_blobs = N_OBJECTS
        blob_names = None
        scene_objects = None  # falls back to OBJECTS (6 primitives)

    # ── Data ──
    print("\n=== Generating data ===")
    train_scenes, val_scenes, test_scenes = generate_comp_gen_split(
        args.n_train, args.n_val, args.n_test,
        n_objects_max=args.n_objects_max, objects=scene_objects, seed=args.seed,
    )
    max_objects = args.n_objects_max
    train_ds = RaumDataset(train_scenes, word2idx, max_objects=max_objects)
    val_ds = RaumDataset(val_scenes, word2idx, max_objects=max_objects)
    test_ds = RaumDataset(test_scenes, word2idx, max_objects=max_objects)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_raum, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raum,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raum,
    )

    # ── Model ──
    print("\n=== Creating model ===")
    model = RaumBridge(
        d_s=d_s, d_f=d_f,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_blobs=n_blobs,
        K=args.K,
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  d_s={d_s}, d_f={d_f}, d_model={args.d_model}, "
          f"n_layers={args.n_layers}, n_heads={args.n_heads}, n_blobs={n_blobs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / "last.pt"
    best_path = save_dir / "best.pt"
    best_score = float("-inf")
    global_step = 0
    start_epoch = 0

    if args.resume and last_path.exists():
        print(f"Resuming from {last_path}")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        best_score = ckpt.get("best_score", float("-inf"))
        print(f"  epoch {start_epoch}, step {global_step}, "
              f"best_score {best_score:.3f}")
    elif args.resume:
        print(f"--resume set but {last_path} not found; starting fresh")

    def save_last(epoch_done: int):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch_done,
            "global_step": global_step,
            "best_score": best_score,
            "args": vars(args),
        }, last_path)

    # Ctrl-C handler
    _state = {"count": 0, "epoch": start_epoch}

    def _sigint(signum, frame):
        _state["count"] += 1
        if _state["count"] == 1:
            print("\n[Ctrl-C] snapshotting to last.pt, press again to abort...")
            try:
                save_last(_state["epoch"])
                print(f"[Ctrl-C] saved {last_path}. Resume with --resume.")
            except Exception as e:
                print(f"[Ctrl-C] snapshot failed: {e}")
            raise KeyboardInterrupt
        else:
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint)

    # ── Train ──
    print(f"\n=== Training ({args.epochs} epochs) ===")
    for epoch in range(start_epoch, args.epochs):
        _state["epoch"] = epoch
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                mu_s, features = get_features(token_ids)

            out = model(mu_s, features, mask)
            loss, metrics = compute_routing_loss(
                out, batch,
                lambda_pos=args.lambda_pos, lambda_dir=args.lambda_dir,
                lambda_tpl=args.lambda_tpl, lambda_col=args.lambda_col,
                lambda_scl=args.lambda_scl, lambda_rol=args.lambda_rol,
                pair_margin=args.pair_margin,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.item())
            n_batches += 1

            if global_step % args.log_interval == 0:
                print(
                    f"  ep {epoch+1:>2d} step {global_step:>6d} | "
                    f"loss {metrics['loss']:.3f} "
                    f"pos {metrics['pos_mse']:.3f} "
                    f"dir {metrics['dir_acc']*100:5.1f}% "
                    f"tpl {metrics['tpl_acc']*100:5.1f}% "
                    f"col {metrics['col_mse']:.3f} "
                    f"scl {metrics['scl_mse']:.3f} "
                    f"rol {metrics['role_ce']:.3f}"
                )

            if global_step % args.eval_interval == 0:
                val = evaluate(model, get_features, val_loader, device, args, max_batches=None)
                score = val["tpl_acc"] + val["dir_acc"] - val["pos_mse"]
                print(
                    f"  >>> val | loss {val['loss']:.3f} "
                    f"pos {val['pos_mse']:.3f} "
                    f"dir {val['dir_acc']*100:5.1f}% "
                    f"tpl {val['tpl_acc']*100:5.1f}% | score {score:.3f}"
                )
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), best_path)
                    print(f"  ** new best, saved {best_path.name}")
                save_last(epoch)
                model.train()

            if args.ckpt_interval > 0 and global_step % args.ckpt_interval == 0:
                save_last(epoch)

        save_last(epoch + 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1} done ({elapsed:.1f}s) | avg_loss={epoch_loss / max(n_batches,1):.3f}")

    # Final comp-gen test report.
    print("\n=== Comp-gen test (held-out object pairs) ===")
    test = evaluate(model, get_features, test_loader, device, args)
    print(
        f"  test | loss {test['loss']:.3f} "
        f"pos {test['pos_mse']:.3f} "
        f"dir {test['dir_acc']*100:5.1f}% "
        f"tpl {test['tpl_acc']*100:5.1f}% "
        f"col {test['col_mse']:.3f}"
    )

    print(f"\nCheckpoint: {best_path}")
    print(
        f"Run analysis: python scripts/analyze_raum_bridge.py "
        f"--checkpoint {best_path} --glove {args.glove}"
    )


if __name__ == "__main__":
    main()
