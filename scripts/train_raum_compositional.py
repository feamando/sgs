"""
Training script for Raum PoC-D: Compositional Scene Graph.

Usage:
    python scripts/train_raum_compositional.py --glove data/glove.6B.300d.txt
    python scripts/train_raum_compositional.py --glove data/glove.6B.300d.txt --sgs-checkpoint checkpoints/sgs_stsb.pt

Runs on CPU. No GPU needed.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_glove
from src.raum.vocab import ALL_SCENE_WORDS
from src.raum.data import (
    generate_dataset, generate_comp_gen_split,
    RaumDataset, collate_raum,
)
from src.raum.compositional import RaumCompositional, compute_loss
from src.raum.eval import compute_metrics, aggregate_metrics, print_metrics
from src.raum.assemble import assemble_scene
from src.raum.templates import build_template_library


def parse_args():
    p = argparse.ArgumentParser(description="Train Raum PoC-D")
    p.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")
    p.add_argument("--sgs-checkpoint", default=None, help="Pre-trained SGS encoder (optional)")
    p.add_argument("--n-train", type=int, default=15000)
    p.add_argument("--n-val", type=int, default=2500)
    p.add_argument("--n-test", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-dir", default="checkpoints/raum_d")
    p.add_argument("--feature-mode", default="glove", choices=["glove", "sgs"],
                   help="Feature source: raw GloVe or SGS-refined")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_features(token_ids, mask, glove_matrix, sgs_encoder=None, feature_mode="glove"):
    """Extract per-word features. Currently only raw GloVe is wired up."""
    if feature_mode == "sgs":
        raise NotImplementedError(
            "feature-mode=sgs is not implemented yet. The SGS encoder forward path "
            "needs to be wired into Raum PoC-D and the encoder checkpoint loaded "
            "from --sgs-checkpoint. Use feature-mode=glove for now."
        )
    return glove_matrix[token_ids]  # [B, L, 300]


@torch.no_grad()
def evaluate(model, loader, glove_matrix, feature_mode="glove", sgs_encoder=None):
    """Run evaluation, return aggregated metrics."""
    model.eval()
    all_metrics = []
    for batch in loader:
        features = get_features(
            batch["token_ids"], batch["mask"],
            glove_matrix, sgs_encoder, feature_mode,
        )
        preds = model(features, batch["mask"])
        metrics = compute_metrics(preds, batch)
        all_metrics.append(metrics)
    model.train()
    return aggregate_metrics(all_metrics)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # PoC-D runs on CPU
    print(f"Device: {device}")

    # ── Load GloVe ──
    print("\n=== Loading GloVe ===")
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    glove_matrix = torch.from_numpy(vectors).float()
    d_f = glove_matrix.shape[1]  # 300

    # Check scene vocabulary coverage
    missing = [w for w in ALL_SCENE_WORDS if w not in word2idx]
    if missing:
        print(f"  WARNING: missing from GloVe: {missing}")
    else:
        print(f"  All {len(ALL_SCENE_WORDS)} scene words found in GloVe")

    # ── Generate data ──
    print("\n=== Generating synthetic data ===")
    train_scenes, val_scenes, test_scenes = generate_comp_gen_split(
        args.n_train, args.n_val, args.n_test, seed=args.seed,
    )

    train_ds = RaumDataset(train_scenes, word2idx)
    val_ds = RaumDataset(val_scenes, word2idx)
    test_ds = RaumDataset(test_scenes, word2idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_raum, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_raum)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_raum)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test (comp-gen): {len(test_ds)}")

    # ── Model ──
    print("\n=== Creating model ===")
    model = RaumCompositional(d_f=d_f)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )

    # ── Train ──
    print(f"\n=== Training ({args.epochs} epochs) ===")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            features = get_features(
                batch["token_ids"], batch["mask"],
                glove_matrix, feature_mode=args.feature_mode,
            )
            preds = model(features, batch["mask"])
            loss, loss_metrics = compute_loss(preds, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            global_step = epoch * len(train_loader) + step + 1
            if global_step % args.log_interval == 0:
                print(f"  epoch {epoch+1} step {global_step:>5d} | "
                      f"loss {loss.item():.4f} "
                      f"role={loss_metrics['role_loss']:.3f} "
                      f"obj={loss_metrics['obj_loss']:.3f} "
                      f"col={loss_metrics['color_loss']:.4f} "
                      f"rel={loss_metrics['rel_loss']:.4f}")

        # ── Epoch eval ──
        elapsed = time.time() - t0
        avg_loss = epoch_loss / n_batches
        val_metrics = evaluate(model, val_loader, glove_matrix, args.feature_mode)
        print(f"  Epoch {epoch+1} ({elapsed:.1f}s) | train_loss={avg_loss:.4f}")
        print_metrics(val_metrics, prefix="val: ")

        if val_metrics["obj_acc"] > best_val_acc:
            best_val_acc = val_metrics["obj_acc"]
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"  ** New best obj_acc={best_val_acc:.1%} → saved")

    # ── Final test (compositional generalization) ──
    print("\n=== Compositional Generalization Test ===")
    model.load_state_dict(torch.load(save_dir / "best.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, glove_matrix, args.feature_mode)
    print_metrics(test_metrics, prefix="comp-gen test: ")

    # Also eval on val for comparison
    val_metrics = evaluate(model, val_loader, glove_matrix, args.feature_mode)
    print_metrics(val_metrics, prefix="val (seen pairs): ")

    # ── Sample scene assembly ──
    print("\n=== Sample assemblies ===")
    templates = build_template_library()
    model.eval()
    for i in range(min(5, len(test_scenes))):
        scene_gt = test_scenes[i]
        batch = collate_raum([tokenize_scene_for_display(scene_gt, word2idx)])
        features = get_features(batch["token_ids"], batch["mask"], glove_matrix)
        preds = model(features, batch["mask"])
        assembled = assemble_scene(preds, templates)

        print(f"\n  Input: \"{scene_gt.sentence}\"")
        print(f"  GT objects: {[(o.obj_type, o.color[:1], o.position) for o in scene_gt.objects]}")
        print(f"  Predicted:  {[(o['type_name'], o['color'][:1].tolist(), o['position'].tolist()) for o in assembled]}")

    print(f"\nDone. Best checkpoint: {save_dir / 'best.pt'}")


def tokenize_scene_for_display(scene, word2idx):
    """Wrapper for tokenize_scene that works at top level."""
    from src.raum.data import tokenize_scene
    return tokenize_scene(scene, word2idx)


if __name__ == "__main__":
    main()
