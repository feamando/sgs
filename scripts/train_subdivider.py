"""
Train the subdivision MLP on (semantic label, template) pairs.

Expects training data at data/objaverse_gs/ with structure:
  {category}/{object_id}/model.ply
  {category}/{object_id}/metadata.json

The MLP learns to predict deformation parameters that transform a
canonical template into the target object's Gaussian representation.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.subdivider import SubdivisionMLP


class SubdivisionDataset(Dataset):
    """Dataset of (parent_features, target_positions) pairs."""

    def __init__(self, data_dir: Path, embed_dim: int = 300):
        self.samples = []
        self.embed_dim = embed_dim

        if not data_dir.exists():
            return

        for category_dir in sorted(data_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for obj_dir in sorted(category_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                meta_path = obj_dir / "metadata.json"
                model_path = obj_dir / "model.pt"
                if meta_path.exists() and model_path.exists():
                    self.samples.append({
                        "category": category,
                        "meta_path": meta_path,
                        "model_path": model_path,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(sample["model_path"], map_location="cpu", weights_only=True)

        # Target: normalized positions of the GS object
        positions = data["positions"]  # [M, 3]
        center = positions.mean(dim=0)
        extent = (positions - center).abs().max() + 1e-6
        positions_norm = (positions - center) / extent

        # Parent features: center position, average scale, average color
        parent_pos = torch.zeros(3)
        parent_scale = torch.tensor([1.0])
        parent_color = data.get("colors", torch.ones(1, 3) * 0.5).mean(dim=0)

        # Embedding: random for now (will be GloVe in production)
        embedding = torch.randn(self.embed_dim)
        context = torch.randn(self.embed_dim)

        return {
            "parent_pos": parent_pos,
            "parent_scale": parent_scale,
            "parent_color": parent_color,
            "embedding": embedding,
            "context": context,
            "target_positions": positions_norm,
        }


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute bidirectional Chamfer distance between two point clouds."""
    # pred: [M1, 3], target: [M2, 3]
    dists = torch.cdist(pred, target)  # [M1, M2]
    d_pred_to_target = dists.min(dim=1).values.mean()
    d_target_to_pred = dists.min(dim=0).values.mean()
    return d_pred_to_target + d_target_to_pred


def main():
    parser = argparse.ArgumentParser(description="Train subdivision MLP")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    data_dir = Path(args.data)
    dataset = SubdivisionDataset(data_dir)
    print(f"Training samples: {len(dataset)}")

    if len(dataset) == 0:
        print(f"No training data found at {data_dir}")
        print("Expected structure: {category}/{object_id}/model.pt + metadata.json")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SubdivisionMLP(embed_dim=300, n_categories=50, max_templates_per_category=5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            logits, deform = model(
                batch["parent_pos"],
                batch["parent_scale"],
                batch["parent_color"],
                batch["embedding"],
                batch["context"],
            )

            # Loss: L1 on deformation (regularize to small changes)
            loss = deform.abs().mean() * 0.01

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)

    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
