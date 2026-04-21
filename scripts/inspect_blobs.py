"""Quick health check for a built blob store.

Usage:
    python scripts/inspect_blobs.py
    python scripts/inspect_blobs.py --blob-dir data/blobs/tinystories
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blob-dir", default="data/blobs/tinystories")
    args = p.parse_args()

    blob_dir = Path(args.blob_dir)
    blobs_path = blob_dir / "blobs.pt"
    meta_path = blob_dir / "meta.json"

    if not blobs_path.exists():
        raise FileNotFoundError(f"Not found: {blobs_path}")

    d = torch.load(blobs_path, map_location="cpu", weights_only=False)
    n_blobs = d["mu"].shape[0]
    d_s = d["mu"].shape[1]
    d_f = d["features"].shape[1]

    print(f"Blob store: {blob_dir}")
    print(f"  Blobs: {n_blobs:,}")
    print(f"  d_s={d_s}  d_f={d_f}")
    print(f"  mu range:      [{d['mu'].min().item():.3f}, {d['mu'].max().item():.3f}]")
    print(f"  log_var range: [{d['log_var'].min().item():.3f}, {d['log_var'].max().item():.3f}]")
    print(f"  alpha range:   [{d['raw_alpha'].min().item():.3f}, {d['raw_alpha'].max().item():.3f}]")

    if "cluster_sizes" in d:
        sizes = d["cluster_sizes"]
        non_empty = int((sizes > 0).sum().item())
        avg = sizes[sizes > 0].mean().item() if non_empty else 0.0
        print(f"  Non-empty clusters: {non_empty:,}/{n_blobs:,}")
        print(f"  Avg cluster size:   {avg:.1f}")

    if meta_path.exists():
        m = json.load(open(meta_path))
        print(f"  Source checkpoint:  {m.get('checkpoint', '?')}")
        print(f"  Chunks processed:   {m.get('n_chunks_processed', '?'):,}")


if __name__ == "__main__":
    main()
