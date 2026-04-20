"""
Build knowledge blobs from TinyStories for Planck 1.1.

Clusters stories by semantic similarity, creates Gaussian blob
representations for each cluster.

Usage:
    python scripts/build_blobs.py --data-dir data/tinystories --n-blobs 50000
    python scripts/build_blobs.py --data-dir data/tinystories --checkpoint checkpoints/planck/best.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Build knowledge blobs")
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--checkpoint", default="checkpoints/planck/best.pt",
                   help="Planck 1.0 checkpoint for encoding")
    p.add_argument("--n-blobs", type=int, default=50000)
    p.add_argument("--chunk-size", type=int, default=128,
                   help="Tokens per chunk for blob construction")
    p.add_argument("--output", default="data/blobs/tinystories",
                   help="Output directory for blob store")
    p.add_argument("--max-chunks", type=int, default=500000,
                   help="Max chunks to process (for memory)")
    p.add_argument("--device", default="auto")

    # Architecture (must match checkpoint)
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    return p.parse_args()


def encode_chunks(model, data_path, chunk_size, max_chunks, device):
    """Encode text chunks using the SGS model's token embeddings.

    Returns mu centroids and feature vectors for each chunk.
    """
    print("Loading token data...")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    n_tokens = len(data)
    n_chunks = min(n_tokens // chunk_size, max_chunks)
    print(f"  {n_tokens:,} tokens → {n_chunks:,} chunks of {chunk_size}")

    all_mu = []
    all_features = []
    batch_size = 256

    print("Encoding chunks...")
    model.eval()
    with torch.no_grad():
        for start in range(0, n_chunks, batch_size):
            end = min(start + batch_size, n_chunks)
            batch_chunks = []
            for i in range(start, end):
                offset = i * chunk_size
                chunk = data[offset:offset + chunk_size].astype(np.int64)
                batch_chunks.append(chunk)

            token_ids = torch.tensor(np.stack(batch_chunks), device=device)

            # Get Gaussian parameters from model
            mu = model.tok_mu(token_ids)                    # [B, chunk_size, d_s]
            pos = torch.arange(chunk_size, device=device)
            mu = mu + model.pos_mu(pos).unsqueeze(0)
            features = model.tok_features(token_ids)        # [B, chunk_size, d_f]

            # Chunk centroid = mean of token mu's
            chunk_mu = mu.mean(dim=1)                       # [B, d_s]
            chunk_feat = features.mean(dim=1)               # [B, d_f]

            all_mu.append(chunk_mu.cpu())
            all_features.append(chunk_feat.cpu())

            if (start // batch_size) % 100 == 0:
                print(f"  Encoded {end:,}/{n_chunks:,} chunks")

    all_mu = torch.cat(all_mu, dim=0)           # [n_chunks, d_s]
    all_features = torch.cat(all_features, dim=0)  # [n_chunks, d_f]
    print(f"  Encoded {all_mu.shape[0]:,} chunks")
    return all_mu, all_features


def cluster_chunks(mu, features, n_blobs):
    """Cluster chunk embeddings into n_blobs clusters using mini-batch k-means."""
    from sklearn.cluster import MiniBatchKMeans

    print(f"\nClustering {mu.shape[0]:,} chunks into {n_blobs:,} blobs...")
    mu_np = mu.numpy()

    kmeans = MiniBatchKMeans(
        n_clusters=n_blobs,
        batch_size=min(10000, mu.shape[0]),
        n_init=3,
        max_iter=100,
        verbose=1,
    )
    labels = kmeans.fit_predict(mu_np)

    print("Computing blob parameters...")
    blob_mu = torch.zeros(n_blobs, mu.shape[1])
    blob_log_var = torch.zeros(n_blobs, mu.shape[1])
    blob_raw_alpha = torch.zeros(n_blobs)
    blob_features = torch.zeros(n_blobs, features.shape[1])
    cluster_sizes = torch.zeros(n_blobs)

    for c in range(n_blobs):
        mask = labels == c
        count = mask.sum()
        cluster_sizes[c] = count

        if count == 0:
            continue

        cluster_mu = mu[mask]
        cluster_feat = features[mask]

        # Blob mu = cluster centroid
        blob_mu[c] = cluster_mu.mean(dim=0)

        # Blob log_var = log variance of cluster (semantic breadth)
        if count > 1:
            var = cluster_mu.var(dim=0).clamp(min=1e-6)
            blob_log_var[c] = var.log()
        else:
            blob_log_var[c] = 0.0  # single-element cluster: unit variance

        # Blob alpha = confidence based on cluster tightness
        # Tight cluster (low entropy) → high alpha
        if count > 1:
            intra_dist = ((cluster_mu - blob_mu[c].unsqueeze(0)) ** 2).sum(-1).mean()
            # Inverse sigmoid: tighter clusters get higher alpha
            blob_raw_alpha[c] = -intra_dist.log().clamp(min=-5, max=5)
        else:
            blob_raw_alpha[c] = 0.0

        # Blob features = mean of chunk features
        blob_features[c] = cluster_feat.mean(dim=0)

    # Report cluster quality
    non_empty = (cluster_sizes > 0).sum().item()
    avg_size = cluster_sizes[cluster_sizes > 0].mean().item()
    print(f"  Non-empty clusters: {non_empty:,}/{n_blobs:,}")
    print(f"  Average cluster size: {avg_size:.1f}")
    print(f"  Blob mu range: [{blob_mu.min():.3f}, {blob_mu.max():.3f}]")
    print(f"  Blob log_var range: [{blob_log_var.min():.3f}, {blob_log_var.max():.3f}]")

    return blob_mu, blob_log_var, blob_raw_alpha, blob_features, cluster_sizes


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    from src.sgs_lm import SGSLanguageModel, migrate_state_dict

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = migrate_state_dict(state)
    vocab_size = state["tok_mu.weight"].shape[0]

    model = SGSLanguageModel(
        vocab_size=vocab_size,
        d_s=args.d_s, d_f=args.d_f,
        n_passes=args.n_passes, n_heads=args.n_heads,
        max_len=args.context_len, ffn_mult=args.ffn_mult,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded {model.count_parameters()/1e6:.1f}M params")

    # Encode chunks
    train_bin = os.path.join(args.data_dir, "train.bin")
    chunk_mu, chunk_features = encode_chunks(
        model, train_bin, args.chunk_size, args.max_chunks, device
    )

    # Cluster into blobs
    blob_mu, blob_log_var, blob_raw_alpha, blob_features, sizes = cluster_chunks(
        chunk_mu, chunk_features, args.n_blobs
    )

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob_data = {
        "mu": blob_mu,
        "log_var": blob_log_var,
        "raw_alpha": blob_raw_alpha,
        "features": blob_features,
        "cluster_sizes": sizes,
    }
    out_path = out_dir / "blobs.pt"
    torch.save(blob_data, out_path)

    meta = {
        "n_blobs": args.n_blobs,
        "d_s": args.d_s,
        "d_f": args.d_f,
        "chunk_size": args.chunk_size,
        "n_chunks_processed": chunk_mu.shape[0],
        "non_empty_clusters": int((sizes > 0).sum().item()),
        "checkpoint": args.checkpoint,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {out_path} ({size_mb:.0f} MB)")
    print(f"  {args.n_blobs:,} blobs, d_s={args.d_s}, d_f={args.d_f}")
    print(f"  Params: {args.n_blobs * (args.d_s * 2 + 1 + args.d_f) / 1e6:.1f}M")


if __name__ == "__main__":
    main()
