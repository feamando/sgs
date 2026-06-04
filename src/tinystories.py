"""
Data pipelines — download, BPE tokenizer, binary dataset.

Misnamed for legacy reasons: this file started as the TinyStories
loader, but now also contains FineWeb-Edu (Hertz) and Wikipedia
(Planck 1.3) pipelines. Rename is deferred; every new corpus lands
here until there's a reason to split.

Trains a 32K BPE tokenizer via sentencepiece.
Stores tokenized data as memory-mapped uint16 arrays for fast loading.
"""

import os
import sys
import json
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ────────────────────────────────────────────────────────────
# Download
# ────────────────────────────────────────────────────────────

DATASET_ID = "roneneldan/TinyStories"
HF_API = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main/data"
HF_RESOLVE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"


def _download_file(url: str, dest: str, desc: str = ""):
    """Download with progress bar, handling Windows SSL."""
    import urllib.request
    import ssl

    # Try normal SSL first, fall back to unverified for Windows compat
    contexts = [None]
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    contexts.append(ctx)

    for ssl_ctx in contexts:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sgs/1.0"})
            resp = urllib.request.urlopen(req, context=ssl_ctx)
            total = int(resp.headers.get("Content-Length", 0))

            with open(dest, "wb") as f:
                downloaded = 0
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(
                            f"\r  {desc} {downloaded/(1024*1024):.0f}/{total/(1024*1024):.0f} MB ({pct}%)",
                            end="", flush=True,
                        )
            print()
            return
        except (ssl.SSLError, urllib.error.URLError):
            if ssl_ctx is not None:
                raise
            continue


def download_tinystories(data_dir: str) -> tuple[list[str], list[str]]:
    """
    Download TinyStories parquet files and extract text.

    Returns:
        (train_texts, val_texts) — lists of story strings.
    """
    import urllib.request
    import ssl

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Get file listing from HF API
    print("Fetching file list from HuggingFace...")
    for ctx in [None, ssl.create_default_context()]:
        if ctx is not None:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        try:
            req = urllib.request.Request(HF_API, headers={"User-Agent": "sgs/1.0"})
            resp = urllib.request.urlopen(req, context=ctx)
            files = json.loads(resp.read())
            break
        except (ssl.SSLError, urllib.error.URLError):
            if ctx is not None:
                raise
            continue

    parquet_files = [
        f["path"] for f in files
        if isinstance(f, dict) and f.get("path", "").endswith(".parquet")
    ]

    # Download each parquet file
    for pf in parquet_files:
        fname = os.path.basename(pf)
        local = raw_dir / fname
        if local.exists():
            print(f"  Already have {fname}")
            continue
        url = f"{HF_RESOLVE}/{pf}"
        _download_file(url, str(local), desc=fname)

    # Read parquet → extract text
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas pyarrow")

    print("Reading parquet files...")
    train_texts, val_texts = [], []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".parquet"):
            continue
        df = pd.read_parquet(raw_dir / fname)
        texts = df["text"].dropna().tolist()
        if "train" in fname:
            train_texts.extend(texts)
        elif "validation" in fname:
            val_texts.extend(texts)

    print(f"  Train: {len(train_texts):,} stories")
    print(f"  Val:   {len(val_texts):,} stories")
    return train_texts, val_texts


# ────────────────────────────────────────────────────────────
# Tokenizer
# ────────────────────────────────────────────────────────────

def train_tokenizer(
    texts: list[str],
    model_prefix: str,
    vocab_size: int = 32000,
) -> "sentencepiece.SentencePieceProcessor":
    """Train BPE tokenizer with sentencepiece."""
    import sentencepiece as spm

    model_file = model_prefix + ".model"
    if os.path.exists(model_file):
        print(f"Tokenizer already exists: {model_file}")
        return spm.SentencePieceProcessor(model_file=model_file)

    # Write training text to temp file (sentencepiece reads files)
    text_file = model_prefix + "_train_corpus.txt"
    print(f"Writing training corpus ({len(texts):,} texts)...")
    with open(text_file, "w", encoding="utf-8") as f:
        for t in texts:
            line = t.strip().replace("\n", " ")
            if line:
                f.write(line + "\n")

    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        num_threads=os.cpu_count() or 4,
        shuffle_input_sentence=True,
        input_sentence_size=1_000_000,   # sample 1M sentences for training
        max_sentence_length=16384,
    )

    # Clean up temp file
    try:
        os.remove(text_file)
    except OSError:
        pass

    print(f"Tokenizer saved: {model_file}")
    return spm.SentencePieceProcessor(model_file=model_file)


# ────────────────────────────────────────────────────────────
# Tokenization → binary
# ────────────────────────────────────────────────────────────

def tokenize_to_binary(
    texts: list[str],
    sp: "sentencepiece.SentencePieceProcessor",
    output_file: str,
) -> int:
    """
    Tokenize all texts and save as a flat uint16 binary file.

    Each story is wrapped with BOS/EOS tokens and concatenated.
    Returns total number of tokens.
    """
    if os.path.exists(output_file):
        data = np.memmap(output_file, dtype=np.uint16, mode="r")
        print(f"  Binary already exists: {output_file} ({len(data):,} tokens)")
        return len(data)

    print(f"Tokenizing {len(texts):,} texts -> {output_file}")
    bos, eos = sp.bos_id(), sp.eos_id()

    # Batch tokenize for speed (sentencepiece supports batch encoding)
    total = 0
    batch_size = 10000
    with open(output_file, "wb") as f:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = sp.encode(batch)  # batch encode — much faster
            for ids in encoded:
                ids = [bos] + ids + [eos]
                arr = np.array(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                total += len(ids)
            if (start + batch_size) % 500_000 < batch_size:
                print(f"  {min(start + batch_size, len(texts)):,}/{len(texts):,} ({total:,} tokens)")

    print(f"  Done: {total:,} tokens ({total*2/(1024**2):.0f} MB)")
    return total


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    """
    Memory-mapped token dataset for causal LM training.

    Uses NON-OVERLAPPING chunks so 1 epoch = 1 pass over all tokens.
    Each sample: (x, y) where y = x shifted right by 1.

    With 450M tokens and context_length=512:
      n_chunks = 450M / 512 ≈ 879K
      1 epoch at batch_size=32 ≈ 27.5K steps
      3 epochs ≈ 82K steps (~2 hours on RTX 4090)
    """

    def __init__(self, bin_file: str, context_length: int = 512):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.context_length = context_length
        # Non-overlapping chunks
        self.n_chunks = len(self.data) // (context_length + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * (self.context_length + 1)
        chunk = self.data[start : start + self.context_length + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def get_dataloader(
    bin_file: str,
    context_length: int = 512,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a binary token file."""
    ds = TinyStoriesDataset(bin_file, context_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ────────────────────────────────────────────────────────────
# Full pipeline
# ────────────────────────────────────────────────────────────

def prepare_data(
    data_dir: str = "data/tinystories",
    vocab_size: int = 32000,
    context_length: int = 512,
) -> dict:
    """
    End-to-end: download → tokenize → binary.

    Returns dict with paths to all artifacts.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    train_texts, val_texts = download_tinystories(str(data_dir))

    # 2. Train tokenizer
    tok_prefix = str(data_dir / "tokenizer")
    sp = train_tokenizer(train_texts, tok_prefix, vocab_size)

    # 3. Tokenize to binary
    train_bin = str(data_dir / "train.bin")
    val_bin = str(data_dir / "val.bin")
    n_train = tokenize_to_binary(train_texts, sp, train_bin)
    n_val = tokenize_to_binary(val_texts, sp, val_bin)

    # Free text memory
    del train_texts, val_texts

    result = {
        "train_bin": train_bin,
        "val_bin": val_bin,
        "tokenizer": tok_prefix + ".model",
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "vocab_size": sp.get_piece_size(),
    }
    print(f"\nData ready:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


# ────────────────────────────────────────────────────────────
# FineWeb-Edu download (for Hertz 1B)
# ────────────────────────────────────────────────────────────

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_HF_RESOLVE = f"https://huggingface.co/datasets/{FINEWEB_DATASET}/resolve/main"


def download_fineweb_edu(data_dir: str, max_tokens: int = 10_000_000_000) -> tuple[list[str], list[str]]:
    """
    Download FineWeb-Edu 10BT sample for Hertz training.

    Uses pre-made sample/10BT/ from HuggingFace (~10B tokens in parquet shards).
    Downloads shards until we have ~max_tokens worth of text.
    """
    import urllib.request
    import ssl

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Use the pre-made 10BT sample
    sample_path = "sample/10BT"
    api_url = f"https://huggingface.co/api/datasets/{FINEWEB_DATASET}/tree/main/{sample_path}"

    print(f"Fetching FineWeb-Edu file list from {sample_path}...")
    for ctx in [None, ssl.create_default_context()]:
        if ctx is not None:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "sgs/1.0"})
            resp = urllib.request.urlopen(req, context=ctx)
            files = json.loads(resp.read())
            break
        except (ssl.SSLError, urllib.error.URLError):
            if ctx is not None:
                raise
            continue

    parquet_files = sorted([
        f["path"] for f in files
        if isinstance(f, dict) and f.get("path", "").endswith(".parquet")
    ])
    print(f"  Found {len(parquet_files)} parquet shards")

    # Download shards until we have enough text
    # Rough estimate: 4 chars per token, so max_tokens * 4 chars = target char count
    target_chars = max_tokens * 4
    total_chars = 0
    all_texts = []

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas pyarrow")

    for i, pf in enumerate(parquet_files):
        if total_chars >= target_chars:
            break

        fname = os.path.basename(pf)
        local = raw_dir / fname

        if not local.exists():
            url = f"{FINEWEB_HF_RESOLVE}/{pf}"
            print(f"  Downloading shard {i+1}: {fname}...")
            _download_file(url, str(local), desc=fname)
        else:
            print(f"  Already have {fname}")

        # Read and count
        df = pd.read_parquet(local)
        texts = df["text"].dropna().tolist()
        shard_chars = sum(len(t) for t in texts)
        total_chars += shard_chars
        all_texts.extend(texts)

        est_tokens = total_chars // 4
        print(f"    {len(texts):,} texts, ~{est_tokens/1e9:.1f}B tokens so far "
              f"({total_chars/1e9:.1f}B chars)")

    print(f"\n  Total: {len(all_texts):,} texts, ~{total_chars // 4 / 1e9:.1f}B estimated tokens")

    # Split 99/1 train/val
    split_idx = int(len(all_texts) * 0.99)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    print(f"  Train: {len(train_texts):,}, Val: {len(val_texts):,}")

    return train_texts, val_texts


def prepare_fineweb(
    data_dir: str = "data/fineweb",
    vocab_size: int = 32000,
    context_length: int = 1024,
    max_tokens: int = 10_000_000_000,
) -> dict:
    """
    End-to-end pipeline for FineWeb-Edu: download → tokenize → binary.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    train_texts, val_texts = download_fineweb_edu(str(data_dir), max_tokens)

    # 2. Train tokenizer
    tok_prefix = str(data_dir / "tokenizer")
    sp = train_tokenizer(train_texts, tok_prefix, vocab_size)

    # 3. Tokenize to binary
    train_bin = str(data_dir / "train.bin")
    val_bin = str(data_dir / "val.bin")
    n_train = tokenize_to_binary(train_texts, sp, train_bin)
    n_val = tokenize_to_binary(val_texts, sp, val_bin)

    del train_texts, val_texts

    result = {
        "train_bin": train_bin,
        "val_bin": val_bin,
        "tokenizer": tok_prefix + ".model",
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "vocab_size": sp.get_piece_size(),
    }
    print(f"\nData ready:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


# ────────────────────────────────────────────────────────────
# Wikipedia (for Planck 1.3) — consumes HuggingFace datasets cache
# ────────────────────────────────────────────────────────────

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_DEFAULT_REVISION = "20231101.en"


def load_wikipedia_texts(
    hf_cache_dir: str,
    revision: str = WIKIPEDIA_DEFAULT_REVISION,
    val_fraction: float = 0.005,
) -> tuple[list[str], list[str]]:
    """Load cleaned Wikipedia articles from a HuggingFace datasets cache.

    Returns (train_texts, val_texts). Uses `datasets.load_dataset` with
    a local cache so the Parquet files downloaded by SETUP §2.1 are
    reused verbatim.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required: pip install datasets")

    print(f"Loading {WIKIPEDIA_DATASET} ({revision}) from cache {hf_cache_dir}...")
    ds = load_dataset(
        WIKIPEDIA_DATASET,
        revision,
        cache_dir=hf_cache_dir,
        split="train",
    )
    texts = ds["text"]
    n = len(texts)
    split_idx = int(n * (1.0 - val_fraction))
    train_texts = list(texts[:split_idx])
    val_texts = list(texts[split_idx:])
    print(f"  {n:,} articles -> train {len(train_texts):,}, val {len(val_texts):,}")
    return train_texts, val_texts


def prepare_wikipedia(
    data_dir: str = "data/wikipedia",
    hf_cache_dir: str = "data/wikipedia/hf",
    revision: str = WIKIPEDIA_DEFAULT_REVISION,
    vocab_size: int = 32000,
) -> dict:
    """End-to-end Wikipedia prep: HF cache → tokenizer → train.bin/val.bin."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_texts, val_texts = load_wikipedia_texts(hf_cache_dir, revision)

    tok_prefix = str(data_dir / "tokenizer")
    sp = train_tokenizer(train_texts, tok_prefix, vocab_size)

    train_bin = str(data_dir / "train.bin")
    val_bin = str(data_dir / "val.bin")
    n_train = tokenize_to_binary(train_texts, sp, train_bin)
    n_val = tokenize_to_binary(val_texts, sp, val_bin)

    del train_texts, val_texts

    result = {
        "train_bin": train_bin,
        "val_bin": val_bin,
        "tokenizer": tok_prefix + ".model",
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "vocab_size": sp.get_piece_size(),
    }
    print(f"\nData ready:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


# ────────────────────────────────────────────────────────────
# FineWeb-Edu + Wikipedia mix (for Hertz 1.2)
# ────────────────────────────────────────────────────────────


def _truncate_to_char_budget(texts: list[str], char_budget: int) -> list[str]:
    """Keep texts in order until their cumulative length hits char_budget."""
    if char_budget <= 0:
        return []
    out = []
    total = 0
    for t in texts:
        out.append(t)
        total += len(t)
        if total >= char_budget:
            break
    return out


def prepare_fineweb_wiki_mix(
    data_dir: str = "data/hertz_mix",
    vocab_size: int = 32000,
    context_length: int = 512,
    max_tokens: int = 10_000_000_000,
    wiki_fraction: float = 0.3,
    hf_cache_dir: str = "data/wikipedia/hf",
    revision: str = WIKIPEDIA_DEFAULT_REVISION,
    seed: int = 1234,
) -> dict:
    """End-to-end FineWeb-Edu + Wikipedia mix.

    Mixes at the TEXT level (before tokenization) so a single shared tokenizer
    covers both corpora and the blob/retrieval distribution stays aligned with
    the base (the Planck 1.3 rationale). Token proportions are approximated by
    a 4-chars-per-token char budget per source. Reproducible: the manifest
    (sources, revision, fractions, budget, seed) is written to data_dir.

    `max_tokens` is the TOTAL budget across both sources; FineWeb gets
    (1 - wiki_fraction), Wikipedia gets wiki_fraction.
    """
    import random

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    fw_tokens = int(max_tokens * (1.0 - wiki_fraction))
    wiki_tokens = int(max_tokens * wiki_fraction)
    print(f"\nMix budget: {max_tokens/1e9:.1f}B total -> "
          f"FineWeb {fw_tokens/1e9:.2f}B + Wikipedia {wiki_tokens/1e9:.2f}B "
          f"(wiki_fraction={wiki_fraction})")

    # 1. FineWeb-Edu (download_fineweb_edu limits shards by token budget)
    fw_train, fw_val = download_fineweb_edu(str(data_dir / "fineweb_raw"), fw_tokens)

    # 2. Wikipedia (load full, then truncate to its char budget to cap memory)
    wk_train_all, wk_val_all = load_wikipedia_texts(hf_cache_dir, revision)
    wk_train = _truncate_to_char_budget(wk_train_all, wiki_tokens * 4)
    wk_val = _truncate_to_char_budget(wk_val_all, max(wiki_tokens * 4 // 100, 1))
    del wk_train_all, wk_val_all
    print(f"  Wikipedia after budget trim: train {len(wk_train):,}, val {len(wk_val):,}")

    # 3. Combine + deterministic shuffle (interleave the two distributions)
    rng = random.Random(seed)
    train_texts = fw_train + wk_train
    val_texts = fw_val + wk_val
    rng.shuffle(train_texts)
    rng.shuffle(val_texts)
    del fw_train, fw_val, wk_train, wk_val
    print(f"  Combined: train {len(train_texts):,}, val {len(val_texts):,}")

    # 4. ONE shared tokenizer over the combined corpus
    tok_prefix = str(data_dir / "tokenizer")
    sp = train_tokenizer(train_texts, tok_prefix, vocab_size)

    # 5. Tokenize each split to binary
    train_bin = str(data_dir / "train.bin")
    val_bin = str(data_dir / "val.bin")
    n_train = tokenize_to_binary(train_texts, sp, train_bin)
    n_val = tokenize_to_binary(val_texts, sp, val_bin)
    del train_texts, val_texts

    # 6. Reproducibility manifest
    manifest = {
        "sources": {
            "fineweb-edu": {"dataset": FINEWEB_DATASET, "target_tokens": fw_tokens},
            "wikipedia": {"dataset": WIKIPEDIA_DATASET, "revision": revision,
                          "target_tokens": wiki_tokens},
        },
        "wiki_fraction": wiki_fraction,
        "max_tokens": max_tokens,
        "vocab_size": sp.get_piece_size(),
        "context_length": context_length,
        "seed": seed,
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
    }
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    result = {
        "train_bin": train_bin,
        "val_bin": val_bin,
        "tokenizer": tok_prefix + ".model",
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "vocab_size": sp.get_piece_size(),
    }
    print(f"\nMix data ready (manifest: {data_dir / 'manifest.json'}):")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result


if __name__ == "__main__":
    import argparse

    # UTF-8 stdout/stderr so arrows/em-dashes in log strings don't crash the
    # Windows console (cp1252). Mirrors the guard in scripts/train_hertz.py.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dataset", default="tinystories",
                        choices=["tinystories", "fineweb-edu", "wikipedia", "fineweb-wiki-mix"],
                        help="Which dataset to download and prepare")
    parser.add_argument("--wiki-fraction", type=float, default=0.3,
                        help="Wikipedia share of the mix (fineweb-wiki-mix only)")
    parser.add_argument("--max-tokens", default="10B",
                        help="Max tokens for FineWeb-Edu (e.g., 1B, 5B, 10B)")
    parser.add_argument("--hf-cache-dir", default="data/wikipedia/hf",
                        help="HuggingFace datasets cache dir (Wikipedia only)")
    parser.add_argument("--revision", default=WIKIPEDIA_DEFAULT_REVISION,
                        help="HF dataset revision (Wikipedia only)")
    args = parser.parse_args()

    max_tok_str = args.max_tokens.upper().replace("B", "000000000").replace("M", "000000")
    max_tokens = int(max_tok_str)

    if args.dataset == "tinystories":
        prepare_data(args.data_dir, args.vocab_size)
    elif args.dataset == "fineweb-edu":
        if args.data_dir == "data/tinystories":
            args.data_dir = "data/fineweb"
        prepare_fineweb(args.data_dir, args.vocab_size, max_tokens=max_tokens)
    elif args.dataset == "wikipedia":
        if args.data_dir == "data/tinystories":
            args.data_dir = "data/wikipedia"
        prepare_wikipedia(
            args.data_dir,
            hf_cache_dir=args.hf_cache_dir,
            revision=args.revision,
            vocab_size=args.vocab_size,
        )
    elif args.dataset == "fineweb-wiki-mix":
        if args.data_dir == "data/tinystories":
            args.data_dir = "data/hertz_mix"
        prepare_fineweb_wiki_mix(
            args.data_dir,
            vocab_size=args.vocab_size,
            max_tokens=max_tokens,
            wiki_fraction=args.wiki_fraction,
            hf_cache_dir=args.hf_cache_dir,
            revision=args.revision,
        )
