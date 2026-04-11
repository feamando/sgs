"""
TinyStories data pipeline — download, BPE tokenizer, binary dataset.

Downloads from HuggingFace without the `datasets` library (Windows SSL compat).
Trains a 32K BPE tokenizer via sentencepiece.
Stores tokenized data as memory-mapped uint16 arrays for fast loading.
"""

import os
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

    print(f"Tokenizing {len(texts):,} texts → {output_file}")
    bos, eos = sp.bos_id(), sp.eos_id()

    # Write incrementally to avoid huge list in memory
    total = 0
    with open(output_file, "wb") as f:
        for i, text in enumerate(texts):
            ids = [bos] + sp.encode(text) + [eos]
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(ids)
            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,}/{len(texts):,} ({total:,} tokens)")

    print(f"  Done: {total:,} tokens ({total*2/(1024**2):.0f} MB)")
    return total


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

class TinyStoriesDataset(Dataset):
    """
    Memory-mapped token dataset for causal LM training.

    Returns random (context_length) windows from the tokenized corpus.
    Each sample: (x, y) where y = x shifted right by 1.
    """

    def __init__(self, bin_file: str, context_length: int = 512):
        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare TinyStories data")
    parser.add_argument("--data-dir", default="data/tinystories")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    prepare_data(args.data_dir, args.vocab_size)
