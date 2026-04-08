"""
Data loading: GloVe embeddings + STS-B dataset.
"""

import os
import csv
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════
# STS-B direct download (avoids HuggingFace cache issues on Windows)
# ═══════════════════════════════════════════════════════════

STSB_URL = "http://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz"


def download_stsb(data_dir: str = "data") -> dict[str, str]:
    """Download STS-B from the official source (tar.gz)."""
    import tarfile

    stsb_dir = os.path.join(data_dir, "stsb")
    os.makedirs(stsb_dir, exist_ok=True)

    paths = {
        "train": os.path.join(stsb_dir, "sts-train.csv"),
        "dev": os.path.join(stsb_dir, "sts-dev.csv"),
        "test": os.path.join(stsb_dir, "sts-test.csv"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        return paths

    tar_path = os.path.join(stsb_dir, "stsbenchmark.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading STS-B from {STSB_URL}...")
        urllib.request.urlretrieve(STSB_URL, tar_path)

    print("Extracting STS-B...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".csv"):
                member.name = os.path.basename(member.name)
                tar.extract(member, stsb_dir)

    return paths


def parse_stsb_file(path: str) -> list[tuple[str, str, float]]:
    """
    Parse STS-B CSV file.

    Format: genre\tfilename\tyear\told_index\tsource1\tsource2\tsentence1\tsentence2\tscore
    (tab-separated, 9 columns, score is column index 4, sentences are 5 and 6)
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                try:
                    score = float(parts[4])
                    sent1 = parts[5]
                    sent2 = parts[6]
                    pairs.append((sent1, sent2, score))
                except (ValueError, IndexError):
                    continue
    return pairs


# ═══════════════════════════════════════════════════════════
# GloVe loading
# ═══════════════════════════════════════════════════════════

def load_glove(path: str, vocab_size: int = 50000) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """
    Load GloVe embeddings.

    Args:
        path: path to glove.6B.300d.txt
        vocab_size: max number of words to load

    Returns:
        word2idx: {word: index} mapping (0 = padding)
        vectors: [vocab_size, 300] numpy array
        freqs: [vocab_size] approximate word frequencies (rank-based)
        words: list of words
    """
    words = ["<PAD>"]
    vectors = [np.zeros(300)]

    print(f"Loading GloVe from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= vocab_size - 1:
                break
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if vec.shape[0] != 300:
                continue
            words.append(word)
            vectors.append(vec)

    vectors = np.stack(vectors)
    word2idx = {w: i for i, w in enumerate(words)}

    # Approximate frequencies from rank (Zipf's law: freq ∝ 1/rank)
    ranks = np.arange(1, len(words) + 1, dtype=np.float32)
    freqs = 1.0 / ranks
    freqs = freqs / freqs.sum()  # Normalize to probabilities

    print(f"Loaded {len(words)} words, embedding dim={vectors.shape[1]}")
    return word2idx, vectors, freqs, words


# ═══════════════════════════════════════════════════════════
# Tokenizer and Dataset
# ═══════════════════════════════════════════════════════════

def tokenize(text: str, word2idx: dict, max_len: int = 50) -> list[int]:
    """Simple whitespace tokenizer with GloVe vocabulary."""
    tokens = text.lower().strip().split()
    ids = []
    for t in tokens[:max_len]:
        # Strip punctuation
        t_clean = ''.join(c for c in t if c.isalnum())
        if t_clean in word2idx:
            ids.append(word2idx[t_clean])
    return ids if ids else [0]  # At least padding token


class STSBDataset(Dataset):
    """STS-B dataset for semantic textual similarity."""

    def __init__(self, split: str, word2idx: dict, max_len: int = 50, data_dir: str = "data"):
        self.word2idx = word2idx
        self.max_len = max_len
        self.pairs = []

        # Download if needed
        paths = download_stsb(data_dir)

        # Map split names
        split_map = {"train": "train", "validation": "dev", "val": "dev", "test": "test"}
        actual_split = split_map.get(split, split)

        raw_pairs = parse_stsb_file(paths[actual_split])

        for sent1, sent2, score in raw_pairs:
            ids_a = tokenize(sent1, word2idx, max_len)
            ids_b = tokenize(sent2, word2idx, max_len)
            self.pairs.append((ids_a, ids_b, score))

        print(f"STS-B {split}: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ids_a, ids_b, score = self.pairs[idx]
        return ids_a, ids_b, score


def collate_fn(batch):
    """Pad sequences and create masks."""
    ids_a_list, ids_b_list, scores = zip(*batch)

    def pad(sequences):
        max_len = max(len(s) for s in sequences)
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        mask = torch.zeros(len(sequences), max_len, dtype=torch.float)
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1.0
        return padded, mask

    ids_a, mask_a = pad(ids_a_list)
    ids_b, mask_b = pad(ids_b_list)
    scores = torch.tensor(scores, dtype=torch.float)

    return ids_a, mask_a, ids_b, mask_b, scores


def get_dataloaders(
    word2idx: dict,
    batch_size: int = 64,
    max_len: int = 50,
    data_dir: str = "data",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/dev/test dataloaders for STS-B."""
    train_ds = STSBDataset("train", word2idx, max_len, data_dir)
    val_ds = STSBDataset("val", word2idx, max_len, data_dir)
    test_ds = STSBDataset("test", word2idx, max_len, data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
