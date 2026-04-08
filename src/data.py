"""
Data loading: GloVe embeddings + STS-B + AllNLI datasets.
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


# ═══════════════════════════════════════════════════════════
# AllNLI (SNLI + MultiNLI) for contrastive training
# ═══════════════════════════════════════════════════════════

ALLNLI_URL = "https://sbert.net/datasets/AllNLI.tsv.gz"


def download_allnli(data_dir: str = "data") -> str:
    """Download AllNLI dataset."""
    import gzip

    nli_dir = os.path.join(data_dir, "allnli")
    os.makedirs(nli_dir, exist_ok=True)
    tsv_path = os.path.join(nli_dir, "AllNLI.tsv")

    if os.path.exists(tsv_path):
        return tsv_path

    gz_path = os.path.join(nli_dir, "AllNLI.tsv.gz")
    if not os.path.exists(gz_path):
        print(f"Downloading AllNLI from {ALLNLI_URL}...")
        urllib.request.urlretrieve(ALLNLI_URL, gz_path)

    print("Extracting AllNLI...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(tsv_path, 'wb') as f_out:
            f_out.write(f_in.read())

    return tsv_path


class NLITripletDataset(Dataset):
    """
    AllNLI dataset yielding (anchor, positive, negative) triplets.

    For each anchor sentence:
    - positive = an entailment pair
    - negative = a contradiction pair
    """

    def __init__(self, word2idx: dict, max_len: int = 50, data_dir: str = "data",
                 split: str = "train", max_samples: int = 0):
        self.word2idx = word2idx
        self.max_len = max_len
        self.triplets = []

        tsv_path = download_allnli(data_dir)

        # Parse: columns are split, genre, filename, year, old_idx, source1, source2,
        #        sentence1, sentence2, score, label (entailment/neutral/contradiction)
        # But AllNLI.tsv from sbert has: split\tsentence1\tsentence2\tlabel
        entail_pairs = {}  # anchor -> list of entailment sentences
        contra_pairs = {}  # anchor -> list of contradiction sentences

        print(f"Loading AllNLI ({split})...")
        with open(tsv_path, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                row_split, sent1, sent2, label = parts[0], parts[1], parts[2], parts[3]
                if row_split != split:
                    continue

                if label == "entailment":
                    entail_pairs.setdefault(sent1, []).append(sent2)
                elif label == "contradiction":
                    contra_pairs.setdefault(sent1, []).append(sent2)

        # Build triplets: anchor + entailment_pos + contradiction_neg
        for anchor, positives in entail_pairs.items():
            if anchor not in contra_pairs:
                continue
            negatives = contra_pairs[anchor]
            anchor_ids = tokenize(anchor, word2idx, max_len)
            for pos in positives:
                pos_ids = tokenize(pos, word2idx, max_len)
                neg = negatives[hash(pos) % len(negatives)]  # Deterministic neg selection
                neg_ids = tokenize(neg, word2idx, max_len)
                self.triplets.append((anchor_ids, pos_ids, neg_ids))

                if max_samples > 0 and len(self.triplets) >= max_samples:
                    break
            if max_samples > 0 and len(self.triplets) >= max_samples:
                break

        print(f"AllNLI {split}: {len(self.triplets)} triplets")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def nli_collate_fn(batch):
    """Collate NLI triplets into padded tensors."""
    anchors, positives, negatives = zip(*batch)

    def pad(sequences):
        max_len = max(len(s) for s in sequences)
        if max_len == 0:
            max_len = 1
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        mask = torch.zeros(len(sequences), max_len, dtype=torch.float)
        for i, s in enumerate(sequences):
            if len(s) > 0:
                padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, :len(s)] = 1.0
        return padded, mask

    a_ids, a_mask = pad(anchors)
    p_ids, p_mask = pad(positives)
    n_ids, n_mask = pad(negatives)

    return a_ids, a_mask, p_ids, p_mask, n_ids, n_mask


def get_nli_dataloader(
    word2idx: dict,
    batch_size: int = 64,
    max_len: int = 50,
    data_dir: str = "data",
    max_samples: int = 0,
) -> DataLoader:
    """Create AllNLI training dataloader."""
    ds = NLITripletDataset(word2idx, max_len, data_dir, split="train",
                           max_samples=max_samples)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=nli_collate_fn, num_workers=0, pin_memory=True,
    )
