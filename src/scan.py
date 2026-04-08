"""
SCAN dataset for compositional generalization.

Lake & Baroni (2018): "Generalization without Systematicity"
Tests: train on primitives, generalize to novel compositions.
"""

import os
import urllib.request
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader


SCAN_URL = "https://github.com/brendenlake/SCAN/archive/refs/heads/master.zip"


def download_scan(data_dir: str = "data") -> str:
    """Download SCAN dataset."""
    scan_dir = os.path.join(data_dir, "scan")
    os.makedirs(scan_dir, exist_ok=True)

    # Check if already extracted
    check_file = os.path.join(scan_dir, "tasks_train_addprim_jump.txt")
    if os.path.exists(check_file):
        return scan_dir

    zip_path = os.path.join(scan_dir, "scan.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading SCAN dataset...")
        try:
            req = urllib.request.Request(
                SCAN_URL,
                headers={"User-Agent": "Mozilla/5.0 (SGS-Experiment)"},
            )
            with urllib.request.urlopen(req) as response:
                with open(zip_path, 'wb') as f:
                    f.write(response.read())
        except Exception:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(
                SCAN_URL,
                headers={"User-Agent": "Mozilla/5.0 (SGS-Experiment)"},
            )
            with urllib.request.urlopen(req, context=ctx) as response:
                with open(zip_path, 'wb') as f:
                    f.write(response.read())

    print("Extracting SCAN...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if member.endswith('.txt') and '/tasks_' in member:
                filename = os.path.basename(member)
                with zf.open(member) as src, open(os.path.join(scan_dir, filename), 'wb') as dst:
                    dst.write(src.read())

    return scan_dir


def parse_scan_file(path: str) -> list[tuple[str, str]]:
    """Parse SCAN file. Format: 'IN: command OUT: action_sequence'"""
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('IN:'):
                continue
            parts = line.split('OUT:')
            if len(parts) == 2:
                inp = parts[0].replace('IN:', '').strip()
                out = parts[1].strip()
                pairs.append((inp, out))
    return pairs


def build_scan_vocab(pairs: list[tuple[str, str]]) -> tuple[dict, dict, dict, dict]:
    """Build input and output vocabularies for SCAN."""
    in_words = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    out_words = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}

    for inp, out in pairs:
        for w in inp.split():
            if w not in in_words:
                in_words[w] = len(in_words)
        for w in out.split():
            if w not in out_words:
                out_words[w] = len(out_words)

    in_idx2word = {v: k for k, v in in_words.items()}
    out_idx2word = {v: k for k, v in out_words.items()}
    return in_words, out_words, in_idx2word, out_idx2word


class SCANDataset(Dataset):
    """SCAN dataset for seq2seq."""

    def __init__(self, pairs, in_vocab, out_vocab, max_len=50):
        self.data = []
        for inp, out in pairs:
            in_ids = [in_vocab.get(w, 0) for w in inp.split()][:max_len]
            out_ids = [out_vocab["<BOS>"]] + [out_vocab.get(w, 0) for w in out.split()][:max_len] + [out_vocab["<EOS>"]]
            self.data.append((in_ids, out_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def scan_collate_fn(batch):
    """Pad SCAN sequences."""
    in_seqs, out_seqs = zip(*batch)

    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.float)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1.0
        return padded, mask

    in_ids, in_mask = pad(in_seqs)
    out_ids, out_mask = pad(out_seqs)
    return in_ids, in_mask, out_ids, out_mask


def get_scan_dataloaders(
    split: str = "addprim_jump",
    batch_size: int = 64,
    data_dir: str = "data",
) -> tuple:
    """
    Get SCAN dataloaders for a given split.

    Splits: addprim_jump, addprim_turn_left, length, simple
    """
    scan_dir = download_scan(data_dir)

    train_path = os.path.join(scan_dir, f"tasks_train_{split}.txt")
    test_path = os.path.join(scan_dir, f"tasks_test_{split}.txt")

    if not os.path.exists(train_path):
        # Try alternative naming
        available = [f for f in os.listdir(scan_dir) if f.endswith('.txt')]
        print(f"Available SCAN files: {available}")
        raise FileNotFoundError(f"SCAN split '{split}' not found at {train_path}")

    train_pairs = parse_scan_file(train_path)
    test_pairs = parse_scan_file(test_path)
    all_pairs = train_pairs + test_pairs

    print(f"SCAN {split}: {len(train_pairs)} train, {len(test_pairs)} test")

    in_vocab, out_vocab, in_idx2word, out_idx2word = build_scan_vocab(all_pairs)
    print(f"  Input vocab: {len(in_vocab)}, Output vocab: {len(out_vocab)}")

    train_ds = SCANDataset(train_pairs, in_vocab, out_vocab)
    test_ds = SCANDataset(test_pairs, in_vocab, out_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=scan_collate_fn, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=scan_collate_fn, num_workers=0, pin_memory=True)

    return train_loader, test_loader, in_vocab, out_vocab, out_idx2word
