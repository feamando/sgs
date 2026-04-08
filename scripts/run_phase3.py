#!/usr/bin/env python3
"""
Phase 3: Three experiments to resolve open questions.

  3C: Close gaps — d_s=300+NLI, more NLI epochs
  3B: NLI 3-way classification (not contrastive)
  3A: SCAN compositional generalization

Usage:
    python scripts/run_phase3.py --glove data/glove.6B.300d.txt --exp 3c
    python scripts/run_phase3.py --glove data/glove.6B.300d.txt --exp 3b
    python scripts/run_phase3.py --glove data/glove.6B.300d.txt --exp 3a
    python scripts/run_phase3.py --glove data/glove.6B.300d.txt --exp all
"""

import argparse
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.data import (
    load_glove, get_dataloaders, get_nli_dataloader, download_allnli, tokenize,
)
from src.gaussian import SemanticGaussianVocab
from src.model import SGSEncoder, FairSoftmaxModel


# ═══════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════

def evaluate_stsb(encoder, loader, device):
    encoder.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for ids_a, mask_a, ids_b, mask_b, scores in loader:
            ids_a, mask_a = ids_a.to(device), mask_a.to(device)
            ids_b, mask_b = ids_b.to(device), mask_b.to(device)
            a = encoder(ids_a, mask_a)
            b = encoder(ids_b, mask_b)
            cos = F.cosine_similarity(a, b, dim=-1) * 5.0
            preds_list.append(cos.cpu())
            labels_list.append(scores)
    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    sp, _ = spearmanr(preds, labels)
    return sp


class MeanPoolEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
    def forward(self, token_ids, mask=None):
        _, _, _, features = self.vocab.get_params(token_ids)
        if mask is not None:
            features = features * mask.float().unsqueeze(-1)
            lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            return features.sum(dim=1) / lengths
        return features.mean(dim=1)


class FairSoftmaxEncoder(nn.Module):
    def __init__(self, vocab, d_s=64, d_f=300):
        super().__init__()
        from src.model import FairSoftmaxModel
        self.inner = FairSoftmaxModel(vocab, d_s=d_s, d_f=d_f, n_layers=2)
    def forward(self, token_ids, mask=None):
        return self.inner._encode(token_ids, mask)


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    def forward(self, anchor, positive):
        a = F.normalize(anchor, p=2, dim=1)
        p = F.normalize(positive, p=2, dim=1)
        sim = a @ p.T / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        return F.cross_entropy(sim, labels)


def train_nli_contrastive(encoder, nli_loader, stsb_val, stsb_test, device, epochs, lr):
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = MultipleNegativesRankingLoss(0.05)
    best_val, best_state = -1, None

    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss, n = 0, 0
        for a_ids, a_mask, p_ids, p_mask, _, _ in nli_loader:
            a_ids, a_mask = a_ids.to(device), a_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)
            loss = loss_fn(encoder(a_ids, a_mask), encoder(p_ids, p_mask))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        val_sp = evaluate_stsb(encoder, stsb_val, device)
        if val_sp > best_val:
            best_val = val_sp
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        extra = f" | τ={encoder.tau.item():.1f}" if hasattr(encoder, 'tau') else ""
        print(f"    Epoch {epoch:2d}/{epochs} | loss={total_loss/n:.4f} | stsb_val={val_sp:.4f} | best={best_val:.4f}{extra}")

    if best_state:
        encoder.load_state_dict(best_state)
    test_sp = evaluate_stsb(encoder, stsb_test, device)
    return test_sp, best_val


# ═══════════════════════════════════════════════════════════
# Experiment 3C: Close the Gaps
# ═══════════════════════════════════════════════════════════

def run_exp_3c(args, word2idx, vectors, freqs, words, device):
    print(f"\n{'='*70}")
    print("EXPERIMENT 3C: Close the Gaps (d_s=300+NLI, more epochs)")
    print(f"{'='*70}")

    _, stsb_val, stsb_test = get_dataloaders(word2idx, batch_size=64, max_len=50)
    nli_loader = get_nli_dataloader(word2idx, batch_size=128, max_len=50)

    runs = [
        ("SGS d_s=64, NLI 10ep",   64,  10, "sgs"),
        ("SGS d_s=300, NLI 3ep",   300, 3,  "sgs"),
        ("SGS d_s=300, NLI 10ep",  300, 10, "sgs"),
        ("FairSfm d_s=64, NLI 10ep", 64, 10, "softmax"),
        ("FairSfm d_s=300, NLI 10ep", 300, 10, "softmax"),
    ]

    results = []
    for name, d_s, epochs, model_type in runs:
        print(f"\n  --- {name} ---")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        d_f = 300
        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab = vocab.to(device)

        if model_type == "sgs":
            enc = SGSEncoder(len(words), d_s=d_s, d_f=d_f, n_passes=2, tau_init=float(d_s))
            enc.vocab = vocab
            enc = enc.to(device)
        else:
            enc = FairSoftmaxEncoder(vocab, d_s=d_s, d_f=d_f).to(device)

        n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        t0 = time.time()
        test_sp, val_sp = train_nli_contrastive(enc, nli_loader, stsb_val, stsb_test, device, epochs, 2e-4)
        dt = time.time() - t0
        print(f"  Final: val={val_sp:.4f} test={test_sp:.4f} ({dt:.0f}s)")
        results.append((name, test_sp, val_sp, n_params))

    return results


# ═══════════════════════════════════════════════════════════
# Experiment 3B: NLI 3-Way Classification
# ═══════════════════════════════════════════════════════════

class NLIClassificationDataset(torch.utils.data.Dataset):
    """AllNLI as 3-way classification: entailment/neutral/contradiction."""

    def __init__(self, word2idx, data_dir="data", split="train", max_len=50, max_samples=0):
        import os
        tsv_path = download_allnli(data_dir)
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.data = []

        with open(tsv_path, 'r', encoding='utf-8') as f:
            f.readline()  # header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                if parts[0] != split:
                    continue
                label = label_map.get(parts[5])
                if label is None:
                    continue
                ids_a = tokenize(parts[3], word2idx, max_len)
                ids_b = tokenize(parts[4], word2idx, max_len)
                self.data.append((ids_a, ids_b, label))
                if max_samples > 0 and len(self.data) >= max_samples:
                    break

        print(f"NLI Classification {split}: {len(self.data)} pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def nli_cls_collate(batch):
    ids_a_list, ids_b_list, labels = zip(*batch)
    def pad(seqs):
        ml = max(len(s) for s in seqs)
        p = torch.zeros(len(seqs), ml, dtype=torch.long)
        m = torch.zeros(len(seqs), ml, dtype=torch.float)
        for i, s in enumerate(seqs):
            p[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            m[i, :len(s)] = 1.0
        return p, m
    a, am = pad(ids_a_list)
    b, bm = pad(ids_b_list)
    return a, am, b, bm, torch.tensor(labels, dtype=torch.long)


class NLIClassifier(nn.Module):
    """Encode both sentences, classify relationship."""
    def __init__(self, encoder, d_f=300):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(d_f * 4, d_f),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_f, 3),
        )

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        a = self.encoder(ids_a, mask_a)
        b = self.encoder(ids_b, mask_b)
        combined = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)
        return self.classifier(combined)


def run_exp_3b(args, word2idx, vectors, freqs, words, device):
    print(f"\n{'='*70}")
    print("EXPERIMENT 3B: NLI 3-Way Classification")
    print(f"{'='*70}")

    train_ds = NLIClassificationDataset(word2idx, split="train", max_samples=300000)
    dev_ds = NLIClassificationDataset(word2idx, split="dev")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=nli_cls_collate, num_workers=0, pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=128, shuffle=False, collate_fn=nli_cls_collate, num_workers=0, pin_memory=True)

    results = []

    for name, model_type in [("SGS-2pass", "sgs"), ("Fair Softmax", "softmax"), ("Mean-pool", "mean_pool")]:
        print(f"\n  --- {name} ---")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        d_s, d_f = 64, 300
        vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        vocab.init_from_glove(vectors, freqs)
        vocab = vocab.to(device)

        if model_type == "sgs":
            enc = SGSEncoder(len(words), d_s=d_s, d_f=d_f, n_passes=2, tau_init=float(d_s))
            enc.vocab = vocab
            enc = enc.to(device)
        elif model_type == "softmax":
            enc = FairSoftmaxEncoder(vocab, d_s=d_s, d_f=d_f).to(device)
        else:
            enc = MeanPoolEncoder(vocab).to(device)

        model = NLIClassifier(enc, d_f=d_f).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
        best_acc, best_state = 0, None

        for epoch in range(1, 6):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for a, am, b, bm, labels in train_loader:
                a, am, b, bm, labels = a.to(device), am.to(device), b.to(device), bm.to(device), labels.to(device)
                logits = model(a, am, b, bm)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += labels.size(0)
            scheduler.step()

            # Dev evaluation
            model.eval()
            dev_correct, dev_total = 0, 0
            with torch.no_grad():
                for a, am, b, bm, labels in dev_loader:
                    a, am, b, bm, labels = a.to(device), am.to(device), b.to(device), bm.to(device), labels.to(device)
                    logits = model(a, am, b, bm)
                    dev_correct += (logits.argmax(-1) == labels).sum().item()
                    dev_total += labels.size(0)

            dev_acc = dev_correct / dev_total
            train_acc = correct / total
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    Epoch {epoch}/5 | loss={total_loss/total:.4f} | train_acc={train_acc:.4f} | dev_acc={dev_acc:.4f} | best={best_acc:.4f}")

        results.append((name, best_acc, n_params))

    return results


# ═══════════════════════════════════════════════════════════
# Experiment 3A: SCAN Compositional Generalization
# ═══════════════════════════════════════════════════════════

def run_exp_3a(args, device):
    print(f"\n{'='*70}")
    print("EXPERIMENT 3A: SCAN Compositional Generalization")
    print(f"{'='*70}")

    from src.scan import get_scan_dataloaders
    from src.seq2seq import SGSSeq2Seq, TransformerSeq2Seq

    results = []

    for split in ["addprim_jump", "length"]:
        print(f"\n  === SCAN split: {split} ===")

        train_loader, test_loader, in_vocab, out_vocab, out_idx2word = get_scan_dataloaders(
            split=split, batch_size=64,
        )

        for name, model_cls, kwargs in [
            ("SGS Seq2Seq", SGSSeq2Seq, {"d_model": 128, "n_passes": 2}),
            ("Transformer", TransformerSeq2Seq, {"d_model": 128, "nhead": 4}),
        ]:
            print(f"\n    --- {name} ({split}) ---")
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            model = model_cls(len(in_vocab), len(out_vocab), **kwargs).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Params: {n_params:,}")

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

            best_acc = 0
            for epoch in range(1, 21):
                # Train
                model.train()
                total_loss, n_batches = 0, 0
                for src, src_mask, tgt, tgt_mask in train_loader:
                    src, src_mask = src.to(device), src_mask.to(device)
                    tgt, tgt_mask = tgt.to(device), tgt_mask.to(device)

                    logits = model(src, src_mask, tgt, tgt_mask)  # [batch, tgt_len-1, vocab]
                    target = tgt[:, 1:]  # shift: predict from position 1 onward

                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        target.reshape(-1),
                        ignore_index=0,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
                scheduler.step()

                # Evaluate: sequence-level accuracy
                if epoch % 5 == 0 or epoch == 1:
                    model.eval()
                    correct, total = 0, 0
                    for src, src_mask, tgt, tgt_mask in test_loader:
                        src, src_mask = src.to(device), src_mask.to(device)
                        tgt = tgt.to(device)

                        preds = model.greedy_decode(src, src_mask,
                                                    bos_id=out_vocab["<BOS>"],
                                                    eos_id=out_vocab["<EOS>"])

                        # Compare full sequences (exact match)
                        target_seqs = tgt[:, 1:]  # remove BOS
                        for i in range(preds.shape[0]):
                            pred_list = preds[i].tolist()
                            tgt_list = target_seqs[i].tolist()

                            # Trim at EOS
                            if out_vocab["<EOS>"] in pred_list:
                                pred_list = pred_list[:pred_list.index(out_vocab["<EOS>"])]
                            if out_vocab["<EOS>"] in tgt_list:
                                tgt_list = tgt_list[:tgt_list.index(out_vocab["<EOS>"])]
                            # Remove padding
                            tgt_list = [t for t in tgt_list if t != 0]

                            if pred_list == tgt_list:
                                correct += 1
                            total += 1

                    acc = correct / max(total, 1)
                    best_acc = max(best_acc, acc)
                    print(f"    Epoch {epoch:2d}/20 | loss={total_loss/n_batches:.4f} | "
                          f"test_seq_acc={acc:.4f} ({correct}/{total}) | best={best_acc:.4f}")

            results.append((f"{name} ({split})", best_acc, n_params))

            # Show a few examples
            model.eval()
            src, src_mask, tgt, _ = next(iter(test_loader))
            src, src_mask = src[:3].to(device), src_mask[:3].to(device)
            preds = model.greedy_decode(src, src_mask, bos_id=out_vocab["<BOS>"], eos_id=out_vocab["<EOS>"])
            in_idx2word = {v: k for k, v in in_vocab.items()}
            print(f"\n    Examples:")
            for i in range(min(3, preds.shape[0])):
                src_words = [in_idx2word.get(t, "?") for t in src[i].tolist() if t != 0]
                pred_words = [out_idx2word.get(t, "?") for t in preds[i].tolist()
                              if t != 0 and t != out_vocab["<EOS>"]]
                tgt_words = [out_idx2word.get(t, "?") for t in tgt[i].tolist()
                             if t != 0 and t != out_vocab["<BOS>"] and t != out_vocab["<EOS>"]]
                print(f"      IN:   {' '.join(src_words)}")
                print(f"      PRED: {' '.join(pred_words)}")
                print(f"      GOLD: {' '.join(tgt_words)}")
                print()

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_results = {}

    # Load GloVe for experiments 3C and 3B
    if args.exp in ("3c", "3b", "all"):
        word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)

    if args.exp in ("3c", "all"):
        results = run_exp_3c(args, word2idx, vectors, freqs, words, device)
        all_results["3C"] = results

    if args.exp in ("3b", "all"):
        results = run_exp_3b(args, word2idx, vectors, freqs, words, device)
        all_results["3B"] = results

    if args.exp in ("3a", "all"):
        results = run_exp_3a(args, device)
        all_results["3A"] = results

    # Summary
    print(f"\n\n{'='*70}")
    print("PHASE 3 SUMMARY")
    print(f"{'='*70}")

    if "3C" in all_results:
        print(f"\n  Experiment 3C: Close the Gaps (NLI → STS-B)")
        print(f"  {'Model':<35s} {'STS-B Test':>12s} {'Params':>10s}")
        print(f"  {'-'*60}")
        for name, test, val, params in sorted(all_results["3C"], key=lambda x: x[1], reverse=True):
            print(f"  {name:<35s} {test:>12.4f} {params:>10,}")

    if "3B" in all_results:
        print(f"\n  Experiment 3B: NLI 3-Way Classification")
        print(f"  {'Model':<35s} {'Dev Acc':>12s} {'Params':>10s}")
        print(f"  {'-'*60}")
        for name, acc, params in sorted(all_results["3B"], key=lambda x: x[1], reverse=True):
            print(f"  {name:<35s} {acc:>12.4f} {params:>10,}")

    if "3A" in all_results:
        print(f"\n  Experiment 3A: SCAN Compositional Generalization")
        print(f"  {'Model':<35s} {'Seq Acc':>12s} {'Params':>10s}")
        print(f"  {'-'*60}")
        for name, acc, params in sorted(all_results["3A"], key=lambda x: x[1], reverse=True):
            marker = " <<<" if acc > 0.5 and "SGS" in name else ""
            print(f"  {name:<35s} {acc:>12.4f} {params:>10,}{marker}")

    # Save
    with open("phase3_results.json", "w") as f:
        json.dump({k: [(n, a) for n, a, *_ in v] for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to phase3_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGS Phase 3")
    parser.add_argument("--glove", type=str, default="data/glove.6B.300d.txt")
    parser.add_argument("--exp", type=str, default="all", choices=["3a", "3b", "3c", "all"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
