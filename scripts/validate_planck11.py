"""
Planck 1.1 validation gates — automated pass/fail.

Runs the four hard gates from SETUP.md §6.2 Step 4 and emits a JSON
report plus a human-readable summary. Exits non-zero if any gate fails.

Usage:
    python scripts/validate_planck11.py
    python scripts/validate_planck11.py --samples 50 --eval-batches 100
    python scripts/validate_planck11.py --skip-gate-1  # if --t-max=0.0 run not available

Gates:
    1. Base generation intact: Planck 1.1 with t_max=0.0 stays within
       ABS_TOL of Planck 1.0 validation loss.
    2. Blobs being used:       mean effective blob weight across a batch
       is above MIN_BLOB_WEIGHT.
    3. Perplexity improves:    Planck 1.1 validation loss is strictly
       lower than Planck 1.0.
    4. Repetition decreases:   Planck 1.1 generates fewer repeated
       4-grams than Planck 1.0 on matched prompts.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel, migrate_state_dict
from src.sgs_lm_hsgs import HSGSLanguageModel
from src.blob_store import BlobStore
from src.tinystories import get_dataloader

# Gate thresholds. Tune these if your empirical baselines differ.
GATE_1_ABS_TOL = 0.05        # val-loss delta vs Planck 1.0 for t_max=0.0
GATE_2_MIN_BLOB_WEIGHT = 0.05  # min mean effective blob weight
GATE_3_MIN_IMPROVEMENT = 0.0   # strictly lower val loss vs Planck 1.0
GATE_4_MIN_DELTA = 0           # Planck 1.1 must have fewer 4-gram repeats


def parse_args():
    p = argparse.ArgumentParser(description="Planck 1.1 validation gates")
    p.add_argument("--planck10-checkpoint", default="checkpoints/planck/best.pt")
    p.add_argument("--planck11-checkpoint", default="checkpoints/planck11/best.pt")
    p.add_argument("--planck11-noblob-checkpoint",
                   default="checkpoints/planck11_noablob/best.pt",
                   help="Planck 1.1 trained with --t-max 0.0 (for gate 1)")
    p.add_argument("--blob-dir", default="data/blobs/tinystories")
    p.add_argument("--data-dir", default="data/tinystories")
    p.add_argument("--tokenizer", default=None,
                   help="Path to .model (auto-detected if None)")

    # Architecture (must match checkpoints)
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)

    # Blob config
    p.add_argument("--n-blobs", type=int, default=50000)
    p.add_argument("--blob-k", type=int, default=8)
    p.add_argument("--t-max", type=float, default=0.3)

    # Eval knobs
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=100,
                   help="Number of batches for val-loss measurement")
    p.add_argument("--samples", type=int, default=50,
                   help="Samples per model for the repetition gate")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)

    # Output
    p.add_argument("--output", default="results/planck11_validation.json")

    # Skip flags for partial runs
    p.add_argument("--skip-gate-1", action="store_true")
    p.add_argument("--skip-gate-2", action="store_true")
    p.add_argument("--skip-gate-3", action="store_true")
    p.add_argument("--skip-gate-4", action="store_true")
    return p.parse_args()


def _resolve_tokenizer(args):
    import sentencepiece as spm
    if args.tokenizer and Path(args.tokenizer).exists():
        path = args.tokenizer
    else:
        path = str(Path(args.data_dir) / "tokenizer.model")
    return spm.SentencePieceProcessor(model_file=path)


def _load_planck10(args, device):
    ckpt = torch.load(args.planck10_checkpoint, map_location=device, weights_only=False)
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
    return model.to(device).eval()


def _load_planck11(args, checkpoint_path, device, t_max_override=None):
    blob_data = torch.load(Path(args.blob_dir) / "blobs.pt",
                           map_location="cpu", weights_only=False)
    n_blobs = blob_data["mu"].shape[0]
    t_max = args.t_max if t_max_override is None else t_max_override
    blob_store = BlobStore(
        n_blobs=n_blobs, d_s=args.d_s, d_f=args.d_f,
        k=args.blob_k, t_max=t_max,
    )
    blob_store.init_from_clusters(
        mu=blob_data["mu"],
        log_var=blob_data["log_var"],
        alpha=blob_data["raw_alpha"],
        features=blob_data["features"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt

    base = SGSLanguageModel(
        vocab_size=state["base.tok_mu.weight"].shape[0],
        d_s=args.d_s, d_f=args.d_f,
        n_passes=args.n_passes, n_heads=args.n_heads,
        max_len=args.context_len, ffn_mult=args.ffn_mult,
    )
    model = HSGSLanguageModel(base, blob_store)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def measure_val_loss(model, val_loader, eval_batches, device):
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        n += 1
    avg = total_loss / max(n, 1)
    return avg, math.exp(min(avg, 20))


@torch.no_grad()
def measure_blob_weight(model, val_loader, eval_batches, device):
    """Mean effective blob contribution: E[(1-t_res) * gate] across positions."""
    total = 0.0
    count = 0
    for i, (x, _) in enumerate(val_loader):
        if i >= eval_batches:
            break
        x = x.to(device)
        B, L = x.shape
        query = model._compute_query(x)
        _, t_residual = model.blobs.render(query)            # [B]
        word_meaning = model._base_forward_meaning(x)        # [B, L, d_f]
        blob_meaning = model.blob_proj(model.blobs.render(query)[0])
        blob_expanded = blob_meaning.unsqueeze(1).expand_as(word_meaning)
        combined_ctx = torch.cat([blob_expanded, word_meaning], dim=-1)
        gate = model.blob_gate(combined_ctx).squeeze(-1)     # [B, L]
        t_res = t_residual.view(B, 1)                        # [B, 1]
        eff = ((1 - t_res) * gate)                           # [B, L]
        total += eff.mean().item()
        count += 1
    return total / max(count, 1)


def _count_4gram_repeats(ids: list[int]) -> int:
    if len(ids) < 4:
        return 0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values() if c > 1)


@torch.no_grad()
def measure_repetition(model, sp, prompts, max_new, temperature, top_k, device):
    total = 0
    for prompt in prompts:
        ids = [sp.bos_id()] + sp.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new=max_new, temperature=temperature, top_k=top_k)
        gen = out[0].tolist()[len(ids):]  # only count new tokens
        total += _count_4gram_repeats(gen)
    return total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sp = _resolve_tokenizer(args)
    val_bin = str(Path(args.data_dir) / "val.bin")
    val_loader = get_dataloader(val_bin, args.context_len, args.batch_size,
                                shuffle=False, num_workers=0)

    # Shared prompts for gate 4
    prompts = [
        "Once upon a time", "The little girl", "In a faraway land",
        "The dog and the cat", "One sunny morning", "There was a brave",
        "Tim and Sam were", "A tiny mouse", "The old wizard", "Lily looked at",
    ]
    while len(prompts) < args.samples:
        prompts.extend(prompts)
    prompts = prompts[:args.samples]

    report = {
        "thresholds": {
            "gate_1_abs_tol": GATE_1_ABS_TOL,
            "gate_2_min_blob_weight": GATE_2_MIN_BLOB_WEIGHT,
            "gate_3_min_improvement": GATE_3_MIN_IMPROVEMENT,
            "gate_4_min_delta": GATE_4_MIN_DELTA,
        },
        "gates": {},
    }
    pass_flags = []

    # ── Gate 3 (also gives us Planck 1.0 baseline used by gate 1) ──
    print("\n[Planck 1.0] Loading and measuring val loss...")
    t0 = time.time()
    planck10 = _load_planck10(args, device)
    planck10_loss, planck10_ppl = measure_val_loss(planck10, val_loader,
                                                   args.eval_batches, device)
    print(f"  Planck 1.0: val_loss={planck10_loss:.4f} ppl={planck10_ppl:.1f} "
          f"({time.time()-t0:.1f}s)")
    del planck10
    torch.cuda.empty_cache() if device.type == "cuda" else None

    if not args.skip_gate_3:
        print("\n[Gate 3] Perplexity improves vs Planck 1.0")
        t0 = time.time()
        planck11 = _load_planck11(args, args.planck11_checkpoint, device)
        planck11_loss, planck11_ppl = measure_val_loss(planck11, val_loader,
                                                       args.eval_batches, device)
        delta = planck10_loss - planck11_loss
        passed = delta > GATE_3_MIN_IMPROVEMENT
        report["gates"]["gate_3_perplexity"] = {
            "planck10_loss": planck10_loss,
            "planck11_loss": planck11_loss,
            "delta": delta,
            "planck10_ppl": planck10_ppl,
            "planck11_ppl": planck11_ppl,
            "passed": passed,
        }
        pass_flags.append(passed)
        print(f"  Planck 1.1: val_loss={planck11_loss:.4f} ppl={planck11_ppl:.1f}")
        print(f"  Delta: {delta:+.4f} (need > {GATE_3_MIN_IMPROVEMENT:.3f}) "
              f"→ {'PASS' if passed else 'FAIL'} ({time.time()-t0:.1f}s)")

        # ── Gate 2: reuse loaded Planck 1.1 model ──
        if not args.skip_gate_2:
            print("\n[Gate 2] Blobs being used (mean effective weight)")
            t0 = time.time()
            mean_w = measure_blob_weight(planck11, val_loader,
                                         min(args.eval_batches, 50), device)
            passed = mean_w > GATE_2_MIN_BLOB_WEIGHT
            report["gates"]["gate_2_blob_usage"] = {
                "mean_blob_weight": mean_w,
                "passed": passed,
            }
            pass_flags.append(passed)
            print(f"  Mean effective blob weight: {mean_w:.4f} "
                  f"(need > {GATE_2_MIN_BLOB_WEIGHT:.3f}) "
                  f"→ {'PASS' if passed else 'FAIL'} ({time.time()-t0:.1f}s)")

        # ── Gate 4: Planck 1.1 generation (reuse loaded model) ──
        if not args.skip_gate_4:
            print(f"\n[Gate 4] 4-gram repetition over {args.samples} samples")
            t0 = time.time()
            planck11_reps = measure_repetition(planck11, sp, prompts,
                                               args.max_new, args.temperature,
                                               args.top_k, device)
            print(f"  Planck 1.1 repeats: {planck11_reps} ({time.time()-t0:.1f}s)")
        else:
            planck11_reps = None

        del planck11
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Gate 1: needs the t_max=0.0 checkpoint ──
    if not args.skip_gate_1:
        print("\n[Gate 1] Base generation intact (Planck 1.1 with t_max=0.0)")
        noblob_path = Path(args.planck11_noblob_checkpoint)
        if not noblob_path.exists():
            print(f"  SKIP: {noblob_path} not found. Run "
                  f"scripts/train_planck11.py --t-max 0.0 "
                  f"--save-dir {noblob_path.parent} first.")
            report["gates"]["gate_1_base_intact"] = {
                "passed": None,
                "skipped": True,
                "reason": f"{noblob_path} not found",
            }
        else:
            t0 = time.time()
            planck11_noblob = _load_planck11(args, str(noblob_path), device,
                                             t_max_override=0.0)
            noblob_loss, noblob_ppl = measure_val_loss(planck11_noblob, val_loader,
                                                       args.eval_batches, device)
            diff = abs(noblob_loss - planck10_loss)
            passed = diff <= GATE_1_ABS_TOL
            report["gates"]["gate_1_base_intact"] = {
                "planck10_loss": planck10_loss,
                "planck11_noblob_loss": noblob_loss,
                "abs_diff": diff,
                "passed": passed,
            }
            pass_flags.append(passed)
            print(f"  Planck 1.1 (t_max=0.0): val_loss={noblob_loss:.4f}")
            print(f"  |delta| = {diff:.4f} (need <= {GATE_1_ABS_TOL:.3f}) "
                  f"→ {'PASS' if passed else 'FAIL'} ({time.time()-t0:.1f}s)")
            del planck11_noblob
            torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Gate 4: needs Planck 1.0 repetition count too ──
    if not args.skip_gate_4 and not args.skip_gate_3 and planck11_reps is not None:
        print(f"\n[Gate 4] Comparing to Planck 1.0 repetitions...")
        t0 = time.time()
        planck10 = _load_planck10(args, device)
        planck10_reps = measure_repetition(planck10, sp, prompts,
                                           args.max_new, args.temperature,
                                           args.top_k, device)
        del planck10
        torch.cuda.empty_cache() if device.type == "cuda" else None
        delta = planck10_reps - planck11_reps
        passed = delta > GATE_4_MIN_DELTA
        report["gates"]["gate_4_repetition"] = {
            "planck10_repeats": planck10_reps,
            "planck11_repeats": planck11_reps,
            "delta": delta,
            "passed": passed,
        }
        pass_flags.append(passed)
        print(f"  Planck 1.0 repeats: {planck10_reps} "
              f"| Planck 1.1 repeats: {planck11_reps}")
        print(f"  Delta: {delta} (need > {GATE_4_MIN_DELTA}) "
              f"→ {'PASS' if passed else 'FAIL'} ({time.time()-t0:.1f}s)")

    # ── Summary ──
    all_passed = all(pass_flags) if pass_flags else False
    report["summary"] = {
        "all_passed": all_passed,
        "gates_run": len(pass_flags),
        "gates_passed": sum(pass_flags),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for gate_name, gate_data in report["gates"].items():
        status = gate_data.get("passed")
        sym = "PASS" if status is True else ("FAIL" if status is False else "SKIP")
        print(f"  [{sym}] {gate_name}")
    print(f"\n  {sum(pass_flags)}/{len(pass_flags)} gates passed")
    print(f"  Report: {out}")

    if all_passed and pass_flags:
        print("\n  >>> Blobs validated. Promote to Hertz 1.2. <<<")
        sys.exit(0)
    else:
        print("\n  >>> Validation FAILED. Drop blobs from 1.2 scope or investigate. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
