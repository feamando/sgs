"""
Evaluate a Radiance (Planck/Hertz) checkpoint against LM baselines.

Metrics:
  - Perplexity on FineWeb-Edu val split (requires data/fineweb/val.bin)
  - HellaSwag (0-shot, multiple-choice loglikelihood)
  - ARC-Easy (0-shot, multiple-choice loglikelihood)

Usage:
    python scripts/evaluate_lm.py --checkpoint checkpoints/hertz/best.pt
    python scripts/evaluate_lm.py --checkpoint checkpoints/hertz/best.pt --tasks perplexity,hellaswag
    python scripts/evaluate_lm.py --checkpoint checkpoints/planck/best.pt --limit 100 --tasks hellaswag
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel
from src.tinystories import get_dataloader
from scripts.generate import infer_arch


ALL_TASKS = ["perplexity", "hellaswag", "arc_easy"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SGS LM on standard benchmarks")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="Path to sentencepiece .model (auto-detected if None)")
    p.add_argument("--tasks", default=",".join(ALL_TASKS),
                   help=f"Comma-separated: {ALL_TASKS}")
    p.add_argument("--val-bin", default="data/fineweb/val.bin",
                   help="Perplexity eval data (binary uint16 tokens)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap examples per task (useful for smoke tests)")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="results/hertz_eval.json")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def load_checkpoint(path: str, device: torch.device) -> SGSLanguageModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    arch = infer_arch(state)
    print(f"  Inferred arch: {arch}")
    model = SGSLanguageModel(**arch)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_tokenizer(tok_path: str | None, checkpoint: str):
    import sentencepiece as spm

    if tok_path is None:
        ckpt_dir = Path(checkpoint).parent
        candidates = [
            Path("data/fineweb/tokenizer.model"),
            Path("data/tinystories/tokenizer.model"),
            ckpt_dir.parent.parent / "data" / "fineweb" / "tokenizer.model",
            ckpt_dir.parent.parent / "data" / "tinystories" / "tokenizer.model",
        ]
        for c in candidates:
            if c.exists():
                tok_path = str(c)
                break
    if tok_path is None:
        raise FileNotFoundError("Tokenizer not found. Pass --tokenizer.")
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    print(f"  Tokenizer: {tok_path} (vocab={sp.get_piece_size()})")
    return sp


# ────────────────────────────────────────────────────────────
# Task 1: Perplexity
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_perplexity(model, val_bin: str, batch_size: int, device, amp_dtype, limit=None):
    if not Path(val_bin).exists():
        print(f"  [skip] {val_bin} not found. Train Hertz first or pass --val-bin.")
        return None

    loader = get_dataloader(
        val_bin, model.max_len, batch_size, shuffle=False, num_workers=0,
    )
    total_loss = 0.0
    total_tokens = 0
    max_batches = limit if limit else len(loader)
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
            )
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_nll = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 20))
    print(f"  Perplexity: {ppl:.2f}  (nll={avg_nll:.4f}, {total_tokens:,} tokens, {time.time()-t0:.1f}s)")
    return {"perplexity": ppl, "nll": avg_nll, "n_tokens": total_tokens}


# ────────────────────────────────────────────────────────────
# Task 2/3: Multiple-choice loglikelihood
# ────────────────────────────────────────────────────────────

@torch.no_grad()
def score_continuation(model, sp, ctx: str, cont: str, device, amp_dtype) -> float:
    """
    Return sum log P(cont | ctx) under the model.

    Scoring convention follows lm-eval-harness:
      tokens = sp.encode(ctx + cont)
      ctx_tokens = sp.encode(ctx)
      Score only the positions corresponding to cont tokens.
    """
    ctx_ids = sp.encode(ctx)
    full_ids = sp.encode(ctx + cont)
    # Align: handle boundary artifacts by recomputing cont as diff
    cont_len = len(full_ids) - len(ctx_ids)
    if cont_len <= 0:
        # Tokenization ate the continuation (rare), fall back to standalone encode
        cont_ids = sp.encode(cont)
        full_ids = ctx_ids + cont_ids
        cont_len = len(cont_ids)
    # Truncate from the left if it overflows context window
    max_len = model.max_len
    if len(full_ids) > max_len:
        full_ids = full_ids[-max_len:]
        cont_len = min(cont_len, max_len - 1)  # at least one ctx token

    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=amp_dtype,
                            enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
        logits = model(x)  # [1, L, V]
    # Predict position t from logits[t-1]. Score cont tokens only.
    # cont occupies positions [len(full) - cont_len, len(full) - 1].
    # Their logits come from [len(full) - cont_len - 1, len(full) - 2].
    L = len(full_ids)
    start = L - cont_len
    logp = F.log_softmax(logits[0, start - 1 : L - 1, :].float(), dim=-1)
    target = torch.tensor(full_ids[start:], dtype=torch.long, device=device)
    token_logp = logp.gather(1, target.unsqueeze(-1)).squeeze(-1)
    return token_logp.sum().item()


def eval_hellaswag(model, sp, device, amp_dtype, limit=None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] `datasets` not installed")
        return None
    try:
        ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"  [skip] HellaSwag download failed: {e}")
        return None

    n = min(limit, len(ds)) if limit else len(ds)
    correct = 0
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        ctx = item["ctx_a"] + " " + item["ctx_b"].capitalize() if item.get("ctx_b") else item["ctx_a"]
        endings = item["endings"]
        gold = int(item["label"])
        # Length-normalize (standard for HellaSwag) to avoid favoring short completions
        scores = []
        for e in endings:
            raw = score_continuation(model, sp, ctx, " " + e, device, amp_dtype)
            n_tok = max(len(sp.encode(" " + e)), 1)
            scores.append(raw / n_tok)
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == gold)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{n}] running acc: {correct/(i+1):.3f}")

    acc = correct / n
    print(f"  HellaSwag acc: {acc:.4f}  ({correct}/{n}, {time.time()-t0:.1f}s)")
    return {"accuracy": acc, "n": n}


def eval_arc_easy(model, sp, device, amp_dtype, limit=None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [skip] `datasets` not installed")
        return None
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    except Exception as e:
        print(f"  [skip] ARC-Easy download failed: {e}")
        return None

    n = min(limit, len(ds)) if limit else len(ds)
    correct = 0
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        ctx = "Question: " + item["question"] + "\nAnswer:"
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold_label = item["answerKey"]
        if gold_label not in labels:
            continue
        gold = labels.index(gold_label)
        scores = []
        for c in choices:
            raw = score_continuation(model, sp, ctx, " " + c, device, amp_dtype)
            n_tok = max(len(sp.encode(" " + c)), 1)
            scores.append(raw / n_tok)
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == gold)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{n}] running acc: {correct/(i+1):.3f}")

    acc = correct / n
    print(f"  ARC-Easy acc: {acc:.4f}  ({correct}/{n}, {time.time()-t0:.1f}s)")
    return {"accuracy": acc, "n": n}


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = resolve_device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task: {t}. Choose from {ALL_TASKS}")

    print(f"Device: {device}, amp: {args.amp}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device)
    print(f"  Params: {model.count_parameters()/1e6:.1f}M")

    sp = load_tokenizer(args.tokenizer, args.checkpoint)

    results = {
        "checkpoint": args.checkpoint,
        "params": model.count_parameters(),
        "limit": args.limit,
    }

    if "perplexity" in tasks:
        print("\n[Perplexity, FineWeb-Edu val]")
        results["perplexity"] = eval_perplexity(
            model, args.val_bin, args.batch_size, device, amp_dtype, args.limit,
        )
    if "hellaswag" in tasks:
        print("\n[HellaSwag, 0-shot]")
        results["hellaswag"] = eval_hellaswag(model, sp, device, amp_dtype, args.limit)
    if "arc_easy" in tasks:
        print("\n[ARC-Easy, 0-shot]")
        results["arc_easy"] = eval_arc_easy(model, sp, device, amp_dtype, args.limit)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
