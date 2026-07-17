"""VSP v1, phase 4: the disambiguation benchmark -- the gate that proves the win.

Val loss will NOT show whether VSP helped; disambiguation will. This scores
sense-correct next-token prediction on minimal pairs: the SAME polysemous word
in two sense-contexts, each with a sense-appropriate vs sense-inappropriate
continuation. A model that disambiguates prefers the matching continuation.

  "the crane flew over the ..."   -> should favor sky/lake  over  site/steel
  "the crane lifted the heavy ..." -> should favor steel/beam over sky/feathers

Score = fraction of (context, right, wrong) triples where
  logP(right | context) > logP(wrong | context).
Chance = 0.5. This is tokenizer-AGNOSTIC (works for VSP and the SentencePiece
baseline), so it's the FAIR gate: run both checkpoints, compare.

GATE: Planck 2.0 (VSP) beats the --random-init baseline at matched params/tokens.
That delta is the publishable result.

Usage (--pairs defaults to the 105-pair asset; --out keeps VSP/baseline separate):
  python scripts/eval_disambiguation.py --checkpoint checkpoints/planck2_vsp/final.pt \
    --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt \
    --subword-model data/hertz12_data/tokenizer.model --out results/disambig_vsp.json
  python scripts/eval_disambiguation.py --selftest
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.sgs_lm import SGSLanguageModel

# The real gate runs on the 105-pair / 42-word asset. Default to it (resolved
# from repo root, so it works from any cwd) rather than the 8-pair inline set,
# so forgetting --pairs never silently downgrades the benchmark to 8 pairs.
DEFAULT_PAIRS_FILE = REPO_ROOT / "scripts" / "assets" / "disambig_pairs.json"


def load_model(ckpt_path, n_tokens, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["arch"]
    model = SGSLanguageModel(
        vocab_size=n_tokens, d_s=a["d_s"], d_f=a["d_f"], n_passes=a["n_passes"],
        n_heads=a["n_heads"], max_len=a["max_len"], ffn_mult=a["ffn_mult"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def seq_logprob(model, ids, device, ctx_len):
    """Sum logP of the continuation tokens given the context. `ids` is the full
    (context + continuation) id list; we score only the continuation positions."""
    x = torch.tensor([ids], dtype=torch.long, device=device)[:, -model.max_len:]
    logits = model(x)[0]                     # [L, V]
    logp = F.log_softmax(logits.float(), dim=-1)
    total, n = 0.0, 0
    # score positions predicting the continuation (from ctx_len-1 onward)
    start = max(ctx_len - 1, 0)
    for t in range(start, x.shape[1] - 1):
        total += logp[t, x[0, t + 1]].item(); n += 1
    return total / max(n, 1)


def score_pairs(model, tok, pairs, device):
    """For each triple, is logP(right|ctx) > logP(wrong|ctx)?"""
    correct, total, rows = 0, 0, []
    for p in pairs:
        ctx_ids = tok(p["context"])
        r_ids = ctx_ids + tok(" " + p["right"])
        w_ids = ctx_ids + tok(" " + p["wrong"])
        lr = seq_logprob(model, r_ids, device, len(ctx_ids))
        lw = seq_logprob(model, w_ids, device, len(ctx_ids))
        ok = lr > lw
        correct += ok; total += 1
        rows.append({"word": p.get("word"), "context": p["context"],
                     "right": p["right"], "wrong": p["wrong"],
                     "logp_right": round(lr, 3), "logp_wrong": round(lw, 3), "ok": bool(ok)})
    return correct / max(total, 1), rows


# ── tokenizer adapters (VSP vocab or SentencePiece baseline) ────────────────

def vsp_tokenizer(vocab_path, glove_path, subword_model):
    """Return an encode(text)->[ids] closure using the VSPS tokenizer."""
    from scripts.tokenize_vsps import VSPSTokenizer
    from scripts.validate_p6_correlation import load_glove
    import re
    vj = json.load(open(vocab_path))
    need = set()
    for t in vj["tokens"]:
        need.add(t["surface"]); need.update(re.findall(r"[a-z]+", (t.get("term") or "").lower()))
    glove = load_glove(Path(glove_path), need)
    tk = VSPSTokenizer(vocab_path, glove, subword_model=subword_model)
    return lambda text: tk.encode(text)[0], len(vj["tokens"])


def sp_tokenizer(sp_model):
    """SentencePiece baseline encode."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(sp_model))
    return (lambda text: sp.encode(text, out_type=int)), sp.get_piece_size()


# Tiny inline set for --selftest ONLY (no file I/O). NOT the benchmark; the real
# gate uses DEFAULT_PAIRS_FILE (105 pairs). Do not run the gate on this.
SELFTEST_PAIRS = [
    {"word": "crane", "context": "the crane flew over the calm", "right": "lake", "wrong": "crane"},
    {"word": "crane", "context": "the construction crane lifted the heavy steel", "right": "beam", "wrong": "feather"},
    {"word": "bank", "context": "they sat on the grassy river", "right": "bank", "wrong": "money"},
    {"word": "bat", "context": "the vampire bat flew out of the dark", "right": "cave", "wrong": "stadium"},
    {"word": "bat", "context": "he swung the wooden baseball", "right": "bat", "wrong": "wing"},
    {"word": "seal", "context": "the seal swam in the cold ocean", "right": "water", "wrong": "envelope"},
    {"word": "spring", "context": "the metal coil spring compressed under", "right": "pressure", "wrong": "flowers"},
    {"word": "mouse", "context": "she clicked the computer", "right": "mouse", "wrong": "cheese"},
]


def selftest():
    """No model: check the scoring plumbing + pairs shape with a random model."""
    print("[selftest] eval_disambiguation")
    torch.manual_seed(0)
    n = 60
    model = SGSLanguageModel(vocab_size=n, d_s=16, d_f=32, n_passes=2, n_heads=2, max_len=64)
    model.eval()
    tok = lambda text: [abs(hash(w)) % n for w in text.split()]
    acc, rows = score_pairs(model, tok, SELFTEST_PAIRS, "cpu")
    ok = 0.0 <= acc <= 1.0 and len(rows) == len(SELFTEST_PAIRS)
    print(f"[selftest] random-model acc {acc:.2f} (~chance 0.5), {len(rows)} pairs "
          f"| {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Disambiguation benchmark (VSP gate)")
    p.add_argument("--checkpoint", help="planck2 checkpoint (.pt)")
    p.add_argument("--vocab", default="data/vsps/vocab.json")
    p.add_argument("--glove", default="data/glove.6B.300d.txt")
    p.add_argument("--subword-model", default="data/hertz12_data/tokenizer.model")
    p.add_argument("--sp-baseline", default=None,
                   help="score a plain-SentencePiece checkpoint instead of VSP vocab")
    p.add_argument("--pairs", default=None,
                   help=f"json list of {{word,context,right,wrong}} "
                        f"(default: {DEFAULT_PAIRS_FILE.relative_to(REPO_ROOT)}, 105 pairs)")
    p.add_argument("--out", default="results/disambig_eval.json")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.checkpoint:
        raise SystemExit("--checkpoint required unless --selftest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs_path = Path(args.pairs) if args.pairs else DEFAULT_PAIRS_FILE
    if not pairs_path.exists():
        raise SystemExit(
            f"[eval] pairs file not found: {pairs_path}\n"
            f"       The gate needs the 105-pair asset. Pass --pairs or restore "
            f"{DEFAULT_PAIRS_FILE.relative_to(REPO_ROOT)}.")
    pairs = json.load(open(pairs_path))
    # Guard against silently running the gate on a stub set. The real benchmark
    # is 105 pairs; anything under ~50 is not statistically meaningful.
    if len(pairs) < 50:
        raise SystemExit(
            f"[eval] {pairs_path} has only {len(pairs)} pairs (<50). The gate is "
            f"not meaningful below ~50; pass --pairs a full set to override.")
    print(f"[eval] {len(pairs)} pairs <- {pairs_path}")

    if args.sp_baseline:
        tok, n_tokens = sp_tokenizer(args.sp_baseline)
        print(f"[eval] SentencePiece tokenizer, vocab {n_tokens}")
    else:
        tok, n_tokens = vsp_tokenizer(args.vocab, args.glove, args.subword_model)
        print(f"[eval] VSPS tokenizer, vocab {n_tokens}")

    model = load_model(args.checkpoint, n_tokens, device)
    acc, rows = score_pairs(model, tok, pairs, device)

    print(f"\n{'word':<10}{'right':<10}{'wrong':<10}{'lp_right':>10}{'lp_wrong':>10}  ok")
    for r in rows:
        print(f"{r['word']:<10}{r['right']:<10}{r['wrong']:<10}"
              f"{r['logp_right']:>10.3f}{r['logp_wrong']:>10.3f}  {'Y' if r['ok'] else '.'}")
    print(f"\n[eval] disambiguation accuracy: {acc:.3f} ({sum(r['ok'] for r in rows)}/{len(rows)}) "
          f"| chance 0.5")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checkpoint": args.checkpoint, "accuracy": acc, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"[eval] saved -> {args.out}")


if __name__ == "__main__":
    main()
