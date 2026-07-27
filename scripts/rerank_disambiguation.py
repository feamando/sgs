"""VSP pivot, probe 1: inference-time rerank with the VSP bundle.

WHY THIS EXISTS. Embedding-init is dead (6-seed 2k: mean +1.6pts, CI includes 0;
40k: -3.8). LM gradients wash the bundle-init warm start out. But the VSP
REPRESENTATION separates senses (grounding gate 0.37). This probe asks the
cheapest possible question: does that grounding help a TRAINED baseline model if
we deliver it as an inference-time signal instead of an init? No retraining --
we reuse the existing --random-init baseline checkpoint.

MECHANISM. The disambiguation gate scores logP(right|ctx) vs logP(wrong|ctx). We
add a sense-consistency term from the bundle:

  score(cand | ctx) = logP(cand | ctx)  +  lambda * consistency(cand, ctx)

  consistency = max over {V,S,P} blocks of cosine(block(ctx_bundle), block(cand_bundle))

  ctx_bundle  = mean bundle of the GROUNDED tokens in the context (the
                disambiguating words -- "river"/"money" -- carry sense)
  cand_bundle = mean bundle of the candidate continuation's grounded tokens

max-agg over blocks mirrors the gating metric (separable if ANY modality
distinguishes). lambda=0 recovers the pure baseline (sanity check == the
baseline's disambig_* accuracy). We sweep lambda and report accuracy at each.

HONEST LIMIT. Rerank can only reorder candidates the base model already
considers; here the candidate set is exactly {right, wrong}, so this is the
CLEANEST test of the signal (no top-k truncation). A win here => grounding has
usable signal that init destroyed => justifies the contrastive aux-loss (probe
2). A flat result at all lambda => the signal does not help a trained LM; the
VSP-for-LMs line closes.

  python scripts/rerank_disambiguation.py --checkpoint checkpoints/planck2_baseline/final.pt \
    --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt \
    --subword-model data/hertz12_data/tokenizer.model --out results/rerank_baseline.json
  python scripts/rerank_disambiguation.py --selftest
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
from scripts.train_planck2 import load_vocab_bundles
from scripts.eval_disambiguation import (
    load_model, seq_logprob, vsp_tokenizer, sp_tokenizer,
    DEFAULT_PAIRS_FILE, SELFTEST_PAIRS,
)

DEFAULT_LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def block_dims(vocab_path):
    d = json.load(open(vocab_path))["dims"]
    return d["V"], d["S"], d["P"]


def bundle_matrix(vocab_path):
    """[n, dV+dS+dP] per-id bundle + per-block validity mask (a block is valid
    for a token only if it is nonzero, so abstract/subword tokens with no V/P
    do not drag the cosine toward zero)."""
    n, V, S, P = load_vocab_bundles(vocab_path)
    B = np.concatenate([V, S, P], axis=1).astype(np.float32)
    has = np.stack([np.abs(V).sum(1) > 0, np.abs(S).sum(1) > 0, np.abs(P).sum(1) > 0], axis=1)
    return B, has, (V.shape[1], S.shape[1], P.shape[1])


def _block_slices(dV, dS, dP):
    return [(0, dV), (dV, dV + dS), (dV + dS, dV + dS + dP)]


def agg_bundle(ids, B, has, blk):
    """Mean bundle over the GROUNDED ids (those with a V block). Returns the
    vector and which blocks are present. If no grounded id, fall back to all ids."""
    ids = [i for i in ids if 0 <= i < len(B)]
    g = [i for i in ids if has[i, 0]]  # grounded == has V
    use = g if g else ids
    if not use:
        return None, np.zeros(3, bool)
    vecs = B[use]
    present = has[use].any(axis=0)
    return vecs.mean(axis=0), present


def consistency(ctx_vec, ctx_present, cand_vec, cand_present, blk):
    """max over blocks present in BOTH of cosine(block_ctx, block_cand)."""
    if ctx_vec is None or cand_vec is None:
        return 0.0
    best = None
    for bi, (a, b) in enumerate(_block_slices(*blk)):
        if not (ctx_present[bi] and cand_present[bi]):
            continue
        u, v = ctx_vec[a:b], cand_vec[a:b]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-8 or nv < 1e-8:
            continue
        c = float(np.dot(u, v) / (nu * nv))
        best = c if best is None else max(best, c)
    return best if best is not None else 0.0


def score_rerank(model, tok, pairs, B, has, blk, lambdas, device):
    """For each pair and each lambda, is base_logp(right)+l*cons(right) >
    base_logp(wrong)+l*cons(wrong)? Base logp computed ONCE per pair."""
    per_lambda = {l: [0, 0] for l in lambdas}   # [correct, total]
    rows = []
    for p in pairs:
        ctx_ids = tok(p["context"])
        r_ids = tok(" " + p["right"])
        w_ids = tok(" " + p["wrong"])
        lr = seq_logprob(model, ctx_ids + r_ids, device, len(ctx_ids))
        lw = seq_logprob(model, ctx_ids + w_ids, device, len(ctx_ids))
        cv, cp = agg_bundle(ctx_ids, B, has, blk)
        rv, rp = agg_bundle(r_ids, B, has, blk)
        wv, wp = agg_bundle(w_ids, B, has, blk)
        cons_r = consistency(cv, cp, rv, rp, blk)
        cons_w = consistency(cv, cp, wv, wp, blk)
        row = {"word": p.get("word"), "context": p["context"],
               "right": p["right"], "wrong": p["wrong"],
               "logp_right": round(lr, 3), "logp_wrong": round(lw, 3),
               "cons_right": round(cons_r, 3), "cons_wrong": round(cons_w, 3), "by_lambda": {}}
        for l in lambdas:
            ok = (lr + l * cons_r) > (lw + l * cons_w)
            per_lambda[l][0] += ok
            per_lambda[l][1] += 1
            row["by_lambda"][str(l)] = bool(ok)
        rows.append(row)
    acc = {l: per_lambda[l][0] / max(per_lambda[l][1], 1) for l in lambdas}
    return acc, rows


def selftest():
    print("[selftest] rerank_disambiguation")
    torch.manual_seed(0)
    n, dV, dS, dP = 60, 8, 6, 4
    model = SGSLanguageModel(vocab_size=n, d_s=16, d_f=32, n_passes=2, n_heads=2, max_len=64)
    model.eval()
    tok = lambda text: [abs(hash(w)) % n for w in text.split()]
    rng = np.random.RandomState(0)
    B = rng.randn(n, dV + dS + dP).astype(np.float32)
    has = np.ones((n, 3), bool)
    acc, rows = score_rerank(model, tok, SELFTEST_PAIRS, B, has, (dV, dS, dP),
                             DEFAULT_LAMBDAS, "cpu")
    ok = (len(rows) == len(SELFTEST_PAIRS)
          and all(0.0 <= a <= 1.0 for a in acc.values())
          and set(acc) == set(DEFAULT_LAMBDAS))
    print(f"[selftest] acc-by-lambda {{l: round(a,2) for...}} = "
          f"{ {l: round(a,2) for l, a in acc.items()} } | {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="VSP inference-time rerank probe")
    ap.add_argument("--checkpoint", help="baseline (or any) planck2 checkpoint")
    ap.add_argument("--vocab", default="data/vsps/vocab.json")
    ap.add_argument("--glove", default="data/glove.6B.300d.txt")
    ap.add_argument("--subword-model", default="data/hertz12_data/tokenizer.model")
    ap.add_argument("--sp-baseline", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--out", default="results/rerank_eval.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.checkpoint:
        raise SystemExit("--checkpoint required unless --selftest")

    pairs_path = Path(args.pairs) if args.pairs else DEFAULT_PAIRS_FILE
    if not pairs_path.exists():
        raise SystemExit(f"[rerank] pairs file not found: {pairs_path}")
    pairs = json.load(open(pairs_path))
    if len(pairs) < 50:
        raise SystemExit(f"[rerank] {pairs_path} has only {len(pairs)} pairs (<50); "
                         f"the probe is not meaningful below ~50.")
    print(f"[rerank] {len(pairs)} pairs <- {pairs_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.sp_baseline:
        tok, n_tokens = sp_tokenizer(args.sp_baseline)
    else:
        tok, n_tokens = vsp_tokenizer(args.vocab, args.glove, args.subword_model)
    B, has, blk = bundle_matrix(args.vocab)
    model = load_model(args.checkpoint, n_tokens, device)

    acc, rows = score_rerank(model, tok, pairs, B, has, blk, args.lambdas, device)

    base = acc[0.0] if 0.0 in acc else None
    print(f"\n{'lambda':>8}{'accuracy':>10}{'vs base':>10}")
    for l in args.lambdas:
        d = f"{acc[l]-base:+.3f}" if base is not None else "n/a"
        print(f"{l:>8.2f}{acc[l]:>10.3f}{d:>10}")
    best_l = max(args.lambdas, key=lambda l: acc[l])
    print(f"\n[rerank] best lambda {best_l} -> {acc[best_l]:.3f} "
          f"(base lambda=0 -> {base:.3f}); delta {acc[best_l]-base:+.3f}"
          if base is not None else f"[rerank] best lambda {best_l} -> {acc[best_l]:.3f}")
    print("[rerank] lambda=0 MUST equal the baseline's disambig accuracy (sanity).")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checkpoint": args.checkpoint, "accuracy_by_lambda": acc, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"[rerank] saved -> {args.out}")


if __name__ == "__main__":
    main()
