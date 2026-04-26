"""
Analysis tools for Raum 1.0 routing bridge.

The old per-token mu_proj probe is gone because the bridge is now
context-aware: a word's predicted position depends on the sentence it
sits in. All probes here feed full token sequences through the model
and read the heads.
"""

from __future__ import annotations

import numpy as np
import torch

from .vocab import OBJECT_NAMES


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _tokenize(words: list[str], word2idx: dict[str, int], device) -> tuple[torch.Tensor, torch.Tensor]:
    unk = word2idx.get("<unk>", word2idx.get("unk", 0))
    ids = [word2idx.get(w.lower(), unk) for w in words]
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(token_ids, dtype=torch.float32)
    return token_ids, mask


def _run(model, vocab, words: list[str]) -> dict:
    device = next(model.parameters()).device
    token_ids, mask = _tokenize(words, model._word2idx, device) if hasattr(model, "_word2idx") else (None, None)
    # Fall through: caller passes vocab + word2idx through explicit helpers.
    raise RuntimeError("internal helper: call probe_* functions instead")


# ──────────────────────────────────────────────────────────────────────
# Sentence probes
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_sentence(
    model,
    vocab,
    word2idx: dict[str, int],
    sentence: str,
) -> dict:
    """Run one sentence and return per-token head outputs + argmax info."""
    device = next(model.parameters()).device
    words = sentence.lower().split()
    token_ids, mask = _tokenize(words, word2idx, device)

    mu_s, _, _, features = vocab.get_params(token_ids)
    out = model(mu_s, features, mask)

    positions = out["positions"][0].cpu().numpy()
    tpl_probs = torch.softmax(out["template_logits"][0], dim=-1).cpu().numpy()
    role_probs = torch.softmax(out["role_logits"][0], dim=-1).cpu().numpy()
    colors = out["colors"][0].cpu().numpy()
    scales = out["scales"][0].cpu().numpy()

    per_word = []
    for i, w in enumerate(words):
        tpl_id = int(tpl_probs[i].argmax())
        per_word.append({
            "word": w,
            "position": positions[i].tolist(),
            "template": OBJECT_NAMES.get(tpl_id, f"id{tpl_id}"),
            "template_confidence": float(tpl_probs[i, tpl_id]),
            "role": int(role_probs[i].argmax()),
            "color": colors[i].tolist(),
            "scale": float(scales[i]),
        })
    return {"sentence": sentence, "words": per_word}


@torch.no_grad()
def probe_interpolation(
    model,
    vocab,
    word2idx: dict[str, int],
    template: list[str],
    swap_index: int,
    word_a: str,
    word_b: str,
    n_steps: int = 5,
) -> list[dict]:
    """
    Interpolate between two words at a chosen position in a fixed
    template sentence, read the resulting position head.

    Example:
        template = ["a", "red", "?", "above", "a", "blue", "cube"]
        swap_index = 2, word_a = "sphere", word_b = "cube"
    """
    device = next(model.parameters()).device
    results = []
    for step in range(n_steps + 1):
        t = step / n_steps
        # Linearly blend the GloVe vectors for word_a and word_b, then
        # tokenize the rest normally. Easier: just replace the token at
        # swap_index with a weighted sum of two embeddings by doing it in
        # vocab space — but the current vocab.get_params expects integer
        # ids, so we cheat by picking whichever side has more weight.
        # That gives us a coarse 2-step interpolation; good enough for
        # a smoke-test. Finer interpolation would need a mu_s injection
        # API on the vocab.
        w = word_a if t < 0.5 else word_b
        sent = " ".join(template[:swap_index] + [w] + template[swap_index + 1:])
        probe = probe_sentence(model, vocab, word2idx, sent)
        results.append({"t": t, "probe": probe})
    return results


# ──────────────────────────────────────────────────────────────────────
# Batched evaluation (comp-gen metrics)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_routing(
    model,
    vocab,
    loader,
    device,
) -> dict:
    """Comp-gen metrics on a labelled loader.

    Returns:
        pos_mse           — mean position MSE across all object tokens
        tpl_acc           — template classification accuracy
        dir_acc           — pairwise direction accuracy (active axes only)
        color_mse         — RGB MSE across object tokens
        n_samples         — total samples
    """
    from .bridge import compute_routing_loss

    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        mask = batch["mask"].to(device)
        mu_s, _, _, features = vocab.get_params(token_ids)
        out = model(mu_s, features, mask)
        _, metrics = compute_routing_loss(out, batch)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    if n == 0:
        return {}
    return {k: v / n for k, v in totals.items()}


# ──────────────────────────────────────────────────────────────────────
# Pretty printers
# ──────────────────────────────────────────────────────────────────────

def print_sentence_probe(probe: dict, file=None):
    print(f'\n  "{probe["sentence"]}"', file=file)
    for w in probe["words"]:
        pos = w["position"]
        col = w["color"]
        print(
            f"    {w['word']:>10s}  pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
            f"tpl={w['template']:<8s}({w['template_confidence']*100:4.0f}%)  "
            f"col=({col[0]:.2f},{col[1]:.2f},{col[2]:.2f})  scl={w['scale']:.2f}",
            file=file,
        )


def print_eval(metrics: dict, label: str, file=None):
    if not metrics:
        return
    print(f"\n  {label}", file=file)
    print(
        f"    loss     {metrics.get('loss', 0):.3f}\n"
        f"    pos_mse  {metrics.get('pos_mse', 0):.3f}\n"
        f"    tpl_acc  {metrics.get('tpl_acc', 0)*100:5.1f}%\n"
        f"    dir_acc  {metrics.get('dir_acc', 0)*100:5.1f}%\n"
        f"    col_mse  {metrics.get('col_mse', 0):.3f}\n"
        f"    scl_mse  {metrics.get('scl_mse', 0):.3f}\n"
        f"    role_ce  {metrics.get('role_ce', 0):.3f}",
        file=file,
    )
