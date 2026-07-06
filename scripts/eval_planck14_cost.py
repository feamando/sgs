"""
Planck 1.4 cost-per-turn measurement.

Runs a synthetic conversation of N turns and measures tokens in context,
tokens generated, and wall-clock time per turn. Compares three modes:
  (a) blob-memory (flat cost via hybrid retrieval)
  (b) full-context (growing cost, all turns concatenated)
  (c) truncation (fixed window, lossy)

Usage:
    python scripts/eval_planck14_cost.py `
      --checkpoint checkpoints/planck13/best.pt `
      --tokenizer data/wikipedia/tokenizer.model

Output: JSON with per-turn measurements and a summary table.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Filler conversation ────────────────────────────────────────────────

FILLER_QUESTIONS = [
    "What's the weather like today?",
    "Can you recommend a good movie?",
    "What are some healthy breakfast ideas?",
    "Tell me about a famous historical event.",
    "What sports are popular in Europe?",
    "How do I make pasta from scratch?",
    "What is the tallest mountain in the world?",
    "Tell me a fun science fact.",
    "What are the best exercises for cardio?",
    "How does photosynthesis work?",
    "What are some tips for better sleep?",
    "Who invented the telephone?",
    "What are the planets in our solar system?",
    "How do airplanes stay in the air?",
    "What is the deepest ocean trench?",
    "Tell me about ancient Rome.",
    "What are good indoor plants for beginners?",
    "How does a computer processor work?",
    "What are some famous paintings?",
    "How do birds migrate?",
    "What causes earthquakes?",
    "Tell me about the water cycle.",
    "What are some popular board games?",
    "How is chocolate made?",
    "What are the major rivers of the world?",
    "Tell me about the moon landing.",
    "What are some good study techniques?",
    "How does wifi work?",
    "What are the different types of clouds?",
    "Tell me about dinosaurs.",
    "What is quantum mechanics about?",
    "How do vaccines work?",
    "What are the tallest buildings in the world?",
    "Tell me about coffee production.",
    "What are musical scales?",
    "How do submarines work?",
    "What causes the northern lights?",
    "Tell me about ancient Egypt.",
    "What are the basics of photography?",
    "How do electric cars work?",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Planck 1.4 cost-per-turn measurement"
    )
    p.add_argument("--checkpoint", required=True, help="Planck checkpoint path")
    p.add_argument("--tokenizer", required=True, help="SentencePiece model path")
    p.add_argument("--n-turns", type=int, default=200, help="Conversation length")
    p.add_argument("--max-blobs", type=int, default=512, help="Blob store capacity")
    p.add_argument("--n-recent", type=int, default=3, help="Verbatim recent turns")
    p.add_argument("--k-retrieve", type=int, default=5, help="Top-k similarity")
    p.add_argument("--decay", type=float, default=0.05, help="Recency decay rate")
    p.add_argument("--max-new", type=int, default=50, help="Max tokens per response")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument(
        "--output", default="results/planck14_cost.json",
        help="Output JSON path"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="auto")
    return p.parse_args()


@torch.no_grad()
def generate_response(model, sp, prompt_text: str, args, device):
    """Generate tokens, returning (response_text, n_input_tokens, n_output_tokens)."""
    token_ids = sp.encode(prompt_text, out_type=int)
    n_input = len(token_ids)

    # Truncate from the left to fit context window
    max_ctx = 512 - args.max_new
    if len(token_ids) > max_ctx:
        token_ids = token_ids[-max_ctx:]
        n_input = len(token_ids)

    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(args.max_new):
        logits = model(ids_t)
        next_logits = logits[0, -1, :] / args.temperature

        if args.top_k > 0:
            topk_vals, topk_idx = next_logits.topk(args.top_k)
            mask = torch.full_like(next_logits, float("-inf"))
            mask.scatter_(0, topk_idx, topk_vals)
            next_logits = mask

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if next_id == sp.eos_id():
            break

        generated.append(next_id)
        ids_t = torch.cat(
            [ids_t, torch.tensor([[next_id]], device=device)], dim=1
        )
        if ids_t.shape[1] > 512:
            ids_t = ids_t[:, -512:]

    return sp.decode(generated), n_input, len(generated)


def build_full_context(turns, current_msg):
    """Full-context mode: concatenate ALL prior turns + current message."""
    parts = []
    for t in turns:
        parts.append(f"User: {t['user_msg']}")
        parts.append(f"Assistant: {t['assistant_msg']}")
    parts.append(f"User: {current_msg}")
    parts.append("Assistant:")
    return "\n".join(parts)


def build_truncation_context(turns, current_msg, n_recent):
    """Truncation mode: only last N turns + current message."""
    parts = []
    recent_start = max(0, len(turns) - n_recent)
    for t in turns[recent_start:]:
        parts.append(f"User: {t['user_msg']}")
        parts.append(f"Assistant: {t['assistant_msg']}")
    parts.append(f"User: {current_msg}")
    parts.append("Assistant:")
    return "\n".join(parts)


def run_mode(mode, model, sp, args, device, blob_store_cls, turn_encoder_cls,
             retriever_cls, session_cls, turn_cls):
    """Run a full conversation in a given mode. Returns per-turn measurements."""
    from src.conversation_memory import DynamicBlobStore, TurnEncoder, HybridRetriever

    measurements = []
    raw_turns = []  # dicts for full-context / truncation modes

    # Setup memory (used in blob-memory mode)
    blob_store = blob_store_cls(
        max_blobs=args.max_blobs,
        d_s=model.d_s,
        d_f=model.d_f,
    ).to(device)
    turn_encoder = turn_encoder_cls(model, sp)
    retriever = retriever_cls(
        blob_store, turn_encoder,
        n_recent=args.n_recent,
        k_retrieve=args.k_retrieve,
        decay=args.decay,
    )
    session = session_cls(session_id=f"cost_{mode}")

    for turn_idx in range(args.n_turns):
        user_msg = random.choice(FILLER_QUESTIONS)

        # Build context based on mode
        if mode == "blob-memory":
            context = retriever.build_context(session, user_msg)
        elif mode == "full-context":
            context = build_full_context(raw_turns, user_msg)
        elif mode == "truncation":
            context = build_truncation_context(raw_turns, user_msg, args.n_recent)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Measure
        t0 = time.time()
        response, n_input, n_output = generate_response(
            model, sp, context, args, device
        )
        wall_time = time.time() - t0

        measurements.append({
            "turn": turn_idx,
            "tokens_in_context": n_input,
            "tokens_generated": n_output,
            "wall_time_s": round(wall_time, 4),
        })

        # Store turn
        turn_data = {"user_msg": user_msg, "assistant_msg": response}
        raw_turns.append(turn_data)

        if mode == "blob-memory":
            mu_s, features = turn_encoder.encode_turn(user_msg, response)
            blob_store.write(mu_s, features, timestamp=float(turn_idx))
            session.turns.append(turn_cls(
                turn_idx=turn_idx,
                user_msg=user_msg,
                assistant_msg=response,
                timestamp=float(turn_idx),
            ))

        # Progress (every 20 turns)
        if (turn_idx + 1) % 20 == 0:
            print(
                f"    [{mode}] turn {turn_idx + 1}/{args.n_turns}: "
                f"{n_input} ctx tokens, {wall_time:.2f}s"
            )

    return measurements


def compute_slope(measurements):
    """Compute the slope of tokens_in_context vs. turn index (linear fit)."""
    n = len(measurements)
    if n < 2:
        return 0.0
    xs = [m["turn"] for m in measurements]
    ys = [m["tokens_in_context"] for m in measurements]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def main():
    args = parse_args()
    random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={sp.get_piece_size()})")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    from src.sgs_lm import SGSLanguageModel, migrate_state_dict
    from scripts.generate import infer_arch
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = migrate_state_dict(state)
    arch = infer_arch(state)  # build with checkpoint's own arch (Planck vs Hertz)
    vocab_size = arch["vocab_size"]

    model = SGSLanguageModel(**arch)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Import memory classes
    from src.conversation_memory import (
        DynamicBlobStore, TurnEncoder, HybridRetriever,
        ConversationSession, Turn,
    )

    modes = ["blob-memory", "full-context", "truncation"]
    all_results = {}

    print(f"\nMeasuring cost-per-turn across {args.n_turns} turns")
    print(f"  Max blobs: {args.max_blobs}, recent: {args.n_recent}")
    print()

    for mode in modes:
        print(f"  Running mode: {mode}...")
        random.seed(args.seed)  # Reset seed so all modes see same questions
        measurements = run_mode(
            mode, model, sp, args, device,
            DynamicBlobStore, TurnEncoder, HybridRetriever,
            ConversationSession, Turn,
        )
        all_results[mode] = measurements

    # Compute summary statistics
    summary = {}
    for mode in modes:
        measurements = all_results[mode]
        ctx_tokens = [m["tokens_in_context"] for m in measurements]
        wall_times = [m["wall_time_s"] for m in measurements]
        slope = compute_slope(measurements)

        summary[mode] = {
            "mean_ctx_tokens": round(sum(ctx_tokens) / len(ctx_tokens), 1),
            "max_ctx_tokens": max(ctx_tokens),
            "min_ctx_tokens": min(ctx_tokens),
            "slope_tokens_per_turn": round(slope, 3),
            "mean_wall_time_s": round(sum(wall_times) / len(wall_times), 4),
            "total_wall_time_s": round(sum(wall_times), 1),
        }

    # Gate check: blob-memory slope < 10% of full-context slope
    blob_slope = summary["blob-memory"]["slope_tokens_per_turn"]
    full_slope = summary["full-context"]["slope_tokens_per_turn"]
    if full_slope > 0:
        slope_ratio = blob_slope / full_slope
    else:
        slope_ratio = 0.0

    gate_pass = slope_ratio < 0.10

    output_data = {
        "config": {
            "n_turns": args.n_turns,
            "max_blobs": args.max_blobs,
            "n_recent": args.n_recent,
            "k_retrieve": args.k_retrieve,
            "decay": args.decay,
            "max_new": args.max_new,
            "checkpoint": args.checkpoint,
            "tokenizer": args.tokenizer,
        },
        "summary": summary,
        "gate": {
            "blob_slope": blob_slope,
            "full_context_slope": full_slope,
            "slope_ratio": round(slope_ratio, 4),
            "threshold": 0.10,
            "pass": gate_pass,
        },
        "measurements": all_results,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  COST-PER-TURN SUMMARY ({args.n_turns} turns)")
    print("=" * 70)
    print(f"  {'Mode':<16} {'Mean ctx':<12} {'Max ctx':<10} {'Slope':<12} {'Mean time':<10}")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for mode in modes:
        s = summary[mode]
        print(
            f"  {mode:<16} {s['mean_ctx_tokens']:<12.1f} "
            f"{s['max_ctx_tokens']:<10} {s['slope_tokens_per_turn']:<12.3f} "
            f"{s['mean_wall_time_s']:<10.4f}"
        )
    print()
    print(f"  Slope ratio (blob / full-context): {slope_ratio:.4f}")
    print(f"  Gate threshold: < 0.10")
    if gate_pass:
        print(f"  GATE PASS: {slope_ratio:.4f} < 0.10")
    else:
        print(f"  GATE FAIL: {slope_ratio:.4f} >= 0.10")
    print("=" * 70)


if __name__ == "__main__":
    main()
