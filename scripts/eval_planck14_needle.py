"""
Planck 1.4 needle-in-conversation benchmark.

Injects a random factoid at a specific turn in a synthetic conversation,
then asks about it at the final turn. Measures whether the memory system
retrieves the needle correctly.

Usage:
    python scripts/eval_planck14_needle.py `
      --checkpoint checkpoints/planck13/best.pt `
      --tokenizer data/wikipedia/tokenizer.model `
      --mode hybrid

Modes:
    hybrid          last-N verbatim + similarity retrieval (default)
    no-retrieval    truncation only, no blob memory
    similarity-only similarity retrieval without recency decay
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Needle generation ──────────────────────────────────────────────────

NEEDLE_TEMPLATES = [
    ("The capital of {country} is {city}.", "{city}"),
    ("My favorite number is {number}.", "{number}"),
    ("The password to my vault is {code}.", "{code}"),
    ("I was born in the year {year}.", "{year}"),
    ("My pet's name is {name}.", "{name}"),
    ("The secret ingredient is {ingredient}.", "{ingredient}"),
    ("I keep my spare key under the {object}.", "{object}"),
    ("My childhood address was {address}.", "{address}"),
]

FAKE_COUNTRIES = [
    "Zarbia", "Floondia", "Krelvatia", "Moondor", "Plixtar",
    "Quendalia", "Yorthwick", "Blimfar", "Snarlovia", "Wumpus",
]

FAKE_CITIES = [
    "Floonville", "Blarghton", "Mizzleplex", "Quorvath", "Zinkledorf",
    "Primbleton", "Snazzleburg", "Glorpwick", "Twiddleham", "Frazzleton",
]

FAKE_NAMES = [
    "Glorpnax", "Mizzleworth", "Frazzlebottom", "Snickerdoodle",
    "Wumpleton", "Blinkmire", "Quazzleflip", "Primbly", "Zorkington",
]

FAKE_INGREDIENTS = [
    "powdered starfruit", "liquid moonbeam", "crushed rainbow",
    "fermented cloud", "distilled thunder", "pickled lightning",
]

FAKE_OBJECTS = [
    "purple flowerpot", "third stepping stone", "ceramic frog",
    "leftmost gnome", "hollow rock", "fake cactus",
]


def generate_needle():
    """Generate a random needle fact and its expected answer substring."""
    template_text, answer_template = random.choice(NEEDLE_TEMPLATES)

    country = random.choice(FAKE_COUNTRIES)
    city = random.choice(FAKE_CITIES)
    number = str(random.randint(1000, 99999))
    code = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8))
    year = str(random.randint(1850, 2010))
    name = random.choice(FAKE_NAMES)
    ingredient = random.choice(FAKE_INGREDIENTS)
    obj = random.choice(FAKE_OBJECTS)
    address = f"{random.randint(1, 999)} {random.choice(FAKE_CITIES)} Street"

    replacements = {
        "{country}": country,
        "{city}": city,
        "{number}": number,
        "{code}": code,
        "{year}": year,
        "{name}": name,
        "{ingredient}": ingredient,
        "{object}": obj,
        "{address}": address,
    }

    fact = template_text
    answer = answer_template
    for k, v in replacements.items():
        fact = fact.replace(k, v)
        answer = answer.replace(k, v)

    return fact, answer


# ── Filler turn generation ─────────────────────────────────────────────

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

NEEDLE_QUESTIONS = [
    "What did I tell you about {topic} earlier? What was the answer?",
    "Do you remember what I said about {topic}? Repeat it.",
    "Earlier I mentioned something about {topic}. What was it?",
    "Recall the {topic} fact I shared. What is the answer?",
]


def get_needle_query(fact: str) -> str:
    """Generate a question asking the model to recall the needle."""
    # Extract a topic keyword from the fact
    if "capital" in fact:
        topic = "the capital"
    elif "favorite number" in fact:
        topic = "my favorite number"
    elif "password" in fact:
        topic = "the password"
    elif "born" in fact:
        topic = "when I was born"
    elif "pet" in fact:
        topic = "my pet's name"
    elif "ingredient" in fact:
        topic = "the secret ingredient"
    elif "spare key" in fact:
        topic = "the spare key"
    elif "address" in fact:
        topic = "my childhood address"
    else:
        topic = "that fact"

    template = random.choice(NEEDLE_QUESTIONS)
    return template.format(topic=topic)


# ── Main evaluation logic ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Planck 1.4 needle-in-conversation benchmark"
    )
    p.add_argument("--checkpoint", required=True, help="Planck checkpoint path")
    p.add_argument("--tokenizer", required=True, help="SentencePiece model path")
    p.add_argument("--n-turns", type=int, default=100, help="Conversation length")
    p.add_argument("--needle-turn", type=int, default=10, help="Turn to inject needle")
    p.add_argument("--n-trials", type=int, default=10, help="Number of conversations")
    p.add_argument("--max-blobs", type=int, default=512, help="Blob store capacity")
    p.add_argument("--n-recent", type=int, default=3, help="Verbatim recent turns")
    p.add_argument("--k-retrieve", type=int, default=5, help="Top-k similarity")
    p.add_argument("--decay", type=float, default=0.05, help="Recency decay rate")
    p.add_argument("--max-new", type=int, default=100, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument(
        "--mode", default="hybrid",
        choices=["hybrid", "no-retrieval", "similarity-only"],
        help="Retrieval mode"
    )
    p.add_argument(
        "--output", default="results/planck14_needle.json",
        help="Output JSON path"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", default="auto")
    return p.parse_args()


@torch.no_grad()
def generate_response(model, sp, prompt_text: str, args, device) -> str:
    """Generate tokens from a prompt string."""
    token_ids = sp.encode(prompt_text, out_type=int)
    max_ctx = 512 - args.max_new
    if len(token_ids) > max_ctx:
        token_ids = token_ids[-max_ctx:]

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

    return sp.decode(generated)


def run_single_trial(
    model, sp, blob_store_class, turn_encoder_class, retriever_class,
    session_class, turn_class, args, device, trial_idx
):
    """Run one conversation trial and return whether the needle was recalled."""
    from src.conversation_memory import DynamicBlobStore, TurnEncoder, HybridRetriever

    # Fresh blob store per trial
    blob_store = blob_store_class(
        max_blobs=args.max_blobs,
        d_s=model.d_s,
        d_f=model.d_f,
    ).to(device)

    turn_encoder = turn_encoder_class(model, sp)

    # Configure retriever based on mode
    if args.mode == "no-retrieval":
        # No similarity retrieval, only recent turns
        retriever = retriever_class(
            blob_store, turn_encoder,
            n_recent=args.n_recent,
            k_retrieve=0,
            decay=args.decay,
        )
    elif args.mode == "similarity-only":
        # No recency decay (set decay to 0)
        retriever = retriever_class(
            blob_store, turn_encoder,
            n_recent=args.n_recent,
            k_retrieve=args.k_retrieve,
            decay=0.0,
        )
    else:  # hybrid
        retriever = retriever_class(
            blob_store, turn_encoder,
            n_recent=args.n_recent,
            k_retrieve=args.k_retrieve,
            decay=args.decay,
        )

    session = session_class(session_id=f"trial_{trial_idx}")

    # Generate needle for this trial
    needle_fact, needle_answer = generate_needle()

    # Run conversation
    for turn_idx in range(args.n_turns):
        if turn_idx == args.needle_turn:
            # Inject the needle
            user_msg = f"Here is an important fact to remember: {needle_fact}"
        elif turn_idx == args.n_turns - 1:
            # Final turn: ask about the needle
            user_msg = get_needle_query(needle_fact)
        else:
            # Filler
            user_msg = random.choice(FILLER_QUESTIONS)

        # Build context
        if args.mode == "no-retrieval":
            # Truncation only: use only recent turns, no blob retrieval
            parts = []
            recent_start = max(0, len(session.turns) - args.n_recent)
            for i in range(recent_start, len(session.turns)):
                t = session.turns[i]
                parts.append(f"User: {t.user_msg}")
                parts.append(f"Assistant: {t.assistant_msg}")
            parts.append(f"User: {user_msg}")
            parts.append("Assistant:")
            context = "\n".join(parts)
        else:
            context = retriever.build_context(session, user_msg)

        # Generate response
        response = generate_response(model, sp, context, args, device)

        # Write blob (even for no-retrieval mode, so we measure the store)
        mu_s, features = turn_encoder.encode_turn(user_msg, response)
        blob_store.write(mu_s, features, timestamp=float(turn_idx))

        # Store turn
        session.turns.append(turn_class(
            turn_idx=turn_idx,
            user_msg=user_msg,
            assistant_msg=response,
            timestamp=float(turn_idx),
        ))

    # ── Metric 1: Generation recall (does the answer appear in output?) ──
    final_response = session.turns[-1].assistant_msg
    generation_recalled = needle_answer.lower() in final_response.lower()

    # ── Metric 2: Retrieval recall (is the needle blob in top-k at query time?) ──
    # Re-run retrieval for the final query to check what got surfaced
    retrieval_recalled = False
    needle_blob_rank = -1
    if args.mode != "no-retrieval":
        final_query = turn_encoder.encode_query(session.turns[-1].user_msg)
        indices, scores, _ = blob_store.retrieve(
            final_query,
            current_time=float(args.n_turns),
            decay=args.decay if args.mode == "hybrid" else 0.0,
        )
        # The needle was written at turn=needle_turn, so its blob timestamp = needle_turn
        needle_timestamps = blob_store.timestamps[indices].tolist() if indices.numel() > 0 else []
        for rank, ts in enumerate(needle_timestamps):
            if abs(ts - args.needle_turn) < 0.5:
                retrieval_recalled = True
                needle_blob_rank = rank
                break

    # ── Metric 3: Perplexity on needle completion ──
    # Feed context + "The answer is: {needle_answer}" and measure NLL
    needle_completion = f" {needle_answer}"
    completion_ids = sp.encode(needle_completion, out_type=int)
    if len(completion_ids) > 0:
        # Build the context that was used for the final turn
        if args.mode == "no-retrieval":
            parts = []
            recent_start = max(0, len(session.turns) - 1 - args.n_recent)
            for i in range(recent_start, len(session.turns) - 1):
                t = session.turns[i]
                parts.append(f"User: {t.user_msg}")
                parts.append(f"Assistant: {t.assistant_msg}")
            parts.append(f"User: {session.turns[-1].user_msg}")
            parts.append("Assistant:")
            final_context = "\n".join(parts)
        else:
            final_context = retriever.build_context(
                session_class(session_id="ppl", turns=session.turns[:-1]),
                session.turns[-1].user_msg,
            )

        context_ids = sp.encode(final_context, out_type=int)
        # Truncate context from left to fit
        max_ctx = 512 - len(completion_ids)
        if len(context_ids) > max_ctx:
            context_ids = context_ids[-max_ctx:]

        full_ids = context_ids + completion_ids
        ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ids_t)
        # NLL on just the completion tokens
        completion_start = len(context_ids)
        completion_logits = logits[0, completion_start - 1:-1, :]
        targets = ids_t[0, completion_start:]
        import torch.nn.functional as F
        nll = F.cross_entropy(completion_logits, targets).item()
        needle_ppl = float(torch.exp(torch.tensor(nll)).item())
    else:
        needle_ppl = float("inf")
        nll = float("inf")

    return {
        "trial": trial_idx,
        "needle_fact": needle_fact,
        "needle_answer": needle_answer,
        "needle_turn": args.needle_turn,
        "query_turn": args.n_turns - 1,
        "final_response": final_response,
        "generation_recalled": generation_recalled,
        "retrieval_recalled": retrieval_recalled,
        "needle_blob_rank": needle_blob_rank,
        "needle_ppl": needle_ppl,
        "needle_nll": nll,
        "n_blobs_written": blob_store.n_valid,
    }


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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = migrate_state_dict(state)
    vocab_size = state["tok_mu.weight"].shape[0]

    model = SGSLanguageModel(vocab_size=vocab_size)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Import memory classes
    from src.conversation_memory import (
        DynamicBlobStore, TurnEncoder, HybridRetriever,
        ConversationSession, Turn,
    )

    # Run trials
    print(f"\nRunning {args.n_trials} trials, mode={args.mode}")
    print(f"  Conversation length: {args.n_turns} turns")
    print(f"  Needle injected at turn {args.needle_turn}")
    print(f"  Max blobs: {args.max_blobs}")
    print()

    results = []
    t0 = time.time()

    for trial_idx in range(args.n_trials):
        trial_result = run_single_trial(
            model, sp,
            DynamicBlobStore, TurnEncoder, HybridRetriever,
            ConversationSession, Turn,
            args, device, trial_idx,
        )
        results.append(trial_result)

        gen = "GEN_YES" if trial_result["generation_recalled"] else "GEN_NO"
        ret = "RET_YES" if trial_result["retrieval_recalled"] else "RET_NO"
        ppl = trial_result["needle_ppl"]
        elapsed = time.time() - t0
        print(
            f"  Trial {trial_idx + 1}/{args.n_trials}: {gen} | {ret} | "
            f"PPL={ppl:.1f} (answer='{trial_result['needle_answer']}') "
            f"[{elapsed:.0f}s]"
        )

    # Compute summary
    total_time = time.time() - t0
    n_gen_recalled = sum(1 for r in results if r["generation_recalled"])
    n_ret_recalled = sum(1 for r in results if r["retrieval_recalled"])
    gen_recall_rate = n_gen_recalled / len(results) if results else 0.0
    ret_recall_rate = n_ret_recalled / len(results) if results else 0.0
    avg_ppl = sum(r["needle_ppl"] for r in results) / len(results) if results else float("inf")

    summary = {
        "mode": args.mode,
        "n_trials": args.n_trials,
        "n_turns": args.n_turns,
        "needle_turn": args.needle_turn,
        "max_blobs": args.max_blobs,
        "n_recent": args.n_recent,
        "k_retrieve": args.k_retrieve,
        "decay": args.decay,
        "generation_recall_rate": gen_recall_rate,
        "retrieval_recall_rate": ret_recall_rate,
        "avg_needle_ppl": round(avg_ppl, 2),
        "n_gen_recalled": n_gen_recalled,
        "n_ret_recalled": n_ret_recalled,
        "total_time_s": round(total_time, 1),
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
    }

    output_data = {
        "summary": summary,
        "trials": results,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"  NEEDLE BENCHMARK SUMMARY")
    print(f"  Mode:            {args.mode}")
    print(f"  Turns:           {args.n_turns}")
    print(f"  Needle at:       turn {args.needle_turn}")
    print(f"  Trials:          {args.n_trials}")
    print(f"  Generation recall:  {gen_recall_rate:.1%} ({n_gen_recalled}/{args.n_trials})")
    print(f"  Retrieval recall:   {ret_recall_rate:.1%} ({n_ret_recalled}/{args.n_trials})")
    print(f"  Avg needle PPL:     {avg_ppl:.1f}")
    print(f"  Total time:         {total_time:.0f}s")
    print("=" * 60)

    # Gate check (retrieval recall is the primary metric for 100M model)
    gate_threshold = 0.60
    if ret_recall_rate >= gate_threshold:
        print(f"\n  GATE PASS: retrieval recall {ret_recall_rate:.1%} >= {gate_threshold:.0%}")
    else:
        print(f"\n  GATE FAIL: retrieval recall {ret_recall_rate:.1%} < {gate_threshold:.0%}")
        print(f"  (Generation recall {gen_recall_rate:.1%} is expected to be low for 100M base LM)")


if __name__ == "__main__":
    main()
