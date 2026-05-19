"""
Planck 1.4 chat harness with conversation-memory blobs.

Loads a Planck 1.3 checkpoint, creates a dynamic blob store, and runs
a CLI chat loop. Each turn is encoded as a blob for future retrieval.
Older turns are retrieved by similarity with recency decay.

Usage:
    python scripts/chat_planck14.py ^
      --checkpoint checkpoints/planck13/best.pt ^
      --tokenizer data/wikipedia/tokenizer.model

Options:
    --max-blobs 512     conversation memory capacity
    --n-recent 3        last N turns included verbatim
    --k-retrieve 5      older turns retrieved by similarity
    --decay 0.05        recency decay rate
    --max-new 200       max tokens to generate per response
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Planck 1.4 chat with memory blobs")
    p.add_argument("--checkpoint", required=True, help="Planck checkpoint path")
    p.add_argument("--tokenizer", required=True, help="SentencePiece model path")
    p.add_argument("--max-blobs", type=int, default=512)
    p.add_argument("--n-recent", type=int, default=3)
    p.add_argument("--k-retrieve", type=int, default=5)
    p.add_argument("--decay", type=float, default=0.05)
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={sp.get_piece_size()})")

    # Load model
    print(f"Loading Planck: {args.checkpoint}")
    from src.sgs_lm import SGSLanguageModel, migrate_state_dict
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = migrate_state_dict(state)
    vocab_size = state["tok_mu.weight"].shape[0]

    model = SGSLanguageModel(vocab_size=vocab_size)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Setup conversation memory
    from src.conversation_memory import (
        DynamicBlobStore, TurnEncoder, HybridRetriever,
        ConversationSession, Turn,
    )

    blob_store = DynamicBlobStore(
        max_blobs=args.max_blobs,
        d_s=model.d_s,
        d_f=model.d_f,
    ).to(device)

    turn_encoder = TurnEncoder(model, sp)
    retriever = HybridRetriever(
        blob_store, turn_encoder,
        n_recent=args.n_recent,
        k_retrieve=args.k_retrieve,
        decay=args.decay,
    )

    session = ConversationSession(session_id="cli")
    print(f"\nPlanck 1.4 chat ready. Memory capacity: {args.max_blobs} turns.")
    print(f"  Last {args.n_recent} turns verbatim + top-{args.k_retrieve} retrieved.")
    print(f"  Type 'quit' to exit, '/memory' to see store stats.\n")

    @torch.no_grad()
    def generate_response(prompt_text: str) -> str:
        """Generate a response from the prompt context."""
        token_ids = sp.encode(prompt_text, out_type=int)
        # Truncate from the LEFT to fit context window
        max_ctx = 512 - args.max_new
        if len(token_ids) > max_ctx:
            token_ids = token_ids[-max_ctx:]

        ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)

        generated = []
        for _ in range(args.max_new):
            logits = model(ids_t)
            next_logits = logits[0, -1, :] / args.temperature

            # Top-k filtering
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
            ids_t = torch.cat([ids_t, torch.tensor([[next_id]], device=device)], dim=1)

            # Truncate if too long
            if ids_t.shape[1] > 512:
                ids_t = ids_t[:, -512:]

        return sp.decode(generated)

    # Chat loop
    turn_idx = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "/memory":
            print(f"  Blobs stored: {blob_store.n_valid}/{args.max_blobs}")
            print(f"  Total turns: {len(session.turns)}")
            print(f"  Recent window: last {args.n_recent}")
            continue

        # Build context with memory retrieval
        context = retriever.build_context(session, user_input)

        # Generate
        t0 = time.time()
        response = generate_response(context)
        elapsed = time.time() - t0

        print(f"Planck: {response}")
        print(f"  [{elapsed:.1f}s, {blob_store.n_valid} blobs in memory]")

        # Write this turn as a blob
        mu_s, features = turn_encoder.encode_turn(user_input, response)
        blob_store.write(mu_s, features, timestamp=float(turn_idx))

        # Store the turn
        session.turns.append(Turn(
            turn_idx=turn_idx,
            user_msg=user_input,
            assistant_msg=response,
            timestamp=time.time(),
        ))
        turn_idx += 1

    print(f"\nSession ended. {len(session.turns)} turns, {blob_store.n_valid} blobs written.")


if __name__ == "__main__":
    main()
