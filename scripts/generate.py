"""
Text generation with a trained Radiance Planck model.

Usage:
    python scripts/generate.py --checkpoint checkpoints/planck/best.pt --prompt "Once upon a time"
    python scripts/generate.py --checkpoint checkpoints/planck/best.pt --interactive
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with Radiance Planck")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--tokenizer", default=None, help="Path to .model file (auto-detected if None)")
    p.add_argument("--prompt", default="Once upon a time", help="Text prompt")
    p.add_argument("--max-new", type=int, default=200, help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--interactive", action="store_true", help="Interactive mode")
    p.add_argument("--device", default="auto")

    # Architecture (must match checkpoint)
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    return p.parse_args()


def load_model(args, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt

    # Infer vocab size from embedding weight
    vocab_size = state["tok_mu.weight"].shape[0]

    model = SGSLanguageModel(
        vocab_size=vocab_size,
        d_s=args.d_s,
        d_f=args.d_f,
        n_passes=args.n_passes,
        n_heads=args.n_heads,
        max_len=args.context_len,
        ffn_mult=args.ffn_mult,
    )
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Loaded {model.count_parameters()/1e6:.1f}M params, vocab={vocab_size}")
    return model


def load_tokenizer(args):
    """Load sentencepiece tokenizer."""
    import sentencepiece as spm

    if args.tokenizer:
        path = args.tokenizer
    else:
        # Auto-detect from data dir
        ckpt_dir = Path(args.checkpoint).parent
        candidates = [
            ckpt_dir.parent.parent / "data" / "tinystories" / "tokenizer.model",
            Path("data/tinystories/tokenizer.model"),
        ]
        path = None
        for c in candidates:
            if c.exists():
                path = str(c)
                break
        if path is None:
            raise FileNotFoundError(
                "Tokenizer not found. Pass --tokenizer path/to/tokenizer.model"
            )

    sp = spm.SentencePieceProcessor(model_file=path)
    print(f"  Tokenizer: {path} (vocab={sp.get_piece_size()})")
    return sp


def generate_text(model, sp, prompt, max_new, temperature, top_k, device):
    """Encode prompt, generate, decode."""
    ids = [sp.bos_id()] + sp.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
    )

    # Decode full output (skip BOS)
    all_ids = output_ids[0].tolist()
    # Find EOS if present
    eos = sp.eos_id()
    if eos in all_ids[1:]:
        all_ids = all_ids[: all_ids.index(eos, 1)]
    text = sp.decode(all_ids[1:])  # skip BOS
    return text


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args, device)
    sp = load_tokenizer(args)

    if args.interactive:
        print("\n=== Interactive Generation ===")
        print(f"  temperature={args.temperature}, top_k={args.top_k}, max_new={args.max_new}")
        print("  Type a prompt and press Enter. Type 'quit' to exit.\n")

        while True:
            try:
                prompt = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt or prompt.lower() in ("quit", "exit"):
                break

            # Parse inline params: "prompt /t=0.5 /k=30"
            parts = prompt.split()
            temp = args.temperature
            top_k = args.top_k
            prompt_parts = []
            for part in parts:
                if part.startswith("/t="):
                    temp = float(part[3:])
                elif part.startswith("/k="):
                    top_k = int(part[3:])
                else:
                    prompt_parts.append(part)
            prompt = " ".join(prompt_parts)

            text = generate_text(model, sp, prompt, args.max_new, temp, top_k, device)
            print(f"\n{text}\n")
    else:
        print(f"\nPrompt: {args.prompt}")
        print(f"Config: temperature={args.temperature}, top_k={args.top_k}, max_new={args.max_new}\n")
        text = generate_text(
            model, sp, args.prompt, args.max_new, args.temperature, args.top_k, device
        )
        print(text)


if __name__ == "__main__":
    main()
