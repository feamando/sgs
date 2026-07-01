"""Score a Hertz checkpoint (milestone bf16 or full step_*.pt) on the val set.

Milestones are model-only, so we rebuild the model with the SAME hyperparams as
the training launch and load just the weights. Use to compare candidates (e.g.
milestone_295000_bf16.pt vs best.pt) before picking the artifact to keep.

Run on the training box (needs the weights + val.bin there):

  python scripts/eval_ckpt.py --ckpt checkpoints/hertz12/milestone_295000_bf16.pt \
    --val-bin data/hertz12_data/val.bin --d-f 3700 --eval-steps 500
"""
import argparse
import math
import torch

from src.sgs_lm import SGSLanguageModel
from src.tinystories import get_dataloader
from scripts.train_hertz import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to milestone_*.pt or step_*.pt / best.pt")
    p.add_argument("--val-bin", required=True, help="Path to val.bin")
    # Model config — MUST match the training launch. Launch used --d-f 3700.
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--d-s", type=int, default=256)
    p.add_argument("--d-f", type=int, default=3700)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=500,
                   help="Val batches to average over. Higher = tighter estimate.")
    p.add_argument("--mixed-precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.mixed_precision]

    model = SGSLanguageModel(
        vocab_size=args.vocab_size, d_s=args.d_s, d_f=args.d_f,
        n_passes=args.n_passes, n_heads=args.n_heads, max_len=args.context_len,
        ffn_mult=args.ffn_mult, dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    step = ckpt.get("opt_step", "?") if isinstance(ckpt, dict) else "?"
    print(f"Loaded {args.ckpt} (opt_step={step})")

    val_loader = get_dataloader(args.val_bin, args.context_len, args.batch_size,
                                shuffle=False, num_workers=0)
    val_loss, val_ppl = evaluate(model, val_loader, args.eval_steps, device, amp_dtype)
    print(f"  val loss {val_loss:.4f}  ppl {val_ppl:.2f}  (over {args.eval_steps} batches)")


if __name__ == "__main__":
    main()
