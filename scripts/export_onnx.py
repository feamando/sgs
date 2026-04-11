"""
Export Radiance Planck to ONNX for browser deployment.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/planck/best.pt
    python scripts/export_onnx.py --checkpoint checkpoints/planck/best.pt --quantize
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sgs_lm import SGSLanguageModel


def parse_args():
    p = argparse.ArgumentParser(description="Export Radiance Planck to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None, help="Output .onnx path (auto if None)")
    p.add_argument("--quantize", action="store_true", help="Also produce int8 quantized model")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", action="store_true", help="Verify ONNX output matches PyTorch")

    # Architecture (must match checkpoint)
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=512)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
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
    model.eval()
    print(f"  Loaded {model.count_parameters()/1e6:.1f}M params")

    # Output path
    if args.output:
        onnx_path = args.output
    else:
        onnx_path = str(Path(args.checkpoint).parent / "planck.onnx")

    # Dummy input
    seq_len = 64  # representative length for tracing
    dummy = torch.randint(0, vocab_size, (1, seq_len))

    # Export
    print(f"Exporting to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["token_ids"],
        output_names=["logits"],
        dynamic_axes={
            "token_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {onnx_path} ({size_mb:.0f} MB)")

    # Verify
    if args.verify:
        print("Verifying ONNX output...")
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path)
            ort_input = {"token_ids": dummy.numpy().astype(np.int64)}
            ort_out = session.run(None, ort_input)[0]

            with torch.no_grad():
                pt_out = model(dummy).numpy()

            max_diff = np.abs(ort_out - pt_out).max()
            mean_diff = np.abs(ort_out - pt_out).mean()
            print(f"  Max diff:  {max_diff:.6f}")
            print(f"  Mean diff: {mean_diff:.6f}")
            if max_diff < 1e-4:
                print("  ✓ ONNX output matches PyTorch")
            else:
                print("  ⚠ Larger than expected diff (may be acceptable for float32 vs graph opts)")
        except ImportError:
            print("  onnxruntime not installed, skipping verification")

    # Quantize
    if args.quantize:
        print("Quantizing to int8...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quant_path = onnx_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(
                onnx_path,
                quant_path,
                weight_type=QuantType.QInt8,
            )
            qsize = Path(quant_path).stat().st_size / (1024 * 1024)
            print(f"  Saved: {quant_path} ({qsize:.0f} MB)")
            print(f"  Compression: {size_mb:.0f} MB → {qsize:.0f} MB ({qsize/size_mb*100:.0f}%)")
        except ImportError:
            print("  onnxruntime.quantization not available, skipping")

    print("\nDone.")


if __name__ == "__main__":
    main()
