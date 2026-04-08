#!/usr/bin/env python3
"""
Phase 1c: Run all ablations and produce comparison table.

Usage:
    python scripts/run_all_ablations.py --glove data/glove.6B.300d.txt
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path


ABLATIONS = [
    # (name, model_flag, extra_args)
    ("SGS-full (P=4)",       "sgs",              ["--n_passes", "4"]),
    ("SGS-1pass",            "sgs",              ["--n_passes", "1"]),
    ("SGS-2pass",            "sgs",              ["--n_passes", "2"]),
    ("SGS-8pass",            "sgs",              ["--n_passes", "8"]),
    ("Mean-pool features",   "mean_pool",        ["--train_vocab"]),
    ("Mean-pool means (μ)",  "mean_pool_mu",     ["--train_vocab"]),
    ("Softmax attention",    "softmax_attn",     []),
    ("No transmittance",     "no_transmittance", []),
    ("Mean-pool (no train)", "mean_pool",        []),
]


def main(args):
    results = []
    script = Path(__file__).parent / "train_stsb.py"

    for name, model, extra in ABLATIONS:
        print(f"\n{'='*70}")
        print(f"  ABLATION: {name}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable, str(script),
            "--glove", args.glove,
            "--model", model,
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
        ] + extra

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
            )
            output = result.stdout
            print(output[-2000:] if len(output) > 2000 else output)

            # Parse test spearman from output
            for line in output.split('\n'):
                if 'Test Spearman:' in line:
                    spearman = float(line.split(':')[-1].strip())
                    results.append((name, spearman))
                    break
            else:
                results.append((name, None))

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 1 hour")
            results.append((name, None))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, None))

    # Print comparison table
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS — PHASE 1c")
    print(f"{'='*70}\n")
    print(f"{'Model':<30s} {'Test Spearman':>15s} {'Status':>10s}")
    print("-" * 60)

    for name, spearman in sorted(results, key=lambda x: x[1] or 0, reverse=True):
        if spearman is None:
            print(f"{name:<30s} {'FAILED':>15s} {'—':>10s}")
        else:
            status = "✓" if spearman >= 0.78 else ("~" if spearman >= 0.58 else "✗")
            print(f"{name:<30s} {spearman:>15.4f} {status:>10s}")

    # Save results
    results_path = Path("ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {name: spearman for name, spearman in results},
            f, indent=2,
        )
    print(f"\nResults saved to {results_path}")

    # Key comparisons
    results_dict = {name: s for name, s in results if s is not None}
    sgs = results_dict.get("SGS-full (P=4)")
    mp = results_dict.get("Mean-pool features")
    sa = results_dict.get("Softmax attention")

    if sgs and mp:
        diff = sgs - mp
        print(f"\n  SGS vs Mean-pool: {'+' if diff > 0 else ''}{diff:.4f}")
        if diff > 0.05:
            print("  → Rendering equation adds significant value over averaging")
        elif diff > 0:
            print("  → Rendering equation adds marginal value")
        else:
            print("  → KILL GATE: Rendering equation adds nothing")

    if sgs and sa:
        diff = sgs - sa
        print(f"  SGS vs Softmax:   {'+' if diff > 0 else ''}{diff:.4f}")
        if diff > 0:
            print("  → Alpha-compositing beats softmax attention!")
        else:
            print("  → Softmax attention wins → pivot to Gaussian Transformer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
