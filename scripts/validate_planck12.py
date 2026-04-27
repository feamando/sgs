#!/usr/bin/env python3
"""
Planck 1.2 acceleration-recipe ablation harness.

Drives the six-run matrix from docs/plans/planck_12_validation.md:
  baseline, tl (§2.1), ap (§2.2), sk (§2.3), shk (§2.4), all (compound).

Each run is launched as a subprocess call to scripts/train_lm.py so
GPU state, VRAM, and wandb runs stay isolated. Outputs accumulate in
results/planck_12/ablation.json so partial progress is preserved
across runs and crashes.

Usage:
    # Full matrix
    python scripts/validate_planck12.py --data-dir data/fineweb

    # Single run (for iteration / debugging)
    python scripts/validate_planck12.py --data-dir data/fineweb --only tl

    # Re-run a specific config after it failed
    python scripts/validate_planck12.py --data-dir data/fineweb --only all --force

    # Dry-run: print the train commands without executing them
    python scripts/validate_planck12.py --data-dir data/fineweb --dry-run

    # Adopt an already-trained run from an existing train log without
    # re-executing (useful if the baseline was started manually):
    python scripts/validate_planck12.py --adopt baseline=path/to/log.txt
    # Provide an explicit wall clock (seconds) when adopting; otherwise
    # the harness estimates it from last_step * batch * ctx / tok/s.
    python scripts/validate_planck12.py --adopt baseline=log.txt --adopt-wall-s 10980
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "planck_12"
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "planck_12"

# Six runs. Order matters: baseline first, compound last.
RUNS: list[dict] = [
    {"id": "baseline", "label": "Plain CE, 3 passes", "flags": []},
    {"id": "tl",       "label": "§2.1 only", "flags": ["--transmittance-loss"]},
    {"id": "ap",       "label": "§2.2 only", "flags": ["--adaptive-passes"]},
    {"id": "sk",       "label": "§2.3 only", "flags": ["--sparse-k", "64"]},
    {"id": "shk",      "label": "§2.4 only", "flags": ["--shared-kernel"]},
    {"id": "all",      "label": "all four composed", "flags": [
        "--transmittance-loss",
        "--adaptive-passes",
        "--sparse-k", "64",
        "--shared-kernel",
    ]},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/fineweb",
                   help="Dataset dir containing train.bin + val.bin")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=1,
                   help="Epochs per run (harness does not currently "
                        "re-size for a fixed 500M tokens — caller picks).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=66750,
                   help="Stop each run at this many global steps for parity "
                        "with the adopted baseline (0 = no cap, full epoch)")
    p.add_argument("--only", choices=[r["id"] for r in RUNS],
                   help="Run only this config (for iteration)")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Run IDs to skip")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if results JSON already has this id")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--wandb", action="store_true",
                   help="Enable wandb for each run (adds --wandb)")
    p.add_argument("--adopt", action="append", default=[],
                   metavar="RUN_ID=LOG_PATH",
                   help="Adopt an already-finished run by parsing its log "
                        "and writing the summary row without re-executing. "
                        "May be passed multiple times.")
    p.add_argument("--adopt-wall-s", type=float, default=None,
                   help="Explicit wall-clock seconds for --adopt. If omitted, "
                        "estimated from last_step * batch * ctx / mean tok_per_sec.")
    return p.parse_args()


def load_results() -> dict:
    path = RESULTS_DIR / "ablation.json"
    if not path.exists():
        return {"runs": {}}
    with open(path) as f:
        return json.load(f)


def save_results(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "ablation.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def build_command(run: dict, args: argparse.Namespace) -> list[str]:
    save_dir = CKPT_ROOT / run["id"]
    cmd = [
        sys.executable, "scripts/train_lm.py",
        "--data-dir", args.data_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--context-len", str(args.context_len),
        "--log-interval", str(args.log_interval),
        "--eval-interval", str(args.eval_interval),
        "--save-dir", str(save_dir),
    ]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.wandb:
        cmd += ["--wandb", "--wandb-project", f"planck-12-{run['id']}"]
    cmd += run["flags"]
    return cmd


# ─── Log parsing ─────────────────────────────────────────────────────

# Train log line (sample):
#   epoch 1 step    500 | loss 3.9012 avg 4.1234 | lr 3.00e-04 gnorm 0.84 |
#     tau 85.2 | 11832 tok/s | passes 2.47 T 0.182
_TRAIN_RE = re.compile(
    r"step\s+(?P<step>\d+)\s*\|\s*loss\s+(?P<loss>[\d.]+)\s+avg\s+[\d.]+\s*\|"
    r".*?tau\s+(?P<tau>[\d.]+)\s*\|\s*(?P<toks>[\d.]+)\s+tok/s"
    r"(?:.*?passes\s+(?P<passes>[\d.]+))?"
    r"(?:.*?T\s+(?P<tmean>[\d.]+))?"
)

# Val line: "  >>> val loss 3.7421 ppl 42.3"
_VAL_RE = re.compile(r">>>\s*val loss\s+(?P<loss>[\d.]+)\s+ppl\s+(?P<ppl>[\d.]+)")


def parse_log(text: str) -> dict:
    """Extract final val loss, tok/s mean (last half), passes EMA,
    T mean. Returns empty dict fields on no-match."""
    final_val_loss = None
    final_val_ppl = None
    for m in _VAL_RE.finditer(text):
        final_val_loss = float(m.group("loss"))
        final_val_ppl = float(m.group("ppl"))

    tok_rates = []
    passes_vals = []
    tmean_vals = []
    last_step = 0
    for m in _TRAIN_RE.finditer(text):
        tok_rates.append(float(m.group("toks")))
        last_step = max(last_step, int(m.group("step")))
        if m.group("passes"):
            passes_vals.append(float(m.group("passes")))
        if m.group("tmean"):
            tmean_vals.append(float(m.group("tmean")))

    # Mean over second half of training for stability.
    half = len(tok_rates) // 2
    tail_tok = tok_rates[half:] if tok_rates else []
    tok_mean = sum(tail_tok) / len(tail_tok) if tail_tok else None

    return {
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "tok_per_sec_mean": tok_mean,
        "passes_ema_final": passes_vals[-1] if passes_vals else None,
        "T_mean_final": tmean_vals[-1] if tmean_vals else None,
        "last_step": last_step,
    }


# ─── Run ─────────────────────────────────────────────────────────────

def run_one(run: dict, args: argparse.Namespace, results: dict) -> None:
    run_id = run["id"]
    cmd = build_command(run, args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / run_id / "train_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== run {run_id} ({run['label']}) ===")
    print("  cmd:", " ".join(shlex.quote(c) for c in cmd))
    print("  log:", log_path)

    if args.dry_run:
        return

    t0 = time.time()
    # tee stdout → file so the run is observable live AND parsed offline.
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_f.write(line)
        proc.wait()
    wall = time.time() - t0

    status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
    parsed = parse_log(log_path.read_text())
    results["runs"][run_id] = {
        "label": run["label"],
        "flags": run["flags"],
        "status": status,
        "wall_clock_s": round(wall, 1),
        "cmd": cmd,
        **parsed,
    }
    save_results(results)
    print(f"  done: status={status} wall={wall:.0f}s "
          f"val_loss={parsed['final_val_loss']} "
          f"tok/s={parsed['tok_per_sec_mean']}")


def adopt_run(spec: str, args: argparse.Namespace, results: dict) -> None:
    """Parse a completed train log and record it as a run, no subprocess."""
    if "=" not in spec:
        raise SystemExit(f"--adopt expects RUN_ID=PATH, got {spec!r}")
    run_id, log_path_str = spec.split("=", 1)
    run = next((r for r in RUNS if r["id"] == run_id), None)
    if run is None:
        raise SystemExit(f"--adopt: unknown run id {run_id!r}")

    src = Path(log_path_str).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"--adopt: log not found: {src}")

    # Copy the log into the canonical location so downstream re-parsing works.
    dst_dir = RESULTS_DIR / run_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "train_log.txt"
    dst.write_text(src.read_text(), encoding="utf-8", errors="replace")

    parsed = parse_log(dst.read_text())

    wall = args.adopt_wall_s
    if wall is None and parsed["last_step"] and parsed["tok_per_sec_mean"]:
        tokens = parsed["last_step"] * args.batch_size * args.context_len
        wall = tokens / parsed["tok_per_sec_mean"]

    results["runs"][run_id] = {
        "label": run["label"],
        "flags": run["flags"],
        "status": "adopted",
        "wall_clock_s": round(wall, 1) if wall else None,
        "cmd": None,
        "adopted_from": str(src),
        **parsed,
    }
    save_results(results)
    print(f"adopted {run_id} from {src}: "
          f"val_loss={parsed['final_val_loss']} "
          f"last_step={parsed['last_step']} "
          f"wall≈{wall:.0f}s" if wall else f"adopted {run_id} (wall unknown)")


def summarise(results: dict) -> None:
    runs = results.get("runs", {})
    if not runs:
        return
    baseline = runs.get("baseline")
    print("\n=== summary ===")
    header = f"{'id':<10} {'status':<10} {'val_loss':>10} {'tok/s':>10} {'wall_s':>8} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for run in RUNS:
        r = runs.get(run["id"])
        if not r:
            continue
        speedup = ""
        if baseline and r.get("wall_clock_s") and baseline.get("wall_clock_s"):
            speedup = f"{baseline['wall_clock_s'] / r['wall_clock_s']:.2f}x"
        val_loss = r.get("final_val_loss")
        tok = r.get("tok_per_sec_mean")
        print(
            f"{run['id']:<10} {r.get('status','?'):<10} "
            f"{val_loss if val_loss is not None else '-':>10} "
            f"{(f'{tok:.0f}' if tok else '-'):>10} "
            f"{r.get('wall_clock_s','-'):>8} {speedup:>8}"
        )


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    results = load_results()

    for spec in args.adopt:
        adopt_run(spec, args, results)

    todo = [r for r in RUNS if r["id"] not in args.skip]
    if args.only:
        todo = [r for r in todo if r["id"] == args.only]

    # Adopted runs satisfy their row; don't re-execute unless --force.
    adopted_ids = {s.split("=", 1)[0] for s in args.adopt if "=" in s}
    if adopted_ids and not args.force:
        todo = [r for r in todo if r["id"] not in adopted_ids]

    for run in todo:
        rid = run["id"]
        prior_status = results["runs"].get(rid, {}).get("status")
        if not args.force and prior_status in ("ok", "adopted"):
            print(f"skip {rid} (status={prior_status} "
                  f"in results/planck_12/ablation.json)")
            continue
        run_one(run, args, results)

    summarise(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
