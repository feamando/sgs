"""
Post-Planck 1.1 disk cleanup.

Planck 1.1 training writes a checkpoint every --save-interval steps plus one
per epoch, each ~400MB on the current architecture. A full run accumulates
many tens of files. This script removes intermediate checkpoints while
preserving the artefacts you actually care about: best.pt and final.pt.

Default mode is dry-run. Pass --apply to actually delete.

Usage:
    # Show what would be deleted across all planck11* checkpoint dirs
    python scripts/cleanup_planck11.py

    # Actually delete
    python scripts/cleanup_planck11.py --apply

    # Keep the last N intermediates per dir (e.g. for resume flexibility)
    python scripts/cleanup_planck11.py --apply --keep-last 2

    # Also sweep compile caches and __pycache__ under the project
    python scripts/cleanup_planck11.py --apply --include-pycache

Safety:
    - Only operates on paths under ./checkpoints/ and project-local cache dirs.
    - Refuses to touch best.pt, final.pt, or any file outside the allowlist.
    - Prints each deletion.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

KEEP_NAMES = {"best.pt", "final.pt"}
INTERMEDIATE_PATTERNS = (
    re.compile(r"^step_\d+\.pt$"),
    re.compile(r"^epoch_\d+\.pt$"),
)
DEFAULT_CHECKPOINT_GLOBS = ("checkpoints/planck11*",)
CACHE_DIRS = ("torch_compile_cache", ".torch_compile_cache")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _is_intermediate(name: str) -> bool:
    return any(p.match(name) for p in INTERMEDIATE_PATTERNS)


def _step_from_name(name: str) -> int:
    m = re.match(r"^step_(\d+)\.pt$", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^epoch_(\d+)\.pt$", name)
    if m:
        # Sort epochs after steps with the same number by biasing up.
        return int(m.group(1)) * 1_000_000_000
    return -1


def sweep_checkpoint_dir(dir_path: Path, keep_last: int, apply: bool) -> int:
    """Return bytes freed (or bytes that would be freed)."""
    if not dir_path.exists():
        return 0
    files = sorted(
        (f for f in dir_path.iterdir()
         if f.is_file() and _is_intermediate(f.name)),
        key=lambda f: _step_from_name(f.name),
    )
    if keep_last > 0:
        files = files[:-keep_last] if len(files) > keep_last else []

    freed = 0
    for f in files:
        if f.name in KEEP_NAMES:
            continue
        size = f.stat().st_size
        freed += size
        action = "DELETE" if apply else "WOULD DELETE"
        print(f"  [{action}] {f} ({_fmt_bytes(size)})")
        if apply:
            f.unlink()
    return freed


def sweep_cache_dirs(root: Path, apply: bool) -> int:
    """Clear __pycache__ and known compile caches under root."""
    freed = 0
    for cache_name in CACHE_DIRS:
        cache_path = root / cache_name
        if cache_path.exists() and cache_path.is_dir():
            size = _dir_size(cache_path)
            freed += size
            action = "DELETE" if apply else "WOULD DELETE"
            print(f"  [{action}] {cache_path} ({_fmt_bytes(size)})")
            if apply:
                shutil.rmtree(cache_path, ignore_errors=True)

    for pycache in root.rglob("__pycache__"):
        if not pycache.is_dir():
            continue
        # Stay inside the repo — rglob already does, but double-check.
        try:
            pycache.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        size = _dir_size(pycache)
        freed += size
        action = "DELETE" if apply else "WOULD DELETE"
        print(f"  [{action}] {pycache} ({_fmt_bytes(size)})")
        if apply:
            shutil.rmtree(pycache, ignore_errors=True)
    return freed


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def parse_args():
    p = argparse.ArgumentParser(description="Clean up Planck 1.1 checkpoints")
    p.add_argument("--apply", action="store_true",
                   help="Actually delete. Default is dry-run.")
    p.add_argument("--keep-last", type=int, default=0,
                   help="Per checkpoint dir, retain the N most recent "
                        "intermediate (step_*.pt / epoch_*.pt) files.")
    p.add_argument("--include-pycache", action="store_true",
                   help="Also delete __pycache__ and compile caches "
                        "under the project root.")
    p.add_argument("--checkpoint-glob", nargs="+",
                   default=list(DEFAULT_CHECKPOINT_GLOBS),
                   help="Glob(s) under the project root to sweep for "
                        "intermediates.")
    p.add_argument("--root", default=".",
                   help="Project root (default: CWD).")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    if not (root / "SETUP.md").exists() and not (root / "src").exists():
        print(f"Refusing to run: {root} does not look like the SGS repo root. "
              f"cd into the repo or pass --root.")
        sys.exit(2)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Planck 1.1 cleanup ({mode}) — root: {root}")
    if not args.apply:
        print("  (no files will be deleted — re-run with --apply to actually "
              "delete)")

    total_freed = 0
    total_dirs = 0
    for glob in args.checkpoint_glob:
        for ckpt_dir in sorted(root.glob(glob)):
            if not ckpt_dir.is_dir():
                continue
            total_dirs += 1
            print(f"\nCheckpoint dir: {ckpt_dir}")
            freed = sweep_checkpoint_dir(ckpt_dir, args.keep_last, args.apply)
            if freed == 0:
                print("  (nothing to remove)")
            total_freed += freed

    if total_dirs == 0:
        print(f"\nNo checkpoint dirs matched {args.checkpoint_glob}.")

    if args.include_pycache:
        print("\nCache sweep:")
        total_freed += sweep_cache_dirs(root, args.apply)

    print("\n" + "=" * 60)
    verb = "Freed" if args.apply else "Would free"
    print(f"{verb}: {_fmt_bytes(total_freed)}")
    print("=" * 60)
    if not args.apply and total_freed > 0:
        print("Re-run with --apply to actually delete.")


if __name__ == "__main__":
    main()
