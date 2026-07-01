"""Quick analysis of a Satz conversation log (JSONL).

Each line is one generation turn written by satz/app.py's ConversationLogger.
This prints a summary; for deeper work load the JSONL with pandas:

    import pandas as pd
    df = pd.read_json("runs/satz_conversations.jsonl", lines=True)

Run:
    python -m satz.analyze_log --log runs/satz_conversations.jsonl
    python -m satz.analyze_log --log runs/satz_conversations.jsonl --model hertz --show 5
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [warn] skipping malformed line {i}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Summarize a Satz conversation log")
    p.add_argument("--log", default="runs/satz_conversations.jsonl")
    p.add_argument("--model", default=None, help="Filter to one model (planck/hertz)")
    p.add_argument("--show", type=int, default=0, help="Print N most recent turns in full")
    args = p.parse_args()

    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"log not found: {path}")

    rows = load(path)
    if args.model:
        rows = [r for r in rows if r.get("model") == args.model]
    if not rows:
        raise SystemExit("no turns to analyze (empty or filtered out)")

    n = len(rows)
    by_model = Counter(r.get("model", "?") for r in rows)
    sessions = defaultdict(int)
    for r in rows:
        sessions[r.get("session_id") or "no-session"] += 1

    def _avg(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"Satz conversation log: {path}")
    print(f"  turns:            {n}")
    print(f"  by model:         {dict(by_model)}")
    print(f"  sessions:         {len(sessions)} "
          f"(avg {n/len(sessions):.1f} turns/session)")
    print(f"  avg prompt tok:   {_avg('prompt_tokens'):.1f}")
    print(f"  avg gen tok:      {_avg('generated_tokens'):.1f}")
    print(f"  avg tok/s:        {_avg('tokens_per_sec'):.1f}")
    print(f"  avg gen seconds:  {_avg('gen_seconds'):.2f}")

    if args.show:
        print(f"\n── {min(args.show, n)} most recent turns ──")
        for r in rows[-args.show:]:
            print(f"\n[{r.get('ts','?')}] model={r.get('model')} "
                  f"gen_tok={r.get('generated_tokens')} k={r.get('k')}")
            print(f"  PROMPT: {r.get('prompt','')[:300]}")
            print(f"  OUTPUT: {r.get('generated_text','')[:500]}")


if __name__ == "__main__":
    main()
