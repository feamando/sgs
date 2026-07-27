"""Aggregate multi-seed disambiguation results into a per-regime mean delta with
error bars and an across-seed significance test. This is the settle-or-kill
readout for the VSP efficiency question: one lucky seed (the +5.7 @2k) is noise;
the MEAN delta across seeds with its spread is the actual signal.

For each compute regime (2k, 5k, 40k) it pairs vsp_<regime>_s<N>.json with
baseline_<regime>_s<N>.json, computes per-seed delta = acc_vsp - acc_base, then
reports mean, std, sem, a 95% CI, and a paired t-test of the per-seed deltas
against 0 (does grounding help ON AVERAGE at this regime?).

Seed 0 files have no _s0 suffix (disambig_vsp_2k.json etc.); handled below.

  python scripts/aggregate_disambig_seeds.py                # all regimes found
  python scripts/aggregate_disambig_seeds.py --regimes 2k 40k
"""

import argparse
import glob
import json
import re
from pathlib import Path
from statistics import mean, stdev

RESULTS = Path(__file__).resolve().parent.parent / "results"


def acc(path):
    return json.load(open(path))["accuracy"]


def seed_of(path, regime):
    """seed 0 = no suffix (…_2k.json); else …_2k_s<N>.json."""
    m = re.search(rf"_{regime}_s(\d+)\.json$", path)
    if m:
        return int(m.group(1))
    return 0 if path.endswith(f"_{regime}.json") else None


def collect(regime):
    """Return {seed: (vsp_acc, base_acc)} for every seed with BOTH arms present."""
    vsp, base = {}, {}
    for p in glob.glob(str(RESULTS / f"disambig_vsp_{regime}*.json")):
        s = seed_of(p, regime)
        if s is not None:
            vsp[s] = acc(p)
    for p in glob.glob(str(RESULTS / f"disambig_baseline_{regime}*.json")):
        s = seed_of(p, regime)
        if s is not None:
            base[s] = acc(p)
    return {s: (vsp[s], base[s]) for s in sorted(vsp) if s in base}


def t_ci(deltas):
    """Paired-style: per-seed deltas vs 0. Returns (mean, sem, t, df, ci95)."""
    n = len(deltas)
    m = mean(deltas)
    if n < 2:
        return m, float("nan"), float("nan"), 0, (float("nan"), float("nan"))
    sd = stdev(deltas)
    sem = sd / n ** 0.5
    t = m / sem if sem else float("inf")
    # 95% CI with a t-multiplier table (df 1..10 then ~1.96); good enough for a readout
    tmult = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
             7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}.get(n - 1, 2.0)
    half = tmult * sem
    return m, sem, t, n - 1, (m - half, m + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["2k", "5k", "40k"])
    args = ap.parse_args()

    print(f"{'regime':<8}{'seeds':<8}{'mean Δ':>9}{'std':>8}{'sem':>8}"
          f"{'t':>7}{'95% CI':>18}   per-seed deltas")
    summary = {}
    for reg in args.regimes:
        data = collect(reg)
        if not data:
            continue
        deltas = [v - b for (v, b) in data.values()]
        m, sem, t, df, ci = t_ci(deltas)
        summary[reg] = (m, ci, len(deltas))
        ds = " ".join(f"{d:+.3f}" for d in deltas)
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
        print(f"{reg:<8}{len(deltas):<8}{m:>+9.3f}{(stdev(deltas) if len(deltas)>1 else 0):>8.3f}"
              f"{sem:>8.3f}{t:>7.2f}{ci_s:>18}   {ds}")

    # verdict hint if we have both a low and the 40k anchor
    if "2k" in summary and "40k" in summary:
        (m2, ci2, _), (m40, ci40, _) = summary["2k"], summary["40k"]
        print()
        print(f"2k mean Δ {m2:+.3f} {list(map(lambda x: round(x,3), ci2))}  "
              f"vs 40k mean Δ {m40:+.3f} {list(map(lambda x: round(x,3), ci40))}")
        overlap = not (ci2[0] > ci40[1] or ci40[0] > ci2[1])
        if not overlap and m2 > m40:
            print("VERDICT: 2k > 40k, CIs disjoint -> efficiency effect holds.")
        elif ci2[0] > 0:
            print("VERDICT: 2k CI excludes 0 -> grounding helps at low compute.")
        else:
            print("VERDICT: 2k CI includes 0 (and/or overlaps 40k) -> "
                  "NOT distinguishable from noise; embedding-init neutral.")


if __name__ == "__main__":
    main()
