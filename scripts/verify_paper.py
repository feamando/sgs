"""Aggregate the paper-verification result files into the corrected paper numbers
and a per-reviewer-concern verdict. Reads results/ JSONs; computes nothing that
needs a GPU. Safe to run mid-way -- it reports what is present and what is still
pending, so the one-script runner can call it after every stage.

Verification matrix (arm x regime x seed), filenames:
  disambig_<arm>_<regime>_s<seed>.json         (arm: vsp|baseline|scrambled)
  legacy: disambig_vsp.json / disambig_baseline.json      == vsp|baseline, 40k, s0
          disambig_vsp_2k.json / disambig_baseline_2k.json == 2k, s0
          disambig_vsp_2k_s<N>.json                        == 2k, sN

The three reviewer concerns this settles:
  C1 (scrambled control): does grounded beat a scrambled bundle? If not, the
     failure is the signal, not the init pipeline.
  C2 (40k reproduction):  the headline -3.8 needs >=2 more seeds on the rescaled
     init; report mean +/- CI, not a single seed.
  C3 (low-context held-out): unchanged, reported for completeness.

  python scripts/verify_paper.py                 # all regimes/arms found
  python scripts/verify_paper.py --json           # machine-readable summary
"""

import argparse
import glob
import json
import re
from pathlib import Path
from statistics import mean, stdev

RESULTS = Path(__file__).resolve().parent.parent / "results"
ARMS = ["vsp", "baseline", "scrambled"]
REGIMES = ["2k", "40k"]

# t multipliers for a two-sided 95% CI, df = n-1
TMULT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
         7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def acc(path):
    return json.load(open(path))["accuracy"]


def parse_name(path):
    """Return (arm, regime, seed) or None. Handles legacy names."""
    b = Path(path).name
    m = re.match(r"disambig_(vsp|baseline|scrambled)_(2k|40k)_s(\d+)\.json$", b)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    m = re.match(r"disambig_(vsp|baseline)_2k\.json$", b)          # legacy 2k s0
    if m:
        return m.group(1), "2k", 0
    m = re.match(r"disambig_(vsp|baseline)_2k_s(\d+)\.json$", b)    # legacy 2k sN
    if m:
        return m.group(1), "2k", int(m.group(2))
    m = re.match(r"disambig_(vsp|baseline)\.json$", b)             # legacy 40k s0
    if m:
        return m.group(1), "40k", 0
    return None


def collect():
    """{(arm, regime): {seed: acc}} over everything in results/."""
    out = {}
    for p in glob.glob(str(RESULTS / "disambig_*.json")):
        parsed = parse_name(p)
        if not parsed:
            continue
        arm, regime, seed = parsed
        out.setdefault((arm, regime), {})[seed] = acc(p)
    return out


def ci95(vals):
    n = len(vals)
    if n == 0:
        return None
    m = mean(vals)
    if n == 1:
        return m, 0.0, (m, m), n
    sd = stdev(vals)
    sem = sd / n ** 0.5
    half = TMULT.get(n - 1, 2.0) * sem
    return m, sd, (m - half, m + half), n


def paired_deltas(data, arm_a, arm_b, regime):
    """Per-seed (arm_a - arm_b) for seeds present in BOTH."""
    a = data.get((arm_a, regime), {})
    b = data.get((arm_b, regime), {})
    seeds = sorted(set(a) & set(b))
    return [(s, a[s] - b[s]) for s in seeds]


def fmt_ci(c):
    if not c:
        return "no data"
    m, sd, (lo, hi), n = c
    if n == 1:
        return f"{m:+.3f} (n=1, no CI)"
    return f"mean {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={n}, sd {sd:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    data = collect()
    summary = {"present": {}, "verdicts": {}}

    # inventory
    for arm in ARMS:
        for reg in REGIMES:
            seeds = sorted(data.get((arm, reg), {}))
            if seeds:
                summary["present"][f"{arm}_{reg}"] = seeds

    if not args.json:
        print("=" * 66)
        print("PAPER VERIFICATION SUMMARY")
        print("=" * 66)
        print("\nInventory (arm/regime -> seeds present):")
        for arm in ARMS:
            for reg in REGIMES:
                seeds = sorted(data.get((arm, reg), {}))
                mark = "" if seeds else "   <-- MISSING"
                print(f"  {arm:<9} {reg:<4}: {seeds}{mark}")

    # ── C2: 40k reproduction (grounded - baseline), multi-seed ──
    d_vsp_40 = paired_deltas(data, "vsp", "baseline", "40k")
    c2 = ci95([d for _, d in d_vsp_40]) if d_vsp_40 else None
    summary["verdicts"]["C2_40k_reproduction"] = {
        "deltas": d_vsp_40,
        "mean": c2[0] if c2 else None, "ci": list(c2[2]) if c2 else None}

    # ── C1: scrambled control. grounded - scrambled at each regime ──
    c1 = {}
    for reg in REGIMES:
        d = paired_deltas(data, "vsp", "scrambled", reg)
        c1[reg] = {"deltas": d, "ci": ci95([x for _, x in d]) if d else None}
    summary["verdicts"]["C1_scrambled_control"] = {
        reg: {"deltas": c1[reg]["deltas"]} for reg in REGIMES}

    if not args.json:
        print("\n" + "-" * 66)
        print("C2  40k reproduction of the -3.8  (grounded - baseline, rescaled init)")
        print("-" * 66)
        if d_vsp_40:
            for s, d in d_vsp_40:
                print(f"    seed {s}: {d:+.3f}")
            print(f"    => {fmt_ci(c2)}")
            if c2 and c2[3] >= 2:
                lo, hi = c2[2]
                verdict = ("bounded: CI excludes a meaningful positive"
                           if hi < 0.01 else
                           "inconclusive: CI spans 0" if lo < 0 < hi else
                           "grounded WINS at 40k (unexpected -- revisit)")
                print(f"    VERDICT: {verdict}")
            else:
                print("    VERDICT: need >=2 seeds for a reproduction claim (have <2)")
        else:
            print("    no paired 40k seeds yet")

        print("\n" + "-" * 66)
        print("C1  Scrambled-bundle control  (grounded - scrambled)")
        print("    If ~0, the grounding SIGNAL is useless, not the init pipeline.")
        print("-" * 66)
        for reg in REGIMES:
            d = c1[reg]["deltas"]
            if d:
                cc = ci95([x for _, x in d])
                for s, dd in d:
                    print(f"    {reg} seed {s}: {dd:+.3f}")
                print(f"    => {reg}: {fmt_ci(cc)}")
            else:
                print(f"    {reg}: no grounded/scrambled paired seeds yet  <-- MISSING")

        # ── C3: low-context held-out (reads the rerank file if present) ──
        print("\n" + "-" * 66)
        print("C3  Low-context reranking (held-out lambda)  [unchanged from paper]")
        print("-" * 66)
        lc = RESULTS / "rerank_baseline_lowctx.json"
        if lc.exists():
            d = json.load(open(lc))
            a = {float(k): v for k, v in d["accuracy_by_lambda"].items()}
            base = a.get(0.0)
            print(f"    base(l=0) {base:.3f}, best {max(a.values()):.3f}, "
                  f"best-minus-base {max(a.values())-base:+.3f}")
        else:
            print("    rerank_baseline_lowctx.json not found")

        print("\n" + "=" * 66)
        print("PAPER NUMBER CHECK (paste-ready)")
        print("=" * 66)
        v40 = data.get(("vsp", "40k"), {})
        b40 = data.get(("baseline", "40k"), {})
        if 0 in v40 and 0 in b40:
            print(f"  40k s0: grounded {v40[0]:.3f} vs baseline {b40[0]:.3f} "
                  f"= {v40[0]-b40[0]:+.3f}  (paper's single-seed -3.8)")
        d2 = paired_deltas(data, "vsp", "baseline", "2k")
        if d2:
            print(f"  2k {len(d2)}-seed grounded-baseline: {fmt_ci(ci95([x for _,x in d2]))}")

    if args.json:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
