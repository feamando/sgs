"""
Round 9 (SETUP_planck_20261019.md): Hertz's reading reader confirmed on fresh questions, as it would run on a phone.

    benchmark  planck3_tasks_fresh9.json (freshbench.py): 2026 Wikidata items and post-2026-10-08 news
               that no earlier round used, built once and committed before any reader runs on it
    product    x-hertz rc (round 7's stage A reader) on CPU, 4 threads, int8 weights, top 16 candidates,
               template writer: the configuration that passed rule M in round 8. FIXED before the run
    arms       the heuristic (no model), round 5's Planck reader (ships today), Gemma (reference),
               x-hertz rc on the GPU (the quantization check), x-planck15 rc (Planck track, reported)

Pre-registered (section 4 of the doc), decided here and written to r9_product.json:
    R9   the product reaches the midpoint between round 5's reader and Gemma ON THE FRESH QUESTIONS, and
         beats round 5's reader there (exact McNemar, p < 0.05): round 6's rule R, new questions
    Q    int8 on CPU loses <= 2 points vs the same reader on the GPU, on the fresh questions
    L    reading + writing <= 5 s per question, measured on CPU over every fresh question
    E    worst-case personal profile: <= 10 points drop, every changed answer flagged
    C1   reported as before (9-10 >= 90%, 7-8 >= 70%), and RECALIBRATED: band thresholds fitted on the
         product's round 8 answers to the 243 old questions, judged on the fresh ones
    outcome  R9 passes -> ship the product (serve picks it up); else -> no ship: round 10 trains Hertz's
             reader on Brave pools with Gemma soft labels (the lever not yet tried)
"""

from pathlib import Path

from .util import read_json, read_jsonl, write_json

PRODUCT = "r9_x-hertz-rc_cpu-int8-top16"
BASE, REF, FLOOR, GPU = "r9_planck_template", "r9_gemma_template", "r9_heuristic_template", "r9_x-hertz-rc_template"
PLANCK15 = "r9_x-planck15-rc_template"
OLD_PRODUCT = "r8_x-hertz-rc_cpu-int8-top16"       # the same configuration on the 243 old questions
Q_DROP, L_MS, CAL_N = 0.02, 5000, 10


def _recs(results: Path, run: str) -> dict:
    p = results / run / "results.jsonl"
    return {r["task_id"]: r for r in read_jsonl(p)} if p.exists() else {}


def paired(results: Path, a: str, b: str) -> dict:
    from .wikirace import mcnemar
    ra, rb = _recs(results, a), _recs(results, b)
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return {"why": f"{a if not ra else b} missing"}
    m = mcnemar([ra[i]["shown_correct"] for i in ids], [rb[i]["shown_correct"] for i in ids])
    return {"a": a, "b": b, "n": len(ids), "acc_a": sum(ra[i]["shown_correct"] for i in ids) / len(ids),
            "acc_b": sum(rb[i]["shown_correct"] for i in ids) / len(ids), "a_only": m["a_only"], "b_only": m["b_only"], "p": m["p"]}


def rule_r9(results: Path, tag: str = "") -> dict:
    from .wikirace import mcnemar
    a, b, g = (_recs(results, x + tag) for x in (PRODUCT, BASE, REF))
    ids = sorted(set(a) & set(b) & set(g))
    if not ids:
        return {"pass": False, "why": "a run is missing"}
    acc = lambda d: sum(d[i]["shown_correct"] for i in ids) / len(ids)  # noqa: E731
    mid = (acc(b) + acc(g)) / 2
    m = mcnemar([a[i]["shown_correct"] for i in ids], [b[i]["shown_correct"] for i in ids])
    ok = acc(a) >= mid and acc(a) > acc(b) and m["p"] < 0.05
    return {"pass": ok, "acc": acc(a), "midpoint": mid, "vs_round5_reader": acc(b), "gemma": acc(g),
            "a_only": m["a_only"], "b_only": m["b_only"], "p": m["p"], "n": len(ids)}


def by_regime(results: Path, run: str) -> dict:
    out = {}
    for r in _recs(results, run).values():
        k = out.setdefault(r["regime"], [0, 0])
        k[0] += 1
        k[1] += r["shown_correct"]
    return {g: {"n": n, "acc": c / n} for g, (n, c) in sorted(out.items())}


def fit_thresholds(recs: list[dict]) -> dict:
    """The lowest confidence t whose answers at >= t are right >= 90% (high) and >= 70% (good), at least
    CAL_N answers each; None when no threshold reaches the bar (then that band is never shown)."""
    scored = [(r["confidence"], bool(r["shown_correct"])) for r in recs if r.get("confidence") is not None]
    out = {}
    for band, bar in (("high", 0.90), ("good", 0.70)):
        best = None
        for t in sorted({c for c, _ in scored}, reverse=True):
            sub = [ok for c, ok in scored if c >= t]
            if len(sub) >= CAL_N and sum(sub) / len(sub) >= bar:
                best = t
        out[band] = best
    return out


def rule_c1_recal(results: Path, tag: str = "") -> dict:
    old = list(_recs(results, OLD_PRODUCT).values())
    new = list(_recs(results, PRODUCT + tag).values())
    if not old or not new:
        return {"why": "a run is missing"}
    th = fit_thresholds(old)
    hi, gd = th["high"], th["good"]

    def prec(lo, up):
        sub = [r["shown_correct"] for r in new if r.get("confidence") is not None
               and (lo is not None and r["confidence"] >= lo) and (up is None or r["confidence"] < up)]
        return {"n": len(sub), "precision": sum(sub) / len(sub) if sub else None}
    high = prec(hi, None) if hi is not None else {"n": 0, "precision": None}
    good = prec(gd, hi) if gd is not None else {"n": 0, "precision": None}
    ok = all(b["precision"] is None or b["n"] < 20 or b["precision"] >= bar for b, bar in ((high, 0.90), (good, 0.70)))
    return {"pass": ok, "thresholds_fit_on": f"{OLD_PRODUCT} (243 old questions)", "high_from": hi, "good_from": gd,
            "high": high, "good": good}


def decide(results: Path, tag: str = "") -> dict:
    from .round7 import product_checks
    r = rule_r9(results, tag)
    q = paired(results, PRODUCT + tag, GPU + tag)
    out = {"reader": "x-hertz", "reader_why": "fixed before the run: round 7's Hertz reading reader, int8 on CPU (passed round 8's M)",
           "reader_checkpoint": "checkpoints/xreader_hertz_rc/reader.pt", "reader_quant": "int8", "reader_cands": 16,
           "writer": "template", "writer_why": "round 6: no small writer passes rule W", "run": PRODUCT, "tag": tag,
           "benchmark": "scripts/assets/planck3_tasks_fresh9.json", "rule_r": r,
           "rule_q": {"pass": "n" in q and q["acc_b"] - q["acc_a"] <= Q_DROP, **q},
           "vs_heuristic": paired(results, PRODUCT + tag, FLOOR + tag),
           "vs_gemma": paired(results, PRODUCT + tag, REF + tag),
           "planck15": paired(results, PLANCK15 + tag, BASE + tag),
           "by_regime": {x: by_regime(results, x + tag) for x in (FLOOR, BASE, GPU, PRODUCT, REF, PLANCK15)},
           "rule_c1_recal": rule_c1_recal(results, tag)}
    out.update(product_checks(results, PRODUCT, tag))
    s = results / (PRODUCT + tag) / "summary.json"
    if s.exists():
        t = read_json(s).get("timing_ms_per_task", {})
        ms = t.get("read_ms", 0) + t.get("write_ms", 0)
        out["rule_l"] = {"pass": ms <= L_MS, "ms": ms, "measured_on": "every fresh question, CPU, 4 threads"}
    out["outcome"] = ("ship: Hertz's reader (int8, CPU) passed R9 on fresh questions" if r.get("pass") else
                      "no ship: round 10 trains Hertz's reader on Brave-collected pools with Gemma soft labels")
    write_json(results / f"r9_product{tag}.json", out)
    print(f"[r9] rule R9 {'PASS' if r.get('pass') else 'FAIL'}; Q {'PASS' if out['rule_q']['pass'] else 'FAIL'} -> {out['outcome']}")
    return out


def report_lines(results: Path) -> list[str]:
    p = results / "r9_product.json"
    if not p.exists():
        return []
    d = read_json(p)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    v = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines = ["## Round 9: Hertz's reader on fresh questions, int8 on CPU (SETUP_planck_20261019.md)", ""]
    r = d.get("rule_r", {})
    if "acc" in r:
        lines.append(f"- **R9** {v(r['pass'])}: product {f3(r['acc'])} vs midpoint {f3(r['midpoint'])} (round 5's reader "
                     f"{f3(r['vs_round5_reader'])}, Gemma {f3(r['gemma'])}); {r['a_only']} vs {r['b_only']} only-right, "
                     f"p {r['p']:.3g}, {r['n']} fresh questions")
    for k, label in (("rule_q", "Q (int8 CPU vs GPU)"), ("vs_heuristic", "vs the heuristic"), ("vs_gemma", "vs Gemma"),
                     ("planck15", "Planck 1.5 rc vs round 5's reader (reported)")):
        x = d.get(k, {})
        if "n" in x:
            lines.append(f"- **{label}**{(' ' + v(x['pass'])) if 'pass' in x else ''}: {f3(x['acc_a'])} vs {f3(x['acc_b'])} "
                         f"({x['a_only']} vs {x['b_only']}, p {x['p']:.3g})")
    for k, label in (("rule_c1", "C1"), ("rule_c1_recal", "C1 recalibrated"), ("rule_l", "L"), ("rule_e", "E")):
        if k in d and "pass" in d[k]:
            lines.append(f"- **{label}** {v(d[k]['pass'])}: " + ", ".join(f"{a} {b}" for a, b in d[k].items() if a != "pass"))
    for run, reg in d.get("by_regime", {}).items():
        if reg:
            lines.append(f"- {run}: " + "; ".join(f"{g} {x['acc']:.3f} (n={x['n']})" for g, x in reg.items()))
    lines += [f"- **Outcome (pre-registered):** {d['outcome']}", ""]
    return lines


def add_cli(sub):
    p = sub.add_parser("r9", help="round 9: decide (pre-registered rules R9, Q, L, E, C1 -> r9_product.json)")
    p.add_argument("stage", choices=["decide"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        decide(RESULTS, a.tag)
    p.set_defaults(fn=run)
