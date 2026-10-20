"""
Round 10 (SETUP_planck_20261020.md): close the Gemma gap. Hertz's reader trained on Brave pools with Gemma soft labels.

Round 9 shipped Hertz's reading reader (49.1% on fresh questions) but Gemma got 60.9%, the gap all on
Wikidata-style questions. Round 6 meant to train the reader on Brave-collected pools; a header misread
paused Brave and the pools came from ddgs and Wikipedia. Round 10 does it properly, on free credit only:

    collect   November's free Brave credit: round 6's training questions on Brave (the 2025 recent
              Wikidata items first, then the 2025 portal news), resumable across months; a credit stop
              never falls back to another engine
    teach     Gemma reads every Brave pool once (the reader prompt it uses on the benchmark): soft labels
    train     from round 7's Hertz reading reader (the shipped product), one pass of our pools as round 7
              plus the Brave pools, with Gemma's soft labels at weight 0.5 and early stopping
    judge     December's free credit: a NEW fresh benchmark (Wikidata 2026 items unused by rounds 1-9 and
              portal news after round 9's window), built once and committed before any reader runs on it

Pre-registered (section 4 of the doc), decided here and written to r10_product.json:
    product  x-hertz r10, int8 on CPU (4 threads), top 16 + template writer, FIXED before the run
    G        the product beats the shipped reader (round 9's product, the same configuration) on the new
             fresh questions by >= 5 points, exact McNemar p < 0.05
    GAP      reported: the share of the shipped reader's gap to Gemma that the product closes, on the new
             questions ((product - shipped) / (Gemma - shipped))
    Q, L, E  as round 9 (int8 vs GPU <= 2 points, <= 5 s on CPU over every question, worst-case profile)
    outcome  G passes (and Q, L, E) -> ship x-hertz r10; else keep round 9's reader
"""

from pathlib import Path

from .util import read_json, write_json

PRODUCT = "r10_x-hertz-r10_cpu-int8-top16"
SHIPPED = "r10_x-hertz-rc_cpu-int8-top16"
GPU, REF, FLOOR = "r10_x-hertz-r10_template", "r10_gemma_template", "r10_heuristic_template"
G_MARGIN = 0.05


def decide(results: Path, tag: str = "") -> dict:
    from .round7 import product_checks
    from .round9 import by_regime, paired
    g = paired(results, PRODUCT + tag, SHIPPED + tag)
    ref = paired(results, REF + tag, SHIPPED + tag)
    gp = "n" in g and g["acc_a"] - g["acc_b"] >= G_MARGIN and g["p"] < 0.05 and g["a_only"] > g["b_only"]
    gap = None
    if "n" in g and "n" in ref and ref["acc_a"] > ref["acc_b"]:
        gap = (g["acc_a"] - g["acc_b"]) / (ref["acc_a"] - ref["acc_b"])
    q = paired(results, PRODUCT + tag, GPU + tag)
    out = {"reader": "x-hertz", "reader_why": "fixed before the run: Hertz's reader trained on Brave pools with Gemma soft labels",
           "reader_checkpoint": "checkpoints/xreader_hertz_r10/reader.pt", "reader_quant": "int8", "reader_cands": 16,
           "writer": "template", "writer_why": "round 6: no small writer passes rule W", "run": PRODUCT, "tag": tag,
           "benchmark": "scripts/assets/planck3_tasks_fresh10.json",
           "rule_g": {"pass": gp, "margin": G_MARGIN, **g},
           "gap_closed": {"share": gap, "gemma_vs_shipped": ref},
           "rule_q": {"pass": "n" in q and q["acc_b"] - q["acc_a"] <= 0.02, **q},
           "vs_gemma": paired(results, PRODUCT + tag, REF + tag),
           "vs_heuristic": paired(results, PRODUCT + tag, FLOOR + tag),
           "by_regime": {x: by_regime(results, x + tag) for x in (FLOOR, SHIPPED, PRODUCT, GPU, REF)}}
    out.update(product_checks(results, PRODUCT, tag))
    s = results / (PRODUCT + tag) / "summary.json"
    if s.exists():
        t = read_json(s).get("timing_ms_per_task", {})
        ms = t.get("read_ms", 0) + t.get("write_ms", 0)
        out["rule_l"] = {"pass": ms <= 5000, "ms": ms, "measured_on": "every fresh question, CPU, 4 threads"}
    checks = all(out.get(k, {}).get("pass") for k in ("rule_q", "rule_l", "rule_e"))
    # rule_r keeps the chat factory's contract: a product ships only if rule_r.pass
    out["rule_r"] = {"pass": gp and checks, "acc": g.get("acc_a"), "why": "round 10: rule G and the product checks"}
    out["outcome"] = ("ship: x-hertz r10 beats the shipped reader on new fresh questions" if gp and checks else
                      "ship blocked by a product check (Q, L or E); keep round 9's reader" if gp else
                      "keep round 9's reader: Brave pools + Gemma soft labels did not beat it by 5 points")
    write_json(results / f"r10_product{tag}.json", out)
    print(f"[r10] rule G {'PASS' if gp else 'FAIL'}; gap closed {gap if gap is None else round(gap, 2)} -> {out['outcome']}")
    return out


def report_lines(results: Path) -> list[str]:
    p = results / "r10_product.json"
    if not p.exists():
        return []
    d = read_json(p)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    v = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines = ["## Round 10: Hertz's reader on Brave pools with Gemma soft labels (SETUP_planck_20261020.md)", ""]
    for k, label in (("rule_g", "G (vs the shipped reader, new fresh questions)"), ("rule_q", "Q (int8 CPU vs GPU)"),
                     ("vs_gemma", "vs Gemma"), ("vs_heuristic", "vs the heuristic")):
        x = d.get(k, {})
        if "n" in x:
            lines.append(f"- **{label}**{(' ' + v(x['pass'])) if 'pass' in x else ''}: {f3(x['acc_a'])} vs {f3(x['acc_b'])} "
                         f"({x['a_only']} vs {x['b_only']}, p {x['p']:.3g}, n {x['n']})")
    gc = d.get("gap_closed", {})
    if gc.get("share") is not None:
        lines.append(f"- **Gap to Gemma closed:** {gc['share']:.0%}")
    for k, label in (("rule_c1", "C1"), ("rule_l", "L"), ("rule_e", "E")):
        if k in d and "pass" in d[k]:
            lines.append(f"- **{label}** {v(d[k]['pass'])}: " + ", ".join(f"{a} {b}" for a, b in d[k].items() if a != "pass"))
    lines += [f"- **Outcome (pre-registered):** {d['outcome']}", ""]
    return lines


def add_cli(sub):
    p = sub.add_parser("r10", help="round 10: decide (pre-registered rules G, Q, L, E -> r10_product.json)")
    p.add_argument("stage", choices=["decide"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        decide(RESULTS, a.tag)
    p.set_defaults(fn=run)
