"""
Round 7 (SETUP_planck_20261014.md): reading taught as a skill, then our own pools; SGS only.

    stage A   x-<base>_rc   the round 6b reader trained on ~150k public reading-comprehension pools
                            (rcdata.py: SQuAD, NewsQA, TriviaQA, SearchQA, NaturalQuestions, SQuAD 2.0
                            unanswerable), 1 epoch, calibrated on their held-out questions
    stage B   x-<base>_r7   stage A, then round 6b's training on our pools (same points, same settings)

Pre-registered (section 5 of the doc), decided here and written to r7_product.json:
    product   x-planck r7 + template writer, FIXED: the phone-size model, no selection (round 6b's
              size rule picked the worse reader); Hertz is the scaling arm, reported
    R         the product reaches the midpoint between round 5's reader and Gemma (r6 runs) and beats
              round 5's reader, p < 0.05 (round6.rule_r)
    N         the rebuild gate: Planck stage A on held-out NewsQA pools beats its own lexical ranker by
              >= 10 points, exact McNemar p < 0.05 ("Planck can read news")
    H         reported: Hertz r7 beats Planck r7 on the benchmark by >= 5 points, p < 0.05
    outcome   R passes -> ship; else N fails -> rebuild Planck (round 8); else -> the shift between
              training pools and Brave (round 8: Brave pools + Gemma soft labels), no rebuild
"""

from pathlib import Path

from .util import read_json, read_jsonl, write_json

N_MARGIN = 0.10
H_MARGIN = 0.05


def _log(results: Path, name: str) -> dict:
    p = results / name / "train_log.json"
    return read_json(p) if p.exists() else {}


def rule_n(results: Path, base: str = "planck", tag: str = "", group: str = "newsqa") -> dict:
    t = _log(results, f"xreader_{base}_rc{tag}")
    g = (t.get("test_by_group") or {}).get(group)
    if not g:
        return {"pass": False, "why": f"no stage A test pools for {group}"}
    delta = g["choice_acc"] - g["ranker_acc"]
    ok = delta >= N_MARGIN and g.get("vs_ranker_p", 1.0) < 0.05 and g.get("only_choice_right", 0) > g.get("only_ranker_right", 0)
    return {"pass": ok, "group": group, "choice_acc": g["choice_acc"], "ranker_acc": g["ranker_acc"], "delta": delta,
            "p": g.get("vs_ranker_p"), "pools": g["pools_with_a_right_candidate"]}


def rule_h(results: Path, tag: str = "") -> dict:
    from .wikirace import mcnemar
    a, b = results / f"r7_x-hertz_template{tag}", results / f"r7_x-planck_template{tag}"
    if not ((a / "results.jsonl").exists() and (b / "results.jsonl").exists()):
        return {"flag": False, "why": "a run is missing"}
    ra = {r["task_id"]: r["shown_correct"] for r in read_jsonl(a / "results.jsonl")}
    rb = {r["task_id"]: r["shown_correct"] for r in read_jsonl(b / "results.jsonl")}
    ids = sorted(set(ra) & set(rb))
    m = mcnemar([ra[i] for i in ids], [rb[i] for i in ids])
    d = (sum(ra[i] for i in ids) - sum(rb[i] for i in ids)) / max(len(ids), 1)
    return {"flag": d >= H_MARGIN and m["p"] < 0.05, "delta": d, "p": m["p"], "n": len(ids)}


def rule_c1(summary: dict) -> dict:
    b = summary.get("bands", {})
    out, ok = {}, bool(summary.get("bands_monotone"))
    for band, bar in (("high", 0.90), ("good", 0.70)):
        n, prec = b.get(band, {}).get("n", 0), b.get(band, {}).get("precision")
        judged = n >= 20 and prec is not None
        out[band] = {"n": n, "precision": prec, "bar": bar, "judged": judged}
        if judged and prec < bar:
            ok = False
    return {"pass": ok, "monotone": bool(summary.get("bands_monotone")), **out}


def decide(results: Path, tag: str = "") -> dict:
    from .round6 import rule_r
    run0 = "r7_x-planck_template"
    r = rule_r(results, run0 + tag, tag)
    n, n_snip = rule_n(results, "planck", tag), rule_n(results, "planck", tag, "searchqa")
    out = {"reader": "x-planck", "reader_why": "fixed before the run: the phone-size model (no selection)",
           "reader_checkpoint": f"checkpoints/xreader_planck_r7{tag}/reader.pt", "writer": "template",
           "writer_why": "round 6: no small writer passes rule W", "run": "r7_x-planck_template", "tag": tag,
           "rule_r": r, "rule_n": n, "rule_n_searchqa": n_snip, "rule_n_hertz": rule_n(results, "hertz", tag),
           "rule_h": rule_h(results, tag)}
    s = results / (run0 + tag) / "summary.json"
    if s.exists():
        out["rule_c1"] = rule_c1(read_json(s))
    cpu = results / f"{run0}_cpu{tag}" / "summary.json"
    if cpu.exists():
        t = read_json(cpu)["timing_ms_per_task"]
        ms = t.get("read_ms", 0) + t.get("write_ms", 0)
        out["rule_l"] = {"pass": ms <= 5000, "ms": ms}
    con = results / f"{run0}_contrarian{tag}"
    if (con / "summary.json").exists() and s.exists():
        recs = read_jsonl(con / "results.jsonl")
        changed = [x for x in recs if x["changed_by_personal"]]
        flagged = sum(any(p.get("divergence") for p in x["parts"]) for x in changed)
        drop = read_json(s)["accuracy"] - read_json(con / "summary.json")["accuracy"]
        out["rule_e"] = {"pass": drop <= 0.10 and flagged == len(changed), "drop": drop, "changed": len(changed), "flagged": flagged}
    out["outcome"] = ("ship: the product passed R" if r.get("pass") else
                      "rebuild: Planck does not read news even after reading training (round 8 rebuilds the base)" if not n.get("pass") else
                      "shift: Planck reads news but not our Brave benchmark (round 8: Brave pools + Gemma soft labels; no rebuild)")
    write_json(results / f"r7_product{tag}.json", out)
    print(f"[r7] rule R {'PASS' if r.get('pass') else 'FAIL'}; rebuild gate N {'PASS' if n.get('pass') else 'FAIL'}; "
          f"-> {out['outcome']}")
    return out


def report_lines(results: Path) -> list[str]:
    p = results / "r7_product.json"
    if not p.exists():
        return []
    d = read_json(p)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    v = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines = ["## Round 7: reading taught as a skill (SETUP_planck_20261014.md)", ""]
    r = d.get("rule_r", {})
    if "acc" in r:
        lines.append(f"- **R** {v(r['pass'])}: x-planck r7 {f3(r['acc'])} vs midpoint {f3(r['midpoint'])}; round 5's reader "
                     f"{f3(r['vs_round5_reader'])}, p {r['p']:.3g}")
    for k, label in (("rule_n", "N (rebuild gate, NewsQA)"), ("rule_n_searchqa", "N on SearchQA (reported)"),
                     ("rule_n_hertz", "N for Hertz (reported)")):
        x = d.get(k, {})
        if "choice_acc" in x:
            lines.append(f"- **{label}** {v(x['pass'])}: choice {f3(x['choice_acc'])} vs ranker {f3(x['ranker_acc'])} "
                         f"(+{x['delta'] * 100:.1f} pts, p {x['p']:.3g}, {x['pools']} pools)")
    h = d.get("rule_h", {})
    if "delta" in h:
        lines.append(f"- **H** (reported): Hertz r7 minus Planck r7 {h['delta'] * 100:+.1f} pts, p {h['p']:.3g} -> "
                     f"{'pretraining breadth matters' if h['flag'] else 'no clear Hertz advantage'}")
    for k, label in (("rule_c1", "C1"), ("rule_l", "L"), ("rule_e", "E")):
        if k in d:
            lines.append(f"- **{label}** {v(d[k]['pass'])}: " + ", ".join(f"{a} {b}" for a, b in d[k].items() if a != "pass"))
    lines += [f"- **Outcome (pre-registered):** {d['outcome']}", ""]
    return lines


def add_cli(sub):
    p = sub.add_parser("r7", help="round 7: decide (pre-registered rules R, N, H, C1, L, E -> r7_product.json)")
    p.add_argument("stage", choices=["decide"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        decide(RESULTS, a.tag)
    p.set_defaults(fn=run)
