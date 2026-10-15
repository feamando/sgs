"""
Round 8 (SETUP_planck_20261015.md): the Planck base rebuilt on the right data, and Hertz on a phone proxy.

    base      Planck 1.5: Planck 1.3's architecture (100M: d_s 128, d_f 1000, 3 passes, 4 heads, 512 context)
              pretrained on Hertz 1.2's corpus (data/hertz12_data: FineWeb-Edu + 30% Wikipedia, 9.69B tokens,
              its 32K tokenizer), 1 epoch, plain AdamW. Same data as Hertz, Planck's size: size and data
              are no longer confounded
    reader    x-planck15 rc: round 7's stage A (the reading sets, 1 epoch) on Planck 1.5; no stage B
              (round 7: our pools added nothing for Planck and cost Hertz 4 points)
    mobile    x-hertz rc (round 7's exploratory 60.1%) on CPU, 4 threads: fp32 / int8 top 16 / int8 top 8

Pre-registered (section 5 of the doc), decided here and written to r8_product.json:
    product   x-planck15 rc + template writer, FIXED (the phone-size model)
    R         as round 6 (round6.rule_r): the midpoint between round 5's reader and Gemma, and p < 0.05
              vs round 5's reader
    N         Planck 1.5's reader beats its lexical ranker on held-out NewsQA by >= 10 points, p < 0.05
    M         Hertz on the phone proxy: an int8 configuration reads in <= 5 s per question on CPU
              (4 threads), loses <= 2 points vs its GPU run on the same questions (>= 200 judged), and
              needs <= 1.5 GB of weights (without the LM head, which a reader never uses)
    reported  size at equal data (Planck 1.5 vs Hertz), data at equal size (Planck 1.5 vs 1.3), the base's
              validation loss on Hertz's validation set (Hertz 1.2: 3.18)
    outcome   R passes -> ship x-planck15 rc; else N passes -> Planck 1.5 reads but not our benchmark
              (round 9: Brave pools + Gemma soft labels, no new base); else -> a 100M SGS base does not
              read even on Hertz's data: size is the lever (Hertz on the phone if M passes)
"""

import platform
from pathlib import Path

from .util import read_json, read_jsonl, write_json

RUN0 = "r8_x-planck15-rc_template"
HERTZ_GPU = "r7_x-hertz-rc_template"
MOBILE = ("cpu-fp32-top16", "cpu-int8-top16", "cpu-int8-top8")
M_MS, M_DROP, M_MB, M_JUDGED = 5000, 0.02, 1536, 200   # 1.5 GB: weights an 8 GB phone can hold with headroom


def _recs(results: Path, run: str) -> dict:
    p = results / run / "results.jsonl"
    return {r["task_id"]: r["shown_correct"] for r in read_jsonl(p)} if p.exists() else {}


def paired(results: Path, a: str, b: str) -> dict:
    from .wikirace import mcnemar
    ra, rb = _recs(results, a), _recs(results, b)
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return {"why": "a run is missing"}
    m = mcnemar([ra[i] for i in ids], [rb[i] for i in ids])
    return {"a": a, "b": b, "n": len(ids), "acc_a": sum(ra[i] for i in ids) / len(ids),
            "acc_b": sum(rb[i] for i in ids) / len(ids), "a_only": m["a_only"], "b_only": m["b_only"], "p": m["p"]}


def mobile_config(results: Path, cfg: str, tag: str = "") -> dict:
    run = f"r8_x-hertz-rc_{cfg}{tag}"
    s = results / run / "summary.json"
    if not s.exists():
        return {"run": run, "why": "not run"}
    summ = read_json(s)
    t = summ.get("timing_ms_per_task", {})
    ms = t.get("read_ms", 0) + t.get("write_ms", 0)
    rc = summ.get("reader_config") or {}
    pr = paired(results, run, HERTZ_GPU + tag)
    drop = pr["acc_b"] - pr["acc_a"] if "n" in pr else None
    judged = pr.get("n", 0) >= M_JUDGED
    ok = (rc.get("quant") == "int8" and ms <= M_MS and (rc.get("size_mb") or 1e9) <= M_MB
          and judged and drop is not None and drop <= M_DROP)
    return {"run": run, "pass": ok, "judged": judged, "ms": ms, "size_mb": rc.get("size_mb"), "quant": rc.get("quant"),
            "max_cands": rc.get("max_cands"), "threads": rc.get("threads"), "n": pr.get("n", 0),
            "acc": pr.get("acc_a"), "acc_gpu_same_questions": pr.get("acc_b"), "drop": drop}


def rule_m(results: Path, tag: str = "") -> dict:
    cfgs = {c: mobile_config(results, c, tag) for c in MOBILE}
    best = [c for c, v in cfgs.items() if v.get("pass")]
    return {"pass": bool(best), "passing": best, "configs": cfgs, "cpu": platform.processor() or platform.machine(),
            "bars": {"ms": M_MS, "drop": M_DROP, "size_mb": M_MB, "judged_at": M_JUDGED}}


def base_metrics(ckpts: Path, tag: str = "") -> dict:
    p = ckpts / f"planck15{tag}" / "final_metrics.json"
    return read_json(p) if p.exists() else {"why": "Planck 1.5 not trained yet"}


def decide(results: Path, tag: str = "", ckpts: Path | None = None) -> dict:
    from .round6 import rule_r
    from .round7 import product_checks, rule_n
    ckpts = ckpts or results.parents[1] / "checkpoints"
    r = rule_r(results, RUN0 + tag, tag)
    n = rule_n(results, "planck", tag, "newsqa", "p15rc")
    out = {"reader": "x-planck", "reader_why": "fixed before the run: Planck 1.5, the phone-size model",
           "reader_checkpoint": f"checkpoints/xreader_planck_p15rc{tag}/reader.pt", "writer": "template",
           "writer_why": "round 6: no small writer passes rule W", "run": RUN0, "tag": tag,
           "base": base_metrics(ckpts, tag), "rule_r": r, "rule_n": n,
           "rule_n_searchqa": rule_n(results, "planck", tag, "searchqa", "p15rc"),
           "size_at_equal_data": {"benchmark": paired(results, RUN0 + tag, HERTZ_GPU + tag),
                                  "newsqa_hertz": rule_n(results, "hertz", tag, "newsqa", "rc")},
           "data_at_equal_size": {"benchmark": paired(results, RUN0 + tag, f"r7_x-planck-rc_template{tag}"),
                                  "newsqa_planck13": rule_n(results, "planck", tag, "newsqa", "rc")},
           "rule_m": rule_m(results, tag)}
    out.update(product_checks(results, RUN0, tag))
    out["outcome"] = ("pending: Planck 1.5 is not evaluated yet (this decision covers the mobile check only)"
                      if not (results / (RUN0 + tag) / "summary.json").exists() else
                      "ship: Planck 1.5's reader passed R (phone-size)" if r.get("pass") else
                      "shift: Planck 1.5 reads news but not our Brave benchmark (round 9: Brave pools + Gemma soft labels; no new base)"
                      if n.get("pass") else
                      "size: a 100M SGS base does not read even on Hertz's data; the reader path is Hertz "
                      + ("(M passes: confirm it on fresh questions)" if out["rule_m"]["pass"] else "(M fails: a mid-size base or a faster Hertz)"))
    write_json(results / f"r8_product{tag}.json", out)
    print(f"[r8] rule R {'PASS' if r.get('pass') else 'FAIL'}; N {'PASS' if n.get('pass') else 'FAIL'}; "
          f"Hertz mobile M {'PASS' if out['rule_m']['pass'] else 'FAIL'} -> {out['outcome']}")
    return out


def report_lines(results: Path) -> list[str]:
    p = results / "r8_product.json"
    if not p.exists():
        return []
    d = read_json(p)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    v = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines = ["## Round 8: Planck 1.5 (Planck's size, Hertz's data) and Hertz on a phone proxy (SETUP_planck_20261015.md)", ""]
    b = d.get("base", {})
    if "val_loss" in b:
        lines.append(f"- **Planck 1.5 base:** {b['params'] / 1e6:.0f}M params, {b['tokens_seen'] / 1e9:.2f}B tokens, "
                     f"val loss {b['val_loss']:.4f} / ppl {b['val_ppl']:.1f} on Hertz's validation set (Hertz 1.2: 3.1832 / 24.1)")
    r = d.get("rule_r", {})
    if "acc" in r:
        lines.append(f"- **R** {v(r['pass'])}: x-planck15 rc {f3(r['acc'])} vs midpoint {f3(r['midpoint'])}; round 5's reader "
                     f"{f3(r['vs_round5_reader'])}, p {r['p']:.3g}")
    for k, label in (("rule_n", "N (NewsQA)"), ("rule_n_searchqa", "N on SearchQA (reported)")):
        x = d.get(k, {})
        if "choice_acc" in x:
            lines.append(f"- **{label}** {v(x['pass'])}: choice {f3(x['choice_acc'])} vs ranker {f3(x['ranker_acc'])} "
                         f"({x['delta'] * 100:+.1f} pts, p {x['p']:.3g}, {x['pools']} pools)")
    for k, label in (("size_at_equal_data", "Size at equal data (vs Hertz rc)"), ("data_at_equal_size", "Data at equal size (vs Planck 1.3 rc)")):
        x = (d.get(k) or {}).get("benchmark", {})
        if "n" in x:
            lines.append(f"- **{label}:** {f3(x['acc_a'])} vs {f3(x['acc_b'])} on {x['n']} questions "
                         f"({x['a_only']} vs {x['b_only']} only-right, p {x['p']:.3g})")
    m = d.get("rule_m", {})
    if m:
        lines.append(f"- **M (Hertz on the phone proxy, {m.get('cpu', '')})** {v(m.get('pass'))}:")
        for c, x in m.get("configs", {}).items():
            if "ms" in x:
                lines.append(f"  - {c}: {x['ms']:.0f} ms per question, {x['size_mb']} MB, right {f3(x['acc'])} vs GPU "
                             f"{f3(x['acc_gpu_same_questions'])} on {x['n']} questions"
                             + ("" if x["judged"] else " (not judged: too few questions)"))
    for k, label in (("rule_c1", "C1"), ("rule_l", "L"), ("rule_e", "E")):
        if k in d:
            lines.append(f"- **{label}** {v(d[k]['pass'])}: " + ", ".join(f"{a} {b}" for a, b in d[k].items() if a != "pass"))
    lines += [f"- **Outcome (pre-registered):** {d['outcome']}", ""]
    return lines


def add_cli(sub):
    p = sub.add_parser("r8", help="round 8: decide (pre-registered rules R, N, M, C1, L, E -> r8_product.json)")
    p.add_argument("stage", choices=["decide"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        decide(RESULTS, a.tag)
    p.set_defaults(fn=run)
