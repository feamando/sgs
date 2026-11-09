"""
Round 11 (SETUP_planck_20261109.md): where is the Gemma gap? Evidence view vs the reader itself.

Round 10 closed none of the gap (shipped reader 48.6%, Gemma 60.9% on fresh10). Its pre-registered next
question: is the gap SIZE (the reader) or Gemma's EVIDENCE VIEW? Gemma and the shipped reader do not see
the same evidence today:

    candidates  Gemma reads the top 20 of each pool, the shipped reader the top 16
    pages       pages are opened only when the snippets weigh in under 7/10, and that weighing uses the
                reader's own r, so a more confident reader opens fewer pages (Gemma 1.15 fetches per
                question on fresh10, the shipped reader 1.50): the two arms read different pages
    context     NOT a Gemma advantage: Gemma sees 200 characters per candidate, the small reader 240
                (all of it; candidates carry 240), so "longer contexts" is ruled out before any run

Phase 1, DIAGNOSE (no searches: round 10's Brave answers are cached; a zero-search ledger enforces it):
the same fresh10 questions, re-read with the view and the pages varied one at a time.

    D_ship        shipped reader (x-hertz rc, int8 CPU, top 16)     replication of round 10 + pool dump
    D_ship32      the same, reading the top 32 (untrained beyond 16)
    D_shipP       the same, always reading the 2 pages               (open_below 11)
    D_ship32P     both
    D_gem         Gemma, top 20 (replication + pool dump)
    D_gem16       Gemma, top 16                                       its view advantage
    D_gem16P      Gemma, top 16, always the 2 pages                   = D_shipP's evidence exactly
    D_old         shipped reader on round 8's 243 questions           latency: ms per candidate read

Pre-registered (section 4 of the doc), decided by `r11 diagnose` -> r11_plan.json BEFORE fresh11 exists:
    valid     D_ship gives round 10's answer on >= 90% of questions (else: not a replay, stop)
    config    the best of D_ship32 / D_shipP / D_ship32P beats D_ship by >= 3 points with more only-right
              than only-wrong -> phase 2 judges that configuration (no training) on fresh11
    view      else, Gemma's view is worth >= 3 points (D_gem - D_gem16) and the top 32 reach >= 3 points
              more right answers than the top 16 -> round 12: train the reader on a wider view (not built)
    size      else, at identical evidence Gemma reads >= 8 points better (D_gem16P - D_shipP) -> the gap
              is the reader: round 12 = the owner's call on a larger SGS base (not built)
    split     else -> no single lever; report and stop

Phase 2, JUDGE (only for "config"): a NEW fresh benchmark (fresh11: Wikidata 2026 items unused by rounds
1-10 + portal news after 2026-11-03), committed before any reader runs. Rules -> r11_product.json:
    G11   the configuration beats the shipped reader by >= 3 points, exact McNemar p < 0.05
    L     <= 5 s per question on CPU, read + write as rounds 8-10 (page fetch time reported), E as before, C1 reported
"""

import gzip
import json
from pathlib import Path

from .util import read_json, write_json

D = {"ship": "r11d_x-hertz-rc_cpu-int8-top16", "ship32": "r11d_x-hertz-rc_cpu-int8-top32",
     "shipP": "r11d_x-hertz-rc_cpu-int8-top16_pages", "ship32P": "r11d_x-hertz-rc_cpu-int8-top32_pages",
     "gem": "r11d_gemma_template", "gem16": "r11d_gemma-top16_template", "gem16P": "r11d_gemma-top16_template_pages",
     "old": "r11d_x-hertz-rc_cpu-int8-top16_old"}
R10_SHIPPED, R8_OLD = "r10_x-hertz-rc_cpu-int8-top16", "r8_x-hertz-rc_cpu-int8-top16"
CONFIGS = {"ship32": {"reader_cands": 32, "open_below": 7.0}, "shipP": {"reader_cands": 16, "open_below": 11.0},
           "ship32P": {"reader_cands": 32, "open_below": 11.0}}
CONFIG_MARGIN, VIEW_MARGIN, REACH_MARGIN, SIZE_MARGIN, REPLAY_MIN = 0.03, 0.03, 0.03, 0.08, 0.90
G_MARGIN, L_MS = 0.03, 5000
PRODUCT, SHIPPED = "r11_product_cpu-int8", "r11_x-hertz-rc_cpu-int8-top16"
REF, FLOOR = "r11_gemma_template", "r11_heuristic_template"


def _recs(results: Path, run: str) -> dict:
    from .round9 import _recs as r
    return r(results, run)


def _acc(results: Path, run: str) -> float | None:
    r = _recs(results, run)
    return sum(x["shown_correct"] for x in r.values()) / len(r) if r else None


def replay_agreement(results: Path, a: str, b: str) -> dict:
    """Share of questions where two runs show the same value (a replay of the same reader on cached evidence)."""
    ra, rb = _recs(results, a), _recs(results, b)
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return {"why": "a run is missing"}
    same = sum(str(ra[i]["value"]) == str(rb[i]["value"]) for i in ids)
    return {"n": len(ids), "same_value": same / len(ids)}


def reach(dump: Path, ks=(16, 20, 32)) -> dict:
    """Oracle: the share of questions where a right candidate sits in the top k of ANY pool the run read
    (the best a reader could do with that view). From a run's pools.jsonl.gz."""
    from .metrics import is_correct
    if not dump.exists():
        return {"why": f"{dump} missing"}
    rows = [json.loads(x) for x in gzip.open(dump, "rt", encoding="utf-8")]
    out = {"n": len(rows)}
    for k in ks:
        out[f"top{k}"] = sum(any(is_correct(c["value"], r["gold"], r["answer_type"]) for p in r["pools"] for c in p["cands"][:k])
                             for r in rows) / max(len(rows), 1)
    return out


def ms_per_cand(results: Path, run: str) -> dict:
    r = list(_recs(results, run).values())
    rc = [(x["timing"]["read_ms"], sum(p.get("read_cands") or 0 for p in x["parts"])) for x in r]
    rc = [(ms, n) for ms, n in rc if n]
    if not rc:
        return {"why": f"{run}: no read_cands recorded"}
    return {"n": len(rc), "cands_per_q": sum(n for _, n in rc) / len(rc), "read_ms_per_q": sum(ms for ms, _ in rc) / len(rc),
            "ms_per_cand": sum(ms for ms, _ in rc) / sum(n for _, n in rc)}


def diagnose(results: Path, tag: str = "") -> dict:
    from .round9 import paired
    t = lambda k: D[k] + tag  # noqa: E731
    acc = {k: _acc(results, t(k)) for k in D}
    rep = replay_agreement(results, t("ship"), R10_SHIPPED)
    valid = rep.get("same_value", 0) >= REPLAY_MIN or bool(tag)       # quick runs sample 12: reported, not gated
    vs_ship = {k: paired(results, t(k), t("ship")) for k in CONFIGS}
    gem_view = paired(results, t("gem"), t("gem16"))
    equal_ev = paired(results, t("gem16P"), t("shipP"))
    rch = reach(results / t("ship32P") / "pools.jsonl.gz")
    gap = paired(results, t("gem"), t("ship"))
    d = lambda x: (x["acc_a"] - x["acc_b"]) if "n" in x else None  # noqa: E731
    best = max((k for k in CONFIGS if "n" in vs_ship[k]), key=lambda k: (d(vs_ship[k]), -len(k)), default=None)
    cfg_ok = best is not None and d(vs_ship[best]) >= CONFIG_MARGIN and vs_ship[best]["a_only"] > vs_ship[best]["b_only"]
    view_ok = (d(gem_view) or 0) >= VIEW_MARGIN and "top32" in rch and rch["top32"] - rch["top16"] >= REACH_MARGIN
    size_ok = (d(equal_ev) or 0) >= SIZE_MARGIN
    if not valid:
        outcome, nxt = "invalid", "the replay does not reproduce round 10 (pages changed?): nothing is decided"
    elif cfg_ok:
        outcome, nxt = "config", f"phase 2 judges {best} ({CONFIGS[best]}) on fresh11, no training"
    elif view_ok:
        outcome, nxt = "view", "round 12 (not built): train the reader on a wider view (top 32)"
    elif size_ok:
        outcome, nxt = "size", "round 12 (not built): the gap is the reader; the owner's call on a larger SGS base"
    else:
        outcome, nxt = "split", "no single lever reaches its bar; report and stop"
    shares = None
    if d(gap):
        shares = {"view (Gemma top 20 vs 16)": d(gem_view) / d(gap) if d(gem_view) is not None else None,
                  "pages for the small reader": d(vs_ship["shipP"]) / d(gap) if d(vs_ship["shipP"]) is not None else None,
                  "reading at identical evidence": d(equal_ev) / d(gap) if d(equal_ev) is not None else None}
    out = {"tag": tag, "benchmark": "scripts/assets/planck3_tasks_fresh10.json (cached Brave answers, no searches)",
           "accuracy": acc, "replay": {"pass": valid, "bar": REPLAY_MIN, "vs": R10_SHIPPED, **rep},
           "gap": gap, "config_vs_shipped": vs_ship, "best_config": best, "config": CONFIGS.get(best) if cfg_ok else None,
           "gemma_view": gem_view, "equal_evidence": equal_ev, "reach": rch, "gap_shares": shares,
           "latency": {"fresh10": ms_per_cand(results, t("ship")), "old243": ms_per_cand(results, t("old")),
                       "r8_read_ms_per_q": (read_json(results / R8_OLD / "summary.json").get("timing_ms_per_task", {}).get("read_ms")
                                            if (results / R8_OLD / "summary.json").exists() else None)},
           "outcome": outcome, "next": nxt}
    write_json(results / f"r11_plan{tag}.json", out)
    print(f"[r11] diagnosis: {outcome} -> {nxt}")
    return out


def decide(results: Path, tag: str = "") -> dict:
    from .round7 import product_checks
    from .round9 import by_regime, paired
    plan = read_json(results / f"r11_plan{tag}.json")
    if plan.get("outcome") != "config":
        raise SystemExit(f"round 11's diagnosis chose {plan.get('outcome')!r}: there is no configuration to judge")
    g = paired(results, PRODUCT + tag, SHIPPED + tag)
    gp = "n" in g and g["acc_a"] - g["acc_b"] >= G_MARGIN and g["p"] < 0.05 and g["a_only"] > g["b_only"]
    cfg = plan["config"]
    out = {"reader": "x-hertz", "reader_why": f"round 11: round 9's reader, configuration {plan['best_config']} fixed by the diagnosis",
           "reader_checkpoint": "checkpoints/xreader_hertz_rc/reader.pt", "reader_quant": "int8",
           "reader_cands": cfg["reader_cands"], "open_below": cfg["open_below"],
           "writer": "template", "writer_why": "round 6: no small writer passes rule W", "run": PRODUCT, "tag": tag,
           "benchmark": "scripts/assets/planck3_tasks_fresh11.json",
           "rule_g11": {"pass": gp, "margin": G_MARGIN, **g},
           "vs_gemma": paired(results, PRODUCT + tag, REF + tag),
           "vs_heuristic": paired(results, PRODUCT + tag, FLOOR + tag),
           "by_regime": {x: by_regime(results, x + tag) for x in (FLOOR, SHIPPED, PRODUCT, REF)}}
    out.update(product_checks(results, PRODUCT, tag))
    s = results / (PRODUCT + tag) / "summary.json"
    if s.exists():
        tm = read_json(s).get("timing_ms_per_task", {})
        ms = tm.get("read_ms", 0) + tm.get("write_ms", 0)
        out["rule_l"] = {"pass": ms <= L_MS, "ms": ms, "fetch_ms_reported": tm.get("fetch_ms"),
                         "measured_on": "every fresh question, CPU, 4 threads, read + write (as rounds 8-10; page fetches reported)"}
    checks = all(out.get(k, {}).get("pass") for k in ("rule_l", "rule_e"))
    out["rule_r"] = {"pass": gp and checks, "acc": g.get("acc_a"), "why": "round 11: rule G11 and the product checks"}
    out["outcome"] = (f"ship: round 9's reader with {plan['best_config']}" if gp and checks else
                      "ship blocked by a product check (L or E); keep round 9's configuration" if gp else
                      "keep round 9's configuration: the diagnosis's lever did not hold on new questions")
    write_json(results / f"r11_product{tag}.json", out)
    print(f"[r11] rule G11 {'PASS' if gp else 'FAIL'} -> {out['outcome']}")
    return out


def report_lines(results: Path) -> list[str]:
    p = results / "r11_plan.json"
    if not p.exists():
        return []
    d = read_json(p)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    lines = ["## Round 11: where is the Gemma gap? (SETUP_planck_20261109.md)", "",
             f"- **Replay** {'PASS' if d['replay']['pass'] else 'FAIL'}: same value as round 10 on {f3(d['replay'].get('same_value'))}",
             "- **Accuracy on fresh10 (cached):** " + "; ".join(f"{k} {f3(v)}" for k, v in d["accuracy"].items() if k != "old")]
    for k, x in [("gap (Gemma vs shipped)", d["gap"]), ("Gemma top 20 vs top 16", d["gemma_view"]),
                 ("Gemma vs shipped at identical evidence", d["equal_evidence"])] + \
                [(f"{c} vs shipped", d["config_vs_shipped"][c]) for c in CONFIGS]:
        if "n" in x:
            lines.append(f"- **{k}:** {f3(x['acc_a'])} vs {f3(x['acc_b'])} ({x['a_only']} vs {x['b_only']}, p {x['p']:.3g}, n {x['n']})")
    r = d.get("reach", {})
    if "top16" in r:
        lines.append(f"- **Reach (a right candidate in view):** top 16 {f3(r['top16'])}, top 20 {f3(r['top20'])}, top 32 {f3(r['top32'])}")
    lat = d.get("latency", {})
    for k in ("fresh10", "old243"):
        x = lat.get(k, {})
        if "ms_per_cand" in x:
            lines.append(f"- **Latency {k}:** {x['cands_per_q']:.0f} candidates/question, {x['ms_per_cand']:.0f} ms each, {x['read_ms_per_q']:.0f} ms/question")
    lines += [f"- **Diagnosis (pre-registered):** {d['outcome']}: {d['next']}", ""]
    q = results / "r11_product.json"
    if q.exists():
        pr = read_json(q)
        g = pr.get("rule_g11", {})
        if "n" in g:
            lines.append(f"- **G11** {'PASS' if g['pass'] else 'FAIL'}: {f3(g['acc_a'])} vs {f3(g['acc_b'])} ({g['a_only']} vs {g['b_only']}, p {g['p']:.3g}, n {g['n']})")
        lines += [f"- **Outcome (pre-registered):** {pr['outcome']}", ""]
    return lines


def add_cli(sub):
    p = sub.add_parser("r11", help="round 11: diagnose (where is the Gemma gap -> r11_plan.json) | decide (G11 -> r11_product.json)")
    p.add_argument("stage", choices=["diagnose", "decide", "plan"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        if a.stage == "plan":           # the runner reads the outcome and the configuration
            d = read_json(RESULTS / f"r11_plan{a.tag}.json")
            c = d.get("config") or {}
            print(f"{d['outcome']} {c.get('reader_cands', 16)} {c.get('open_below', 7.0)}")
            return
        (diagnose if a.stage == "diagnose" else decide)(RESULTS, a.tag)
    p.set_defaults(fn=run)
