"""
Round 12 (SETUP_planck_20261113.md): distillation at scale. Can Hertz's 640M reader learn Gemma's reading?

Round 11 found the whole 12-point gap is reading: at identical evidence (top 16, both pages) Gemma is right
on 60.9% of fresh10, the shipped reader on 48.2% (p 0.0005). Our own size slope (Planck 100M -> Hertz 640M
on the same reading recipe, about +11 points) puts closing the gap by size alone at a base the 4090 cannot
pretrain. The one lever left that is cheap is the RECIPE: round 10 trained on Gemma's reading of only 2,710
pools (weight 0.5) and moved nothing. Round 12 asks the same question at 14x the scale:

    sample    30,000 of round 7's public reading pools (seeded, streamed; the source file is never touched)
              + a copy of our own pools (round 6's points.jsonl) + round 10's Brave pools (already taught)
    teach     Gemma reads every pool once (the reader prompt and top-16 view it uses on the benchmark):
              ~37,000 soft labels, ~13 h at round 10's measured 1.28 s/pool, resumable
    train     from the SHIPPED reader (x-hertz rc), all of it, Gemma's soft labels at weight 1.0 (round 10:
              0.5), 2 epochs, early stop 1
    pilot     0 searches: the new reader re-reads round 11's cached fresh10 at the shipped configuration
              (int8 CPU, top 16, pages below 7). Compared with round 11's D_ship on the same evidence
    judge     only if the pilot passes: a NEW fresh benchmark (fresh12: Wikidata 2026 items unused by
              rounds 1-11 + portal news after 2026-11-03), committed before any reader runs; ~300 searches

Pre-registered (section 4 of the doc), BEFORE any round 12 number exists:
    P     pilot: the product beats D_ship on fresh10 by >= 3 points with more only-right than only-wrong.
          Else stop: distillation at 37k pools does not move the reader; the recipe is spent at 640M and
          size is the only lever left (owner's call, no round 13 built)
    G12   on fresh12 the product beats the shipped reader by >= 3 points, exact McNemar p < 0.05
    GAP   reported: the share of the gap to Gemma the product closes on fresh12 (and on the pilot)
    Q     int8 CPU within 2 points of the same reader on the GPU; L <= 5 s per question on CPU; E as before
    D     reported: on held-out pools, how often the reader picks Gemma's candidate (the mechanism)
    outcome   P + G12 + Q + L + E -> ship x-hertz r12; else keep round 9's reader
"""

from pathlib import Path

from .util import read_json, write_json

R11_SHIP, R11_GEM = "r11d_x-hertz-rc_cpu-int8-top16", "r11d_gemma_template"     # round 11's cached-evidence arms
PILOT = "r12p_x-hertz-r12_cpu-int8-top16"
PRODUCT, SHIPPED = "r12_x-hertz-r12_cpu-int8-top16", "r12_x-hertz-rc_cpu-int8-top16"
GPU, REF, FLOOR = "r12_x-hertz-r12_template", "r12_gemma_template", "r12_heuristic_template"
CKPT = "checkpoints/xreader_hertz_r12/reader.pt"
TRAIN_LOG = "results/planck3/xreader_hertz_r12/train_log.json"
P_MARGIN, G_MARGIN, Q_MAX, L_MS = 0.03, 0.03, 0.02, 5000
RC_SAMPLE, TEACHER_W = 30000, 1.0


def _gap_share(prod: dict, ref: dict) -> float | None:
    """(product - shipped) / (Gemma - shipped), both paired against the same shipped run."""
    if "n" not in prod or "n" not in ref or ref["acc_a"] <= ref["acc_b"]:
        return None
    return (prod["acc_a"] - prod["acc_b"]) / (ref["acc_a"] - ref["acc_b"])


def _mechanism(tag: str = "") -> dict:
    from .util import REPO_ROOT
    p = REPO_ROOT / TRAIN_LOG.replace("xreader_hertz_r12", f"xreader_hertz_r12{tag}")
    if not p.exists():
        return {"why": f"{p} missing"}
    d = read_json(p)
    return {"teacher_agreement": d.get("test_teacher_agreement"), "test": d.get("test"),
            "teacher_labelled": d.get("teacher_labelled"), "n_points": d.get("n_points"), "epochs": d.get("epochs")}


def pilot(results: Path, tag: str = "") -> dict:
    """Rule P on cached fresh10 (0 searches) -> r12_pilot.json. The runner reads `outcome`."""
    from .round9 import paired
    g = paired(results, PILOT + tag, R11_SHIP + tag)
    ref = paired(results, R11_GEM + tag, R11_SHIP + tag)
    ok = "n" in g and g["acc_a"] - g["acc_b"] >= P_MARGIN and g["a_only"] > g["b_only"]
    out = {"tag": tag, "benchmark": "scripts/assets/planck3_tasks_fresh10.json (round 11's cached evidence, 0 searches)",
           "rule_p": {"pass": ok, "margin": P_MARGIN, **g}, "gap_closed": {"share": _gap_share(g, ref), "gemma_vs_shipped": ref},
           "mechanism": _mechanism(tag),
           "outcome": "judge" if ok else "stop",
           "next": ("phase 2: fresh12 and the ship rule (~300 Brave searches)" if ok else
                    "stop: distillation at scale does not move the reader at identical evidence; the recipe is spent at "
                    "640M and a larger base is the only lever left (owner's call; round 13 not built)")}
    write_json(results / f"r12_pilot{tag}.json", out)
    d = (g["acc_a"] - g["acc_b"]) if "n" in g else None
    print(f"[r12] rule P {'PASS' if ok else 'FAIL'} ({'n/a' if d is None else f'{d:+.3f}'}) -> {out['outcome']}")
    return out


def decide(results: Path, tag: str = "") -> dict:
    from .round7 import product_checks
    from .round9 import by_regime, paired
    pl = read_json(results / f"r12_pilot{tag}.json")
    if pl.get("outcome") != "judge":
        raise SystemExit("round 12's pilot did not pass rule P: there is nothing to judge on fresh12")
    g = paired(results, PRODUCT + tag, SHIPPED + tag)
    ref = paired(results, REF + tag, SHIPPED + tag)
    gp = "n" in g and g["acc_a"] - g["acc_b"] >= G_MARGIN and g["p"] < 0.05 and g["a_only"] > g["b_only"]
    q = paired(results, PRODUCT + tag, GPU + tag)
    out = {"reader": "x-hertz", "reader_why": "fixed before the run: the shipped reader distilled from Gemma on ~37k pools",
           "reader_checkpoint": CKPT, "reader_quant": "int8", "reader_cands": 16, "open_below": 7.0,
           "writer": "template", "writer_why": "round 6: no small writer passes rule W", "run": PRODUCT, "tag": tag,
           "benchmark": "scripts/assets/planck3_tasks_fresh12.json",
           "rule_p": pl["rule_p"],
           "rule_g12": {"pass": gp, "margin": G_MARGIN, **g},
           "gap_closed": {"share": _gap_share(g, ref), "gemma_vs_shipped": ref},
           "rule_q": {"pass": "n" in q and q["acc_b"] - q["acc_a"] <= Q_MAX, **q},
           "vs_gemma": paired(results, PRODUCT + tag, REF + tag),
           "vs_heuristic": paired(results, PRODUCT + tag, FLOOR + tag),
           "mechanism": _mechanism(tag),
           "by_regime": {x: by_regime(results, x + tag) for x in (FLOOR, SHIPPED, PRODUCT, GPU, REF)}}
    out.update(product_checks(results, PRODUCT, tag))
    s = results / (PRODUCT + tag) / "summary.json"
    if s.exists():
        tm = read_json(s).get("timing_ms_per_task", {})
        ms = tm.get("read_ms", 0) + tm.get("write_ms", 0)
        out["rule_l"] = {"pass": ms <= L_MS, "ms": ms, "fetch_ms_reported": tm.get("fetch_ms"),
                         "measured_on": "every fresh question, CPU, 4 threads, read + write (as rounds 8-11)"}
    checks = all(out.get(k, {}).get("pass") for k in ("rule_q", "rule_l", "rule_e"))
    # rule_r keeps the chat factory's contract: a product ships only if rule_r.pass
    out["rule_r"] = {"pass": gp and checks, "acc": g.get("acc_a"), "why": "round 12: rules P, G12 and the product checks"}
    out["outcome"] = ("ship: x-hertz r12 (distilled) beats the shipped reader on new fresh questions" if gp and checks else
                      "ship blocked by a product check (Q, L or E); keep round 9's reader" if gp else
                      "keep round 9's reader: the pilot's gain did not hold on new questions")
    write_json(results / f"r12_product{tag}.json", out)
    print(f"[r12] rule G12 {'PASS' if gp else 'FAIL'} -> {out['outcome']}")
    return out


def report_lines(results: Path) -> list[str]:
    p = results / "r12_pilot.json"
    if not p.exists():
        return []
    d = read_json(p)
    f3 = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    pf = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    lines = ["## Round 12: distillation at scale (SETUP_planck_20261113.md)", ""]
    m = d.get("mechanism", {})
    ta = m.get("teacher_agreement") or {}
    if ta.get("pools"):
        lines.append(f"- **D (held-out pools Gemma read, {ta['pools']}):** reader picks Gemma's candidate {f3(ta['reader_picks_teacher'])}; "
                     f"right where a right one exists: reader {f3(ta['reader_right'])} vs Gemma {f3(ta['teacher_right'])}")
    for k, label, x in (("rule_p", "P (pilot, cached fresh10)", d), ):
        r = x.get(k, {})
        if "n" in r:
            lines.append(f"- **{label}** {pf(r['pass'])}: {f3(r['acc_a'])} vs shipped {f3(r['acc_b'])} "
                         f"({r['a_only']} vs {r['b_only']}, p {r['p']:.3g}, n {r['n']}); gap closed {f3(d['gap_closed']['share'])}")
    lines.append(f"- **Pilot outcome (pre-registered):** {d['outcome']}: {d['next']}")
    q = results / "r12_product.json"
    if q.exists():
        pr = read_json(q)
        for k, label in (("rule_g12", "G12 (vs the shipped reader, fresh12)"), ("rule_q", "Q (int8 CPU vs GPU)")):
            r = pr.get(k, {})
            if "n" in r:
                lines.append(f"- **{label}** {pf(r['pass'])}: {f3(r['acc_a'])} vs {f3(r['acc_b'])} ({r['a_only']} vs {r['b_only']}, p {r['p']:.3g}, n {r['n']})")
        for k in ("rule_l", "rule_e", "rule_c1"):
            if k in pr:
                lines.append(f"- **{k[5:].upper()}** {pf(pr[k].get('pass'))}: " + ", ".join(f"{a} {b}" for a, b in pr[k].items() if a != "pass" and not isinstance(b, dict)))
        lines.append(f"- **Gap closed on fresh12:** {f3(pr['gap_closed']['share'])}")
        lines.append(f"- **Outcome (pre-registered):** {pr['outcome']}")
    return lines + [""]


def add_cli(sub):
    p = sub.add_parser("r12", help="round 12: pilot (rule P on cached fresh10 -> r12_pilot.json) | decide (G12 -> r12_product.json) | plan")
    p.add_argument("stage", choices=["pilot", "decide", "plan"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        if a.stage == "plan":           # the runner reads the pilot's outcome
            print(read_json(RESULTS / f"r12_pilot{a.tag}.json")["outcome"])
            return
        (pilot if a.stage == "pilot" else decide)(RESULTS, a.tag)
    p.set_defaults(fn=run)
