"""
Round 6 (SETUP_planck_20261012.md): report rows, the product decision, and the writer judge.

The runs themselves reuse the round 5 evaluator (`planck3.py r5 ... --out results/planck3/r6_*`):
same 243 questions, same weighing, now on the Brave Search API with the relevance check, with the
reader trained once as a cross-encoder (xreader.py) and the writers retrained on the fixed data.

    product   r6_product.json, written by the runner from the PRE-REGISTERED rules (section 5):
              reader = x-planck unless x-hertz beats it by >= 3 points of test choice accuracy on
              the reader's own held-out questions; writer = the smallest writer passing rule W on
              the benchmark run, else the template
    judge     Gemma compares the small writer's shown answer with the template answer for the same
              question, blind and in random order: which reads better (reported, no bar)
"""

import json
import random
import re
from pathlib import Path

from .util import read_json, read_jsonl, write_json

JUDGE_SYSTEM = """You compare two answers a search assistant could show for the same question. Both state the
same answer from the same sources and were checked for accuracy. Judge only how well they read for the
person who asked: clear, natural, direct, no padding, hedged when unsure. Reply with ONLY one of: A, B, tie"""


def judge(run_dir: Path, gemma_path: str, limit: int = 120, seed: int = 0) -> dict:
    """Blind A/B of the small writer's own shown answers against the template (Gemma decides)."""
    from .policies import make_policy
    rng = random.Random(seed)
    pairs = [(p["q"], p["text"], p["template"]) for r in read_jsonl(run_dir / "results.jsonl") for p in r["parts"]
             if p.get("tries") and not p.get("fallback") and p.get("template") and p["text"] != p["template"]]
    rng.shuffle(pairs)
    pairs = pairs[:limit]
    pol = make_policy("gemma", gemma_path=gemma_path)
    wins = {"model": 0, "template": 0, "tie": 0}
    rows = []
    for q, model_text, templ in pairs:
        flip = rng.random() < 0.5
        a, b = (templ, model_text) if flip else (model_text, templ)
        raw = pol._complete(JUDGE_SYSTEM, f"QUESTION: {q}\nA: {a}\nB: {b}").strip()
        m = re.match(r"(?i)\s*(A|B|tie)\b", raw)
        v = m.group(1).lower() if m else "tie"
        who = "tie" if v == "tie" else ("model" if (v == "a") != flip else "template")
        wins[who] += 1
        rows.append({"q": q, "model": model_text, "template": templ, "verdict": who})
    n = max(len(pairs), 1)
    out = {"run": run_dir.name, "pairs": len(pairs), **wins, "model_share": wins["model"] / n,
           "template_share": wins["template"] / n, "tie_share": wins["tie"] / n}
    write_json(run_dir / "judge.json", out)
    with open(run_dir / "judge_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[judge] {run_dir.name}: small writer preferred {wins['model']}, template {wins['template']}, tie {wins['tie']} (n={len(pairs)})")
    return out


def w_passes(summary: dict) -> bool:
    """Rule W: the writer's own answers (any of its tries) pass the faithfulness check >= 95%."""
    return (summary.get("writer_faithful") or 0) >= 0.95


def choose_reader(results: Path, tag: str = "") -> dict:
    """Pre-registered: Hertz (640M) only if it beats Planck (100M) by >= 3 points on held-out questions."""
    logs = {b: results / f"xreader_{b}{tag}" / "train_log.json" for b in ("planck", "hertz")}
    acc = {b: read_json(p)["test"]["choice_acc"] for b, p in logs.items() if p.exists()}
    if "planck" not in acc:
        return {"reader": "x-hertz" if "hertz" in acc else "planck", "test_choice_acc": acc, "why": "x-planck missing"}
    if "hertz" in acc and acc["hertz"] - acc["planck"] >= 0.03:
        return {"reader": "x-hertz", "test_choice_acc": acc, "why": "Hertz reads >= 3 points better on held-out questions"}
    return {"reader": "x-planck", "test_choice_acc": acc, "why": "Planck reads within 3 points of Hertz (or Hertz missing)"}


def report_lines(results: Path) -> list[str]:
    from .wikirace import mcnemar
    runs = {p.parent.name: read_json(p) for p in sorted(results.glob("r6_*/summary.json"))}
    lines = []
    xr = {b: read_json(p) for b in ("planck", "hertz") if (p := results / f"xreader_{b}" / "train_log.json").exists()}
    if xr:
        lines += ["## Round 6: the reader trained once (cross-encoder) (SETUP_planck_20261012.md)", "",
                  "| reader | params | train / val / test pools | test choice acc | ranker | gate AUC | gate ECE | by group (choice vs ranker) |",
                  "|---|---|---|---|---|---|---|---|"]
        for b, t in xr.items():
            n, te = t["n_points"], t["test"]
            grp = "; ".join(f"{g} {m['choice_acc']:.3f} vs {m['ranker_acc']:.3f} (n={m['pools_with_a_right_candidate']})"
                            for g, m in t["test_by_group"].items())
            lines.append(f"| x-{b} | {t['params'] / 1e6:.0f}M | {n['train']} / {n['val']} / {n['test']} | {te['choice_acc']:.3f} | "
                         f"{te['ranker_acc']:.3f} | {te['gate_auc'] if te['gate_auc'] is None else round(te['gate_auc'], 3)} | "
                         f"{te['gate_ece'] if te['gate_ece'] is None else round(te['gate_ece'], 3)} | {grp} |")
        lines.append("")
    if not runs:
        return lines
    f3 = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
    lines += ["## Round 6: read -> weigh -> write on Brave, relevance-checked (SETUP_planck_20261012.md)", "",
              "| run | n | value shown is right | 143 fresh + long-tail | 100 news + info | high: n / precision | good | low | sceptical | none | monotone | search on topic | writer faithful (any try) | first try | tries | read ms | write ms | changed by personal |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def recs(name):
        return {r["task_id"]: r for r in read_jsonl(results / name / "results.jsonl")}

    def acc_of(rs, regimes):
        sub = [r for r in rs.values() if r["regime"] in regimes]
        return f"{sum(r['shown_correct'] for r in sub) / len(sub):.3f} (n={len(sub)})" if sub else ""
    for name, s in runs.items():
        b, t, rs = s["bands"], s.get("timing_ms_per_task", {}), recs(name)
        cell = lambda k: f"{b[k]['n']} / {f3(b[k]['precision'])}"  # noqa: E731
        lines.append(f"| {name} | {s['n']} | {s['accuracy']:.3f} | {acc_of(rs, ('fresh', 'long_tail'))} | "
                     f"{acc_of(rs, ('news', 'general_info'))} | {cell('high')} | {cell('good')} | {cell('low')} | {cell('sceptical')} | "
                     f"{b['none']['n']} | {s['bands_monotone']} | {f3(s.get('search_on_topic'))} | {f3(s['writer_faithful'])} | "
                     f"{f3(s.get('writer_first_try_faithful'))} | {f3(s.get('writer_mean_tries'))} | {t.get('read_ms', 0):.0f} | "
                     f"{t.get('write_ms', 0):.0f} | {s['changed_by_personal']:.3f} |")
    lines.append("")
    prod = read_json(results / "r6_product.json") if (results / "r6_product.json").exists() else {}
    pname = prod.get("run")
    pairs = []
    if pname in runs:
        for other in ("r6_planck_template", "r6_gemma_template", "r6_gemma_gemma", "r6_heuristic_template"):
            if other in runs:
                pairs.append((pname, other))
        for name in runs:
            if name.startswith(pname + "_"):
                pairs.append((name, pname))
    for a, b in (("r6_x-planck_template", "r6_planck_template"), ("r6_x-hertz_template", "r6_x-planck_template"),
                 ("r6_x-planck_template", "r6_gemma_template")):
        if a in runs and b in runs and (a, b) not in pairs:
            pairs.append((a, b))
    if pairs:
        lines += ["| paired (same questions) | n | run | vs | only run right | only vs right | McNemar p |", "|---|---|---|---|---|---|---|"]
        for a, b in pairs:
            ra, rb = recs(a), recs(b)
            ids = sorted(set(ra) & set(rb))
            if not ids:
                continue
            xa, xb = [ra[i]["shown_correct"] for i in ids], [rb[i]["shown_correct"] for i in ids]
            m = mcnemar(xa, xb)
            lines.append(f"| {a} vs {b} | {len(ids)} | {sum(xa) / len(ids):.3f} | {sum(xb) / len(ids):.3f} | {m['a_only']} | {m['b_only']} | {m['p']:.3g} |")
        lines.append("")
    if prod:
        lines.append(f"- product (pre-registered choice): reader {prod.get('reader')} ({prod.get('reader_why')}); "
                     f"writer {prod.get('writer')} ({prod.get('writer_why')})")
    for name in runs:
        j = results / name / "judge.json"
        if j.exists():
            d = read_json(j)
            lines.append(f"- writer judge ({name}, Gemma, blind): small writer preferred {d['model']}, template {d['template']}, "
                         f"tie {d['tie']} of {d['pairs']}")
    for p_ in sorted(results.glob("kvcheck_*/summary.json")):
        d = read_json(p_)
        lines.append(f"- KV cache ({p_.parent.name[8:]}): {d['identical']:.0%} of {d['prompts']} greedy answers identical; "
                     f"{d['ms_cached']:.0f} ms cached vs {d['ms_uncached']:.0f} ms uncached ({d['device']})")
    for name in ("writer_hertz6", "writer_planck6"):
        p_ = results / name / "train_log.json"
        if p_.exists():
            t = read_json(p_)
            lines.append(f"- {name}: {t['examples']} training examples ({t['train']} train / {t['val']} val), "
                         f"best val loss {t['best_val_loss']:.3f}, {t['params'] / 1e6:.0f}M params")
    return lines + [""]


def product_decision(results: Path, tag: str = "") -> dict:
    """Called by the runner after the writer runs: which reader and writer ship (rules in section 5)."""
    rd = choose_reader(results, tag)
    choice = {"reader": rd["reader"], "reader_why": rd["why"], "test_choice_acc": rd["test_choice_acc"]}
    for w in ("planck", "hertz"):                    # smallest first
        s = results / f"r6_{rd['reader']}_{w}{tag}" / "summary.json"
        if s.exists() and w_passes(read_json(s)):
            choice.update(writer=w, writer_why=f"{w} writer passes rule W ({read_json(s)['writer_faithful']:.3f})")
            break
    else:
        choice.update(writer="template", writer_why="no small writer passes rule W: the template writes")
    choice["run"] = f"r6_{choice['reader']}_{choice['writer']}"
    choice["tag"] = tag
    write_json(results / f"r6_product{tag}.json", choice)
    print(f"[r6] product: reader {choice['reader']} ({choice['reader_why']}); writer {choice['writer']} ({choice['writer_why']})")
    return choice


def add_cli(sub):
    p = sub.add_parser("r6", help="round 6 helpers: product (pre-registered choice) | reader (which reader ships)")
    p.add_argument("stage", choices=["product", "reader"])
    p.add_argument("--tag", default="")

    def run(a):
        from .round5 import RESULTS
        if a.stage == "reader":
            print(choose_reader(RESULTS, a.tag)["reader"])
        else:
            product_decision(RESULTS, a.tag)
    p.set_defaults(fn=run)
