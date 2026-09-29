"""
Planck 3.0 command line. On the Windows box, prefer the wrapper:

    .\\scripts\\planck3.ps1 <command>        (sets up venv/env, then calls this)

Commands:
    ask "question" --type year        one question -> answer card (try it first)
    g0  --policy gemma                G0: run the seed benchmark, write trajectories + gate verdict
    watch add|list|run                background watch tasks (read-only)
    search-check                      is SearXNG up? (auto falls back to Wikipedia search)
    wikirace build|embed|train|eval   G1 pipeline (see src/planck3/wikirace.py)
    report                            print every G0/G1 summary under results/planck3/

Plan: SETUP_092026_planck3.md
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.planck3.util import REPO_ROOT, append_jsonl, read_json, utf8_console, write_json  # noqa: E402

RESULTS = REPO_ROOT / "results" / "planck3"
CACHE = REPO_ROOT / "data" / "planck3_cache"
TASKS = REPO_ROOT / "scripts" / "assets" / "planck3_tasks.json"
G0_PASS, G0_KILL = 0.60, 0.40


def build_web(args, log_path=None):
    from src.planck3.web import Web, WebCache
    cache = WebCache(args.cache_dir, mode=args.net)
    backend = args.search
    web = Web(cache, search_backend="searxng" if backend == "auto" else backend,
              searxng_url=args.searxng_url, log_path=log_path)
    if backend == "auto":
        if args.net != "replay" and web.searxng_alive():
            web.backend = "searxng"
        else:
            if args.net != "replay":
                print(f"[planck3] SearXNG not reachable at {args.searxng_url}; using Wikipedia search "
                      f"(start it with: .\\scripts\\planck3.ps1 searxng)")
            web.backend = "wikipedia"
    print(f"[planck3] search={web.backend} net={args.net} cache={args.cache_dir}")
    return web


def add_web_args(p):
    p.add_argument("--search", default="auto", choices=["auto", "searxng", "wikipedia"])
    p.add_argument("--searxng-url", default="http://localhost:8888")
    p.add_argument("--net", default="live", choices=["live", "replay", "refresh"],
                   help="live: cache-then-network (default); replay: cache only; refresh: network only")
    p.add_argument("--cache-dir", default=str(CACHE))
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "gemma", "bedrock"])
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")
    p.add_argument("--model-id", default=None, help="Bedrock model id (bedrock policy)")
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=None)


def make_policy(args):
    from src.planck3.policies import make_policy as mk
    return mk(args.policy, gemma_path=args.gemma_path, model_id=args.model_id,
              region=args.region, profile=args.profile)


# ── ask ──────────────────────────────────────────────────────────────────
def cmd_ask(args):
    from src.planck3.harness import Harness, render_card
    from src.planck3.store import Store
    web = build_web(args)
    store = Store(args.store)
    h = Harness(make_policy(args), web, store)
    res = h.run_fact(args.question, args.type)
    print()
    print(render_card(args.question, res))
    if args.verbose:
        for s in res["trajectory"]:
            print(f"  step {s['step']} [{s['phase']}] -> {s['decision']}" + ("" if s["valid"] else "  (INVALID)"))
    print(f"\n  {res['steps']} decisions, {res['web_calls']} web calls, "
          f"{sum(res['decision_ms']) / max(len(res['decision_ms']), 1):.0f} ms/decision; store: {args.store}")


# ── g0 ───────────────────────────────────────────────────────────────────
def cmd_g0(args):
    from src.planck3.harness import Harness, render_card, render_table
    from src.planck3.metrics import is_correct, summarize
    from src.planck3.store import Store

    spec = read_json(args.tasks)
    tasks = spec["tasks"][: args.limit] if args.limit else spec["tasks"]
    run = args.out or str(RESULTS / f"g0_{args.policy}_{datetime.now():%Y%m%d_%H%M}")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    store_path = Path(args.store) if args.store else run / "store.sqlite"
    if args.store is None and store_path.exists():
        store_path.unlink()  # a run's own store starts empty unless you pass --store (G3)
    web = build_web(args, log_path=run / "web_log.jsonl")
    store = Store(store_path)
    h = Harness(make_policy(args), web, store)
    traj_path, res_path = run / "trajectories.jsonl", run / "results.jsonl"
    for p in (traj_path, res_path):
        if p.exists():
            p.unlink()

    records, cards = [], []
    t_start = time.time()
    for i, task in enumerate(tasks, 1):
        if task["family"] == "compare":
            out = h.run_compare(task)
            cells_ok = []
            for c in out["cells"]:
                gold = task["gold"][c["item"]][c["attribute"]]
                c["correct"] = c["answered"] and is_correct(c["value"], gold, c.get("answer_type") or
                                                            _att_type(task, c["attribute"]))
                cells_ok.append(c["correct"])
                c["reachable"] = _gold_reachable(c, gold, _att_type(task, c["attribute"]))
                _log_traj(traj_path, task["id"], c, c["correct"], sub=f"{c['item']}/{c['attribute']}")
            answered = all(c["answered"] for c in out["cells"])
            rec = {"task_id": task["id"], "family": "compare", "answered": answered,
                   "correct": all(cells_ok), "cell_accuracy": sum(cells_ok) / len(cells_ok),
                   "p": min((c["p"] for c in out["cells"]), default=0.0),
                   "steps": sum(c["steps"] for c in out["cells"]),
                   "web_calls": sum(c["web_calls"] for c in out["cells"]),
                   "decision_ms": [m for c in out["cells"] for m in c["decision_ms"]],
                   "invalid": sum(c["invalid"] for c in out["cells"]),
                   "gold_reachable": all(c["reachable"] for c in out["cells"]),
                   "cells": [{k: c[k] for k in ("item", "attribute", "value", "p", "source_url", "correct")}
                             for c in out["cells"]]}
            cards.append(f"### {task['id']} {task['question']}\n\n{render_table(task, out)}\n")
        else:
            res = h.run_fact(task["question"], task["answer_type"], entity=task.get("entity"),
                             attribute=task.get("attribute"), task_id=task["id"])
            correct = res["answered"] and is_correct(res["value"], task["gold"], task["answer_type"])
            _log_traj(traj_path, task["id"], res, correct)
            rec = {"task_id": task["id"], "family": "fact", "answered": res["answered"], "correct": correct,
                   "value": res["value"], "gold": task["gold"], "p": res["p"], "verified": res["verified"],
                   "source_url": res["source_url"], "reason": res["reason"], "from_store": res["from_store"],
                   "steps": res["steps"], "web_calls": res["web_calls"], "decision_ms": res["decision_ms"],
                   "invalid": res["invalid"],
                   "gold_reachable": _gold_reachable(res, task["gold"], task["answer_type"])}
            cards.append(f"### {task['id']} ({'OK' if correct else 'WRONG' if res['answered'] else 'abstain'})\n\n"
                         f"{render_card(task['question'], res)}\n")
        append_jsonl(res_path, rec)
        records.append(rec)
        mark = "OK " if rec["correct"] else ("-- " if not rec["answered"] else "XX ")
        print(f"[{i:>3}/{len(tasks)}] {mark} {task['id']:<4} {rec.get('value') or ''!s:<28} "
              f"p={rec['p']:.2f} steps={rec['steps']} web={rec['web_calls']}")

    summ = summarize(records)
    summ.update({"policy": args.policy, "search": web.backend, "net": args.net, "tasks": str(args.tasks),
                 "n_invalid_decisions": sum(r["invalid"] for r in records),
                 "gold_reachable": sum(r["gold_reachable"] for r in records) / max(len(records), 1),
                 "web_calls_network": {k: v for k, v in web.calls.items()},
                 "wall_s": round(time.time() - t_start, 1), "store_facts": store.n_facts(),
                 "top_domains": store.top_domains(8)})
    summ["gate"] = _g0_verdict(summ)
    write_json(run / "summary.json", summ)
    with open(run / "cards.md", "w", encoding="utf-8") as f:
        f.write(f"# G0 answer cards ({args.policy})\n\n" + "\n".join(cards))
    _print_summary("G0", summ)
    print(f"\n  run dir: {run}")


def _gold_reachable(res, gold, answer_type) -> bool:
    """Was a correct value among the spans shown on ANY opened page? (tool ceiling)"""
    from src.planck3.metrics import is_correct
    for step in res["trajectory"]:
        for sp in step["obs"]["lists"].get("spans", []):
            if is_correct(sp["value"], gold, answer_type):
                return True
    return False


def _att_type(task, name):
    return next(a["answer_type"] for a in task["attributes"] if a["name"] == name)


def _log_traj(path, task_id, res, correct, sub=None):
    for step in res["trajectory"]:
        append_jsonl(path, {"task_id": task_id, "sub": sub, "task_correct": bool(correct), **step})


def _g0_verdict(s):
    if s["policy"] == "heuristic":
        return {"verdict": "BASELINE", "note": "heuristic is the no-model floor; the G0 gate applies to teachers"}
    if s["success"] >= G0_PASS:
        v = "PASS"
    elif s["success"] < G0_KILL:
        v = "KILL (fix the action space / tools before training)"
    else:
        v = "GREY ZONE (inspect failures in cards.md before training)"
    return {"verdict": v, "success": round(s["success"], 3), "pass_at": G0_PASS, "kill_below": G0_KILL}


def _print_summary(name, s):
    print(f"\n==== {name} summary ({s.get('policy')}) ====")
    for k in ("n", "success", "answered_rate", "wrong_when_answered", "ece", "mean_steps",
              "mean_web_calls", "mean_decision_ms", "n_invalid_decisions", "gold_reachable"):
        v = s.get(k)
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
    for fam, d in s.get("by_family", {}).items():
        print(f"  family {fam:<15} n={d['n']} success={d['success']:.3f}")
    if "gate" in s:
        print(f"  GATE: {s['gate']['verdict']}")


# ── watch ────────────────────────────────────────────────────────────────
def cmd_watch(args):
    import json
    from src.planck3.harness import Harness
    from src.planck3.store import Store
    store = Store(args.store)
    if args.action == "add":
        cond = {"op": args.op} if args.op == "change" else {"op": args.op, "value": args.value}
        wid = store.add_watch(args.question, args.type, cond, entity=args.entity, attribute=args.attribute)
        print(f"watch #{wid} added: {args.question} [{json.dumps(cond)}]")
        return
    if args.action == "list":
        for w in store.watches():
            print(f"#{w['id']} {w['question']} cond={w['condition']} last={w['last_value']}")
        return
    args.net = "refresh"  # a watch must look at the live web, not the cache
    h = Harness(make_policy(args), build_web(args), store, use_store=False)
    for w in store.watches():
        r = h.run_watch(w)
        flag = "FIRED" if r["fired"] else "no change"
        print(f"#{r['watch_id']} {flag}: {r['question']}  {r['old']} -> {r['new']}")


# ── misc ─────────────────────────────────────────────────────────────────
def cmd_search_check(args):
    from src.planck3.web import Web, WebCache
    web = Web(WebCache(args.cache_dir, mode="refresh"), searxng_url=args.searxng_url)
    ok = web.searxng_alive()
    print(f"SearXNG at {args.searxng_url}: {'UP' if ok else 'DOWN'}")
    sys.exit(0 if ok else 1)


def cmd_report(args):
    for summ in sorted(RESULTS.glob("*/summary.json")):
        s = read_json(summ)
        gate = s.get("gate", {}).get("verdict", "")
        succ = s.get("success", s.get("rollout_success"))
        print(f"{summ.parent.name:<40} success={succ if succ is None else round(succ, 3)}  {gate}")


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--type", default="entity", choices=["year", "number", "date", "entity", "text"])
    p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
    p.add_argument("-v", "--verbose", action="store_true")
    add_web_args(p)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("g0")
    p.add_argument("--tasks", default=str(TASKS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--store", default=None, help="reuse a store across runs (G3); default = fresh per run")
    add_web_args(p)
    p.set_defaults(fn=cmd_g0)

    p = sub.add_parser("watch")
    p.add_argument("action", choices=["add", "list", "run"])
    p.add_argument("question", nargs="?")
    p.add_argument("--type", default="number")
    p.add_argument("--op", default="change", choices=["change", "lt", "gt"])
    p.add_argument("--value", type=float, default=None)
    p.add_argument("--entity", default=None)
    p.add_argument("--attribute", default=None)
    p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
    add_web_args(p)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("search-check")
    p.add_argument("--searxng-url", default="http://localhost:8888")
    p.add_argument("--cache-dir", default=str(CACHE))
    p.set_defaults(fn=cmd_search_check)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    from src.planck3 import wikirace
    wikirace.add_cli(sub)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
