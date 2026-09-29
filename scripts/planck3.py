"""
Planck 3.0 command line. On the Windows box, prefer the wrapper:

    .\\scripts\\planck3.ps1 <command>        (sets up venv/env, then calls this)

Commands:
    chat                              terminal chat with follow-ups (answer type inferred)
    serve                             local web chat at http://127.0.0.1:8010
    ask "question" --type year        one question -> answer card + decision trace
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


# ── chat ─────────────────────────────────────────────────────────────────
def _chat_harness_factory(args):
    """One web client + policy + store shared across chat sessions (Gemma loads once)."""
    from src.planck3.harness import Harness
    from src.planck3.store import Store
    web, policy, store = build_web(args), make_policy(args), Store(args.store)
    return lambda: Harness(policy, web, store)


def cmd_chat(args):
    from src.planck3.chat import ChatSession, chat_answer
    sess = ChatSession(_chat_harness_factory(args)())
    print("Planck 3.0 chat. Follow up naturally ('and H&M?', 'when was it founded?'). 'new' resets, 'quit' exits.")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ("quit", "exit"):
            break
        if msg == "new":
            sess.turns.clear()
            continue
        if not msg:
            continue
        t = sess.ask(msg)
        print(f"planck> {chat_answer(t)}")
        print(f"        (asked: \"{t.question}\" · {t.answer_type} · {t.rewrite} · "
              f"{t.result['steps']} decisions · {t.result['web_calls']} web calls)")


def cmd_serve(args):
    from src.planck3.serve import make_server
    srv = make_server(_chat_harness_factory(args), host=args.host, port=args.port)
    print(f"Planck 3.0 chat on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ── g0 ───────────────────────────────────────────────────────────────────
def cmd_g0(args):
    from src.planck3.cost import load_prices, summarize_cost
    from src.planck3.harness import Harness
    from src.planck3.metrics import summarize
    from src.planck3.store import Store

    spec = read_json(args.tasks)
    tasks = spec["tasks"][: args.limit] if args.limit else spec["tasks"]
    if args.family:
        tasks = [t for t in tasks if t["family"] in args.family.split(",")]
    run = args.out or str(RESULTS / f"g0_{args.policy}_{datetime.now():%Y%m%d_%H%M}")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    store_path = Path(args.store) if args.store else run / "store.sqlite"
    if args.store is None and store_path.exists():
        store_path.unlink()  # a run's own store starts empty unless you pass --store (G3)
    web = build_web(args, log_path=run / "web_log.jsonl")
    store = Store(store_path)
    policy = make_policy(args)
    h = Harness(policy, web, store)
    prices = load_prices(args.prices)
    ctx = {"h": h, "policy": policy, "web": web, "prices": prices, "traj": run / "trajectories.jsonl"}
    res_path = run / "results.jsonl"
    for p in (ctx["traj"], res_path):
        if p.exists():
            p.unlink()

    records, cards = [], []
    t_start = time.time()
    runners = {"fact": _run_fact_task, "compare": _run_compare_task, "chat": _run_chat_task}
    for i, task in enumerate(tasks, 1):
        rec, card = runners[task["family"]](task, ctx)
        append_jsonl(res_path, rec)
        records.append(rec)
        cards.append(card)
        mark = "OK " if rec["correct"] else ("-- " if not rec["answered"] else "XX ")
        print(f"[{i:>3}/{len(tasks)}] {mark} {task['id']:<4} {rec.get('value') or ''!s:<28} "
              f"p={rec['p']:.2f} steps={rec['steps']} web={rec['web_calls']} ${rec['cost']['total_usd']:.5f}")

    summ = summarize(records)
    summ.update({"policy": args.policy, "search": web.backend, "net": args.net, "tasks": str(args.tasks),
                 "n_invalid_decisions": sum(r["invalid"] for r in records),
                 "gold_reachable": sum(r["gold_reachable"] for r in records) / max(len(records), 1),
                 "serp_answer_rate_at3": sum(r["serp_visible"] for r in records) / max(len(records), 1),
                 "cost": summarize_cost(records, prices),
                 "web_calls_network": {k: v for k, v in web.calls.items()},
                 "wall_s": round(time.time() - t_start, 1), "store_facts": store.n_facts(),
                 "top_domains": store.top_domains(8)})
    summ["gate"] = _g0_verdict(summ)
    write_json(run / "summary.json", summ)
    with open(run / "cards.md", "w", encoding="utf-8") as f:
        f.write(f"# G0 answer cards ({args.policy})\n\n" + "\n".join(cards))
    _print_summary("G0", summ)
    print(f"\n  run dir: {run}")


def _score_fact(ctx, question, answer_type, gold, task_id, entity=None, attribute=None, sub=None):
    """Run one fact question through the harness and attach scoring + cost + baselines."""
    from src.planck3.cost import task_cost
    from src.planck3.metrics import is_correct
    res = ctx["h"].run_fact(question, answer_type, entity=entity, attribute=attribute, task_id=task_id)
    res["correct"] = res["answered"] and is_correct(res["value"], gold, answer_type)
    res["reachable"] = _gold_reachable(res, gold, answer_type)
    res["serp_visible"] = _serp_visible(res, gold, answer_type)
    res["cost"] = task_cost(ctx["prices"], ctx["policy"].kind, res["usage"], res["search_calls"],
                            ctx["web"].backend, res["decision_ms"])
    _log_traj(ctx["traj"], task_id, res, res["correct"], sub=sub)
    return res


def _combine(task, family, parts, extra):
    """One scored record from several sub-questions (compare cells / chat turns)."""
    cost = {k: sum(p["cost"][k] for p in parts) for k in parts[0]["cost"]}
    rec = {"task_id": task["id"], "family": family, "answered": all(p["answered"] for p in parts),
           "correct": all(p["correct"] for p in parts),
           "part_accuracy": sum(p["correct"] for p in parts) / len(parts),
           "p": min(p["p"] for p in parts), "steps": sum(p["steps"] for p in parts),
           "web_calls": sum(p["web_calls"] for p in parts),
           "decision_ms": [m for p in parts for m in p["decision_ms"]],
           "invalid": sum(p["invalid"] for p in parts),
           "gold_reachable": all(p["reachable"] for p in parts),
           "serp_visible": all(p["serp_visible"] for p in parts), "cost": cost}
    rec.update(extra)
    return rec


def _run_fact_task(task, ctx):
    from src.planck3.harness import render_card
    r = _score_fact(ctx, task["question"], task["answer_type"], task["gold"], task["id"],
                    entity=task.get("entity"), attribute=task.get("attribute"))
    rec = {"task_id": task["id"], "family": "fact", "answered": r["answered"], "correct": r["correct"],
           "value": r["value"], "gold": task["gold"], "p": r["p"], "verified": r["verified"],
           "source_url": r["source_url"], "reason": r["reason"], "from_store": r["from_store"],
           "steps": r["steps"], "web_calls": r["web_calls"], "decision_ms": r["decision_ms"],
           "invalid": r["invalid"], "gold_reachable": r["reachable"], "serp_visible": r["serp_visible"],
           "cost": r["cost"]}
    tag = "OK" if r["correct"] else "WRONG" if r["answered"] else "abstain"
    return rec, f"### {task['id']} ({tag})\n\n{render_card(task['question'], r)}\n"


def _run_compare_task(task, ctx):
    from src.planck3.harness import render_table
    cells, table = [], {}
    for item in task["items"]:
        table[item] = {}
        for att in task["attributes"]:
            q = att["template"].format(item=item)
            c = _score_fact(ctx, q, att["answer_type"], task["gold"][item][att["name"]], task["id"],
                            entity=item, attribute=att["name"], sub=f"{item}/{att['name']}")
            c.update(item=item, attribute=att["name"])
            table[item][att["name"]] = c
            cells.append(c)
    rec = _combine(task, "compare", cells, {"cells": [
        {k: c[k] for k in ("item", "attribute", "value", "p", "source_url", "correct")} for c in cells]})
    return rec, f"### {task['id']} {task['question']}\n\n{render_table(task, {'table': table})}\n"


def _run_chat_task(task, ctx):
    """Multi-turn: answer type is INFERRED and follow-ups are rewritten, both scored."""
    from src.planck3.chat import Turn, chat_answer, infer_answer_type, resolve_followup
    prev, turns, lines = None, [], [f"### {task['id']} (chat)"]
    type_ok = rewrite_ok = 0
    for j, t in enumerate(task["turns"]):
        q, kind, ent = resolve_followup(t["user"], prev)
        at = infer_answer_type(q)
        type_ok += at == t["answer_type"]
        rewrite_ok += kind == t.get("rewrite", kind)
        r = _score_fact(ctx, q, at, t["gold"], task["id"], sub=f"turn{j}")
        turn = Turn(user=t["user"], question=q, answer_type=at, entity=ent, rewrite=kind, result=r)
        turns.append(r)
        prev = turn
        lines.append(f"**user:** {t['user']}  \n_(asked: {q} · {at} · {kind})_  \n"
                     f"{'OK' if r['correct'] else 'WRONG' if r['answered'] else 'abstain'}: {chat_answer(turn)}\n")
    rec = _combine(task, "chat", turns, {"type_accuracy": type_ok / len(task["turns"]),
                                         "rewrite_accuracy": rewrite_ok / len(task["turns"])})
    return rec, "\n".join(lines) + "\n"


def _gold_reachable(res, gold, answer_type) -> bool:
    """Was a correct value among the spans shown on ANY opened page? (tool ceiling)"""
    from src.planck3.metrics import is_correct
    for step in res["trajectory"]:
        for sp in step["obs"]["lists"].get("spans", []):
            if is_correct(sp["value"], gold, answer_type):
                return True
    return False


def _serp_visible(res, gold, answer_type, k=3) -> bool:
    """Search-only baseline: could the user read the answer off the top-k result snippets?"""
    from src.planck3.candidates import mentions, split_sentences
    from src.planck3.metrics import is_correct
    for step in res["trajectory"]:
        results = step["obs"]["lists"].get("results")
        if results:
            text = "\n".join(r.get("snippet", "") for r in results[:k])
            return any(is_correct(v, gold, answer_type)
                       for sent in split_sentences(text) for v, _ in mentions(sent, answer_type))
    return False


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
              "mean_web_calls", "mean_decision_ms", "n_invalid_decisions", "gold_reachable",
              "serp_answer_rate_at3"):
        v = s.get(k)
        print(f"  {k:<22} {v:.3f}" if isinstance(v, float) else f"  {k:<22} {v}")
    c = s.get("cost") or {}
    if c:
        per_ok = f"${c['usd_per_correct']:.5f}" if c.get("usd_per_correct") else "n/a"
        print(f"  cost/1k tasks          ${c['usd_per_1k_tasks']:.4f}   per correct {per_ok}   "
              f"LLM tokens/task {c['llm_tokens_per_task']:.0f} (priced as {c['priced_as']})")
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

    for name, fn in (("chat", cmd_chat), ("serve", cmd_serve)):
        p = sub.add_parser(name)
        p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8010)
        add_web_args(p)
        p.set_defaults(fn=fn)

    p = sub.add_parser("g0")
    p.add_argument("--tasks", default=str(TASKS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--store", default=None, help="reuse a store across runs (G3); default = fresh per run")
    p.add_argument("--family", default=None, help="comma list: fact,compare,chat")
    p.add_argument("--prices", default=str(REPO_ROOT / "config" / "planck3_prices.json"))
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
