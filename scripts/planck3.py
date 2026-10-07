"""
Planck 3.0 command line. On the Windows box, prefer the wrapper:

    .\\scripts\\planck3.ps1 <command>        (sets up venv/env, then calls this)

Commands:
    chat                              terminal chat with follow-ups (answer type inferred)
    serve                             local web chat at http://127.0.0.1:8010 (answers + depth + "For you")
    digest [--explore]                "for you" from your local knowledge graph; --explore reads adjacent
                                      entities from TRUSTED sources only (the continuous-retrieval loop)
    feedback source en.wikipedia.org up    thumbs up/down a source or entity (moves trust / interest)
    ask "question" --type year        one question -> answer card + decision trace
    g0  --policy gemma                G0: run the seed benchmark, write trajectories + gate verdict
    watch add|list|run                background watch tasks (read-only)
    search-check                      is SearXNG up? (auto falls back to Wikipedia search)
    wikirace build|embed|train|eval   G1 pipeline (see src/planck3/wikirace.py)
    doctor [--deep]                   preflight + ETA per stage; --deep loads Gemma/Planck and measures them
    report                            every G0/G1 result in one table -> results/planck3/REPORT.md

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
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "gemma", "bedrock", "planck"])
    p.add_argument("--g2-head", default=None, help="planck policy: results/planck3/g2_head_<enc>_s<seed>/head.pt")
    p.add_argument("--planck-checkpoint", default="checkpoints/planck13/best.pt")
    p.add_argument("--planck-tokenizer", default="data/wikipedia/tokenizer.model")
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")
    p.add_argument("--model-id", default=None, help="Bedrock model id (bedrock policy)")
    p.add_argument("--region", default=None)
    p.add_argument("--profile", default=None)


def make_policy(args):
    if args.policy == "planck":
        from src.planck3.g2 import PlanckPolicy
        if not args.g2_head:
            raise SystemExit("--policy planck needs --g2-head results/planck3/g2_head_<enc>_s<seed>/head.pt")
        return PlanckPolicy(args.g2_head, checkpoint=args.planck_checkpoint, tokenizer=args.planck_tokenizer)
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
    if args.type == "auto":
        from src.planck3.chat import infer_answer_type
        args.type = infer_answer_type(args.question)
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
    return lambda: Harness(policy, web, store, depth_pages=args.depth_pages)


def cmd_chat(args):
    from src.planck3.chat import ChatSession, chat_answer, depth_summary, render_depth
    sess = ChatSession(_chat_harness_factory(args)())
    print("Planck 3.0 chat. Follow up naturally ('and H&M?', 'when was it founded?'). "
          "'more' shows the evidence behind the last answer, 'new' resets, 'quit' exits.")
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
        if msg == "more":
            print(render_depth(sess.turns[-1]) if sess.turns else "(ask something first)")
            continue
        if not msg:
            continue
        t = sess.ask(msg)
        print(f"planck> {chat_answer(t)}")
        print(f"        (asked: \"{t.question}\" · {t.answer_type} · {t.rewrite} · "
              f"{t.result['steps']} decisions · {t.result['web_calls']} web calls)")
        print(f"        {depth_summary(t)}  (type 'more')")


def cmd_serve(args):
    from src.planck3.serve import make_server
    srv = make_server(_chat_harness_factory(args), host=args.host, port=args.port)
    print(f"Planck 3.0 chat on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_digest(args):
    from src.planck3.digest import build_digest, explore, render_digest
    from src.planck3.store import Store
    store = Store(args.store)
    if args.explore:
        added = explore(store, build_web(args), min_trust=args.min_trust)
        print(f"[planck3] explored {len(added)} adjacent sources from trusted domains")
        for a in added:
            print(f"  + {a['entity']} (via {a['via']}): {a['passages']} passages  {a['url']}")
    text = render_digest(build_digest(store, min_trust=args.min_trust))
    print(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "digest.md", "w", encoding="utf-8") as f:  # the scheduled task has no console
        f.write(text + "\n")


def cmd_feedback(args):
    from src.planck3.store import Store
    Store(args.store).feedback(args.kind, args.target, 1 if args.value == "up" else -1)
    print(f"recorded {args.value} for {args.kind} {args.target}")


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
    if args.sample:
        tasks = _stratified(tasks, args.sample)
    suffix = "_closedbook" if args.closed_book else ""
    run = args.out or str(RESULTS / f"g0_{args.policy}{suffix}_{datetime.now():%Y%m%d_%H%M}")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if args.closed_book:
        return _run_closed_book(args, tasks, run)
    store_path = Path(args.store) if args.store else run / "store.sqlite"
    if args.store is None and store_path.exists():
        store_path.unlink()  # a run's own store starts empty unless you pass --store (G3)
    web = build_web(args, log_path=run / "web_log.jsonl")
    store = Store(store_path)
    policy = make_policy(args)
    h = Harness(policy, web, store, snippet_first=not args.no_snippet_first,
                answer_threshold=args.answer_threshold, depth_pages=args.depth_pages)
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
        rec["regime"] = task.get("regime", "seed")
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
                 "depth_evidence_recall": sum(r["evidence_hit"] for r in records) / max(len(records), 1),
                 "snippet_first": not args.no_snippet_first, "answer_threshold": args.answer_threshold,
                 "mean_fetches": sum(r["fetch_calls"] for r in records) / max(len(records), 1),
                 "mean_answer_fetches": sum(r["answer_fetch_calls"] for r in records) / max(len(records), 1),
                 "depth_pages": args.depth_pages,
                 "answered_from_snippet": sum(r["from_snippet"] for r in records) / max(len(records), 1),
                 "gated_rate": sum(r["gated"] for r in records) / max(len(records), 1),
                 "gated_would_be_correct": sum(r["gated_correct"] for r in records),
                 "graph": store.graph_stats(),
                 "cost": summarize_cost(records, prices),
                 "web_calls_network": {k: v for k, v in web.calls.items()},
                 "wall_s": round(time.time() - t_start, 1), "store_facts": store.n_facts(),
                 "top_domains": store.top_domains(8)})
    summ["gate"] = _g0_verdict(summ)
    write_json(run / "summary.json", summ)
    with open(run / "cards.md", "w", encoding="utf-8") as f:
        f.write(f"# G0 answer cards ({args.policy})\n\n" + "\n".join(cards))
    _print_summary("G0", summ)
    _gzip(ctx["traj"])
    print(f"\n  run dir: {run}")


def _stratified(tasks, n):
    """Round-robin over families so a small sample still exercises fact, compare and chat."""
    by = {}
    for t in tasks:
        by.setdefault(t["family"], []).append(t)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by.values()):
        for fam in sorted(by):
            if i < len(by[fam]) and len(out) < n:
                out.append(by[fam][i])
        i += 1
    return out


def _gzip(path):
    """Compressed copy of the trajectories for git (the teacher data for G2 distillation)."""
    import gzip
    import shutil
    if Path(path).exists():
        with open(path, "rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
            shutil.copyfileobj(src, dst)


def _run_closed_book(args, tasks, run):
    """Base Claude/ChatGPT comparator: the same LLM, answering from its weights (no tools, no sources)."""
    from src.planck3.chat import infer_answer_type
    from src.planck3.cost import load_prices, summarize_cost, task_cost
    from src.planck3.metrics import is_correct, summarize
    policy = make_policy(args)
    if policy.kind != "llm":
        raise SystemExit("--closed-book needs an LLM policy (gemma or bedrock)")
    prices = load_prices(args.prices)
    res_path = run / "results.jsonl"
    if res_path.exists():
        res_path.unlink()

    def one(q, at, gold, history=None):
        u0 = dict(policy.usage)
        t0 = time.perf_counter()
        ans = policy.answer_closed_book(q, history)
        ms = (time.perf_counter() - t0) * 1000
        usage = {k: policy.usage[k] - u0.get(k, 0) for k in policy.usage}
        idk = not ans or any(x in ans.lower() for x in ("i don't know", "i do not know", "not sure", "unknown"))
        return {"q": q, "answer": ans, "answered": not idk, "correct": (not idk) and is_correct(ans, gold, at),
                "cost": task_cost(prices, "llm", usage, 0, "none", [ms]), "ms": ms}

    records, t_start = [], time.time()
    for i, task in enumerate(tasks, 1):
        if task["family"] == "fact":
            parts = [one(task["question"], task["answer_type"], task["gold"])]
        elif task["family"] == "compare":
            parts = [one(a["template"].format(item=it), a["answer_type"], task["gold"][it][a["name"]])
                     for it in task["items"] for a in task["attributes"]]
        else:  # chat: the base model gets the raw conversation, exactly like a chat product
            parts, hist = [], []
            for t in task["turns"]:
                r = one(t["user"], t["answer_type"] or infer_answer_type(t["user"]), t["gold"], list(hist))
                parts.append(r)
                hist.append((t["user"], r["answer"]))
        cost = {k: sum(p["cost"][k] for p in parts) for k in parts[0]["cost"]}
        rec = {"task_id": task["id"], "family": task["family"], "answered": all(p["answered"] for p in parts),
               "correct": all(p["correct"] for p in parts), "p": 1.0 if all(p["answered"] for p in parts) else 0.0,
               "steps": len(parts), "web_calls": 0, "decision_ms": [p["ms"] for p in parts], "cost": cost,
               "answers": [p["answer"] for p in parts], "regime": task.get("regime", "seed")}
        append_jsonl(res_path, rec)
        records.append(rec)
        print(f"[{i:>3}/{len(tasks)}] {'OK ' if rec['correct'] else 'XX '} {task['id']:<4} {' | '.join(rec['answers'])[:60]}")
    summ = summarize(records)
    summ.update({"policy": f"{args.policy} (closed-book)", "mode": "closed_book", "tasks": str(args.tasks),
                 "cost": summarize_cost(records, prices), "wall_s": round(time.time() - t_start, 1),
                 "note": "base-chat comparator: no tools, no sources, no freshness. 'p' is 1 when it answers "
                         "(a base chat reply carries no calibrated confidence), so ECE here measures overconfidence."})
    summ["gate"] = {"verdict": "COMPARATOR", "note": "compare success + cost per correct against the tool-using runs"}
    write_json(run / "summary.json", summ)
    _print_summary("G0 closed-book", summ)
    print(f"\n  run dir: {run}")


def _score_fact(ctx, question, answer_type, gold, task_id, entity=None, attribute=None, sub=None):
    """Run one fact question through the harness and attach scoring + cost + baselines."""
    from src.planck3.cost import task_cost
    from src.planck3.metrics import is_correct
    res = ctx["h"].run_fact(question, answer_type, entity=entity, attribute=attribute, task_id=task_id)
    res["correct"] = res["answered"] and is_correct(res["value"], gold, answer_type)
    res["reachable"] = _gold_reachable(res, gold, answer_type)
    res["serp_visible"] = _serp_visible(res, gold, answer_type)
    res["evidence_hit"] = _evidence_hit(res, gold, answer_type)
    res["gated"] = res["reason"] == "low_confidence"
    res["gated_correct"] = res["gated"] and is_correct(res.get("gated_value"), gold, answer_type)
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
           "serp_visible": all(p["serp_visible"] for p in parts),
           "evidence_hit": all(p["evidence_hit"] for p in parts),
           "n_passages": sum(len(p["depth"]["passages"]) for p in parts), "cost": cost,
           "fetch_calls": sum(p["fetch_calls"] for p in parts),
           "answer_fetch_calls": sum(p["answer_fetch_calls"] for p in parts),
           "from_snippet": all(p["from_snippet"] for p in parts),
           "gated": any(p["gated"] for p in parts), "gated_correct": any(p["gated_correct"] for p in parts)}
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
           "evidence_hit": r["evidence_hit"], "n_passages": len(r["depth"]["passages"]),
           "fetch_calls": r["fetch_calls"], "answer_fetch_calls": r["answer_fetch_calls"],
           "from_snippet": r["from_snippet"],
           "gated": r["gated"], "gated_correct": r["gated_correct"], "gated_value": r.get("gated_value"),
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


def _evidence_hit(res, gold, answer_type) -> bool:
    """Depth quality: does the evidence pack behind the answer actually contain the gold value?"""
    from src.planck3.candidates import mentions
    from src.planck3.metrics import is_correct
    for p in (res.get("depth") or {}).get("passages", []):
        if any(is_correct(v, gold, answer_type) for v, _ in mentions(p["text"], answer_type)):
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
    if "fresh" in str(s.get("tasks", "")):
        return {"verdict": "ROUND 3 (rules A/B)", "note": "fresh + long-tail benchmark: judged by SETUP_planck_20260903.md section 3"}
    if s["policy"] == "planck":
        return {"verdict": "G2 (rule B)", "note": "the learned policy: judged by SETUP_planck_20260903.md section 3B"}
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
              "serp_answer_rate_at3", "depth_evidence_recall", "mean_fetches", "mean_answer_fetches", "answered_from_snippet",
              "gated_rate", "gated_would_be_correct"):
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


def cmd_doctor(args):
    from src.planck3.doctor import run
    sys.exit(run(args.deep, args.checkpoint, args.tokenizer, args.gemma_path, args.searxng_url))


def cmd_report(args):
    """One table for every run under results/planck3/, written to REPORT.md (committed by the runner)."""
    lines = [f"# Planck 3.0 results ({datetime.now():%Y-%m-%d %H:%M})", ""]
    g0 = sorted(RESULTS.glob("g0*_*/summary.json"))  # g0_ = seed benchmark, g0f_ = fresh + long-tail
    if g0:
        lines += ["## G0 (seed benchmark)", "",
                  "| run | n | success | answered | wrong when answered | ECE | depth evidence | search-only @3 | answer fetches/task | from snippet | gated | $/correct | LLM tok/task | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = {}
        for p in g0:
            s = read_json(p)
            rows[p.parent.name] = s
            c = s.get("cost") or {}
            f = lambda k: "" if s.get(k) is None else f"{s[k]:.3f}"  # noqa: E731
            per_ok = "" if not c.get("usd_per_correct") else f"{c['usd_per_correct']:.6f}"
            lines.append(f"| {p.parent.name} | {s.get('n')} | {f('success')} | {f('answered_rate')} | "
                         f"{f('wrong_when_answered')} | {f('ece')} | {f('depth_evidence_recall')} | "
                         f"{f('serp_answer_rate_at3')} | {f('mean_answer_fetches') or f('mean_web_calls')} | {f('answered_from_snippet')} | "
                         f"{f('gated_rate')} | {per_ok} | {c.get('llm_tokens_per_task', 0):.0f} | "
                         f"{s.get('gate', {}).get('verdict', '')} |")
        lines += ["", *_vs_rival(rows), ""]
        reg_lines = []
        for name, s in rows.items():
            br = s.get("by_regime") or {}
            if any(k != "seed" for k in br):
                reg_lines.append(f"| {name} | " + " | ".join(
                    f"{k}: {v['success']:.3f} (n={v['n']}, wrong {v['wrong_when_answered']:.3f})" for k, v in sorted(br.items())) + " |")
        if reg_lines:
            lines += ["### By regime (fresh = 2026 facts; long_tail = <= 3 Wikipedia editions)", "",
                      "| run | per regime: success (n, wrong when answered) |", "|---|---|", *reg_lines, ""]
    g2 = sorted(RESULTS.glob("g2_head_*/train_log.json"))
    if g2:
        lines += ["## G2 heads (learned from known answers)", "",
                  "| head | train points | val points with a right candidate | learned choice accuracy | deterministic ranker | 'none' correct (points without one) | val ECE | answer accuracy when p >= 0.5 |",
                  "|---|---|---|---|---|---|---|---|"]
        for p in g2:
            t = read_json(p)
            lines.append(f"| {p.parent.name} | {t['n_train_points']} | {t.get('val_points_with_gold', '')} | "
                         f"{t.get('val_choice_acc', 0):.3f} | {t['val_heuristic_top1']:.3f} | "
                         f"{t.get('val_none_acc', 0):.3f} ({t.get('val_points_without_gold', '')}) | {t['val_ece']:.3f} | "
                         f"{t['val_answer_acc_at_0.5']:.3f} |")
        lines.append("")
    g1 = sorted(RESULTS.glob("g1_eval_*/summary.json"))
    for p in g1:
        s = read_json(p)
        lines += [f"## G1 Wikiracing ({p.parent.name})", "",
                  "| policy | pairs | rollout success | steps / optimal | step top-1 | step ECE | ms / decision |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in s["results"].items():
            if "rollout_success" not in r:
                continue
            g = lambda k: "" if r.get(k) is None else f"{r[k]:.3f}"  # noqa: E731
            lines.append(f"| {name} | {r['n_pairs']} | {g('rollout_success')} | {g('mean_steps_over_optimal')} | "
                         f"{g('step_top1')} | {g('step_ece')} | {g('median_decision_ms')} |")
        cold = s["results"].get("latency_cpu_cold")
        if cold:
            lines.append(f"\nCold CPU latency (encode target + all candidate titles): median {cold['median_ms']:.0f} ms, "
                         f"p90 {cold['p90_ms']:.0f} ms, {cold['mean_candidates']:.0f} candidates on average")
        if s.get("paired"):
            lines += ["", "| paired comparison (same races) | n | head | other | ratio | only head wins | only other wins | McNemar p |",
                      "|---|---|---|---|---|---|---|---|"]
            for k, v in s["paired"].items():
                other = v.get("teacher", v.get("control"))
                ratio = "" if v.get("ratio") is None else f"{v['ratio']:.2f}"
                lines.append(f"| {k} | {v['n']} | {v['head']:.3f} | {other:.3f} | {ratio} | {v['a_only']} | {v['b_only']} | {v['p']:.3g} |")
        lines += ["", f"**Verdict:** {s['gate']['verdict']}. {s['gate'].get('note', '')}", ""]
    for p in sorted(RESULTS.glob("g1_aggregate*/summary.json")):
        s = read_json(p)
        lines += [f"## G1 across seeds ({p.parent.name}: seeds {s['seeds']})", "",
                  "| policy | mean rollout success | sd | 95% CI | per seed |", "|---|---|---|---|---|"]
        for pol, t in s["policies"].items():
            ci = f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]" if t["ci95"] else ""
            lines.append(f"| {pol} | {t['mean']:.3f} | {t['sd']:.3f} | {ci} | {', '.join(f'{x:.3f}' for x in t['per_seed'])} |")
        for arm, r in s.get("paired_ratio_vs_teacher", {}).items():
            lines.append(f"\n{arm}: paired ratio vs teacher {r['mean']:.2f} (per seed {', '.join(f'{x:.2f}' for x in r['per_seed'])})")
        lines += ["", f"**Verdict (seeds):** {s['gate']['verdict']}. {s['gate']['note']}", ""]
    if not g0 and not g1:
        lines.append("No runs yet.")
    text = "\n".join(lines)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def _vs_rival(rows):
    """Headline: the product (tool-using run) against the base-chat rival (closed-book)."""
    out = []
    for name, s in rows.items():
        if "_closedbook" in name or s.get("mode") == "closed_book":
            continue
        pol = str(s.get("policy", "")).split(" ")[0]
        prefix = name.split("_")[0]  # g0 = seed benchmark, g0f = fresh + long-tail benchmark
        rival = rows.get(f"{prefix}_{pol}_closedbook" + ("_quick" if "_quick" in name else "")) or \
            next((r for n, r in rows.items() if n.startswith(f"{prefix}_") and "_closedbook" in n
                  and ("_quick" in n) == ("_quick" in name)), None)
        if not rival:
            continue
        a, b = s.get("cost") or {}, rival.get("cost") or {}
        ratio = (b["usd_per_correct"] / a["usd_per_correct"]) if a.get("usd_per_correct") and b.get("usd_per_correct") else None
        if ratio is None:
            cost_txt = "n/a"
        elif ratio >= 1:
            cost_txt = f"{ratio:.1f}x cheaper"
        else:
            cost_txt = (f"{1 / ratio:.1f}x MORE expensive ({a.get('llm_tokens_per_task', 0):.0f} vs "
                        f"{b.get('llm_tokens_per_task', 0):.0f} LLM tokens/task)")
        teacher = (s.get("cost") or {}).get("llm_tokens_per_task", 0) > 0
        note = (" An LLM driving the tools is the expensive path by design; the cheap path is the distilled "
                "Planck policy (G2), which this run does not include.") if teacher else ""
        out.append(f"**{name} vs base chat:** success {s['success']:.3f} vs {rival['success']:.3f} "
                   f"({s['success'] / max(rival['success'], 1e-9):.2f}x); depth evidence {s.get('depth_evidence_recall', 0):.3f} "
                   f"vs none (base chat shows no sources); cost per correct {cost_txt}.{note}")
    return out


def main():
    utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--type", default="auto", choices=["auto", "year", "number", "date", "entity", "text"],
                   help="answer type; auto = inferred from the question (default)")
    p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
    p.add_argument("-v", "--verbose", action="store_true")
    add_web_args(p)
    p.set_defaults(fn=cmd_ask)

    for name, fn in (("chat", cmd_chat), ("serve", cmd_serve)):
        p = sub.add_parser(name)
        p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8010)
        p.add_argument("--depth-pages", type=int, default=1,
                       help="extra sources read after answering, for the depth pack (0 = only pages already read)")
        add_web_args(p)
        p.set_defaults(fn=fn)

    p = sub.add_parser("digest", help="'for you' digest from your local knowledge graph")
    p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
    p.add_argument("--explore", action="store_true", help="also read adjacent entities from trusted sources (web)")
    p.add_argument("--min-trust", type=float, default=0.6)
    add_web_args(p)
    p.set_defaults(fn=cmd_digest)

    p = sub.add_parser("feedback", help="thumbs up/down a source domain or an entity")
    p.add_argument("kind", choices=["source", "entity"])
    p.add_argument("target")
    p.add_argument("value", choices=["up", "down"])
    p.add_argument("--store", default=str(RESULTS / "personal_store.sqlite"))
    p.set_defaults(fn=cmd_feedback)

    p = sub.add_parser("g0")
    p.add_argument("--tasks", default=str(TASKS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--store", default=None, help="reuse a store across runs (G3); default = fresh per run")
    p.add_argument("--family", default=None, help="comma list: fact,compare,chat")
    p.add_argument("--sample", type=int, default=0, help="stratified subset of N tasks across families (quick runs)")
    p.add_argument("--no-snippet-first", action="store_true", help="pages-first (the 2026-10-05 baseline behaviour)")
    p.add_argument("--depth-pages", type=int, default=1,
                   help="pages read AFTER answering for the depth pack (product setting: answer fast, then 1 page)")
    p.add_argument("--answer-threshold", type=float, default=0.3,
                   help="ANSWER with p below this abstains instead (0 disables the gate)")
    p.add_argument("--closed-book", action="store_true",
                   help="base-chat comparator: the LLM answers from its weights, no tools/sources")
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

    p = sub.add_parser("doctor", help="preflight: environment, models, network, ETA per stage")
    p.add_argument("--deep", action="store_true", help="load Gemma + Planck and measure them (~2 min)")
    p.add_argument("--checkpoint", default="checkpoints/planck13/best.pt")
    p.add_argument("--tokenizer", default="data/wikipedia/tokenizer.model")
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")
    p.add_argument("--searxng-url", default="http://localhost:8888")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    from src.planck3 import g2, wikirace
    wikirace.add_cli(sub)
    g2.add_cli(sub, add_web_args, build_web)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
