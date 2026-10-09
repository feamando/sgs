"""
Round 5 CLI (SETUP_planck_20261009.md): the read -> weigh -> write pipeline on three benchmarks.

    r5 --reader heuristic|planck|gemma --writer template|hertz|planck|gemma --tasks a.json,b.json
        evaluates the answer pipeline: accuracy of the value shown, precision per confidence band,
        writer faithfulness and fallbacks, latency per stage, LLM tokens; --personal contrarian runs the
        worst-case personal profile (every source scored 10 - system) to bound the echo-chamber effect
    writer collect   Gemma writes the answer text for the G2 TRAINING questions (never a benchmark);
                     faithful examples become the one-time training set for the small writers
    writer train --base hertz|planck   fine-tune once (answer tokens only), best by validation loss
    latency          the runner repeats one r5 run on CPU only, 4 threads (a phone proxy):
                     CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 ... r5 --sample 20
"""

import json
import time
from pathlib import Path

from .metrics import is_correct
from .util import REPO_ROOT, append_jsonl, read_json, read_jsonl, write_json

RESULTS = REPO_ROOT / "results" / "planck3"
WRITER_DATA = REPO_ROOT / "data" / "planck3" / "writer" / "train.jsonl"
BANDS = ("high", "good", "low", "sceptical", "none")
QTYPE_OF_REGIME = {"fresh": "news", "long_tail": "encyclopedic", "news": "news", "general_info": "general_info"}


def build_answerer(args, build_web, store_path=None, personal_override=None, store_answer=True):
    from .answer import Answerer
    from .readers import make_reader
    from .store import Store
    from .writer import make_writer
    web = build_web(args)
    store = Store(store_path or args.store)
    reader = make_reader(args.reader, head=args.g2_head, checkpoint=args.planck_checkpoint,
                         tokenizer=args.planck_tokenizer, gemma_path=args.gemma_path)
    shared = getattr(reader, "policy", None) if args.reader == args.writer == "gemma" else None
    if shared is not None:  # one Gemma in memory for both roles
        from .writer import LLMWriter
        writer = LLMWriter(shared)
    else:
        writer = make_writer(args.writer, checkpoint=args.writer_checkpoint, gemma_path=args.gemma_path)
    return Answerer(web, store, reader, writer, personal_override=personal_override, store_answer=store_answer), web


def _turns(task):
    if task["family"] == "fact":
        return [(task["question"], task["answer_type"], task["gold"])]
    return [(t["user"], t.get("answer_type"), t["gold"]) for t in task["turns"]]


def run_eval(args, build_web):
    from .chat import Turn, infer_answer_type, resolve_followup
    from .cost import load_prices
    tasks = []
    for path in args.tasks.split(","):
        spec = read_json(path)
        tasks += [t for t in spec["tasks"] if t["family"] in ("fact", "chat")]
    if args.limit:
        tasks = tasks[: args.limit]
    if args.sample:
        step = max(1, len(tasks) // args.sample)
        tasks = tasks[::step][: args.sample]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    override = (lambda dom, s: 10.0 - s) if args.personal == "contrarian" else None
    store_path = out / "store.sqlite"
    if store_path.exists():
        store_path.unlink()
    ans, web = build_answerer(args, build_web, store_path=store_path, personal_override=override)
    res_path = out / "results.jsonl"
    if res_path.exists():
        res_path.unlink()
    prices = load_prices(args.prices)
    records = []
    t_start = time.time()
    for i, task in enumerate(tasks, 1):
        prev, parts = None, []
        for q_user, at, gold in _turns(task):
            if task["family"] == "chat":
                q, kind, ent = resolve_followup(q_user, prev)
                at = infer_answer_type(q)
            else:
                q = q_user
            u0 = dict(ans.reader.usage)
            w0 = dict(getattr(ans.writer, "usage", {}))
            r = ans.run(q, at, task_id=task["id"])
            ex = r["explain"]
            tin = ans.reader.usage.get("input_tokens", 0) - u0.get("input_tokens", 0)
            tout = ans.reader.usage.get("output_tokens", 0) - u0.get("output_tokens", 0)
            if getattr(ans.writer, "usage", None) is not None and ans.writer.usage is not ans.reader.usage:
                tin += ans.writer.usage.get("input_tokens", 0) - w0.get("input_tokens", 0)
                tout += ans.writer.usage.get("output_tokens", 0) - w0.get("output_tokens", 0)
            toks = {"in": tin, "out": tout}
            parts.append({"q": q, "value": r["value"], "band": ex["band"], "confidence": ex["confidence10"],
                          "correct": is_correct(r["value"], gold, at), "text": r["answer_text"],
                          "faithful": ex["writer_faithful"], "fallback": ex["writer_fallback"],
                          "problems": ex["writer_problems"], "qtype": ex["qtype"], "timing": r["timing"],
                          "tokens": toks, "divergence": ex["divergence"], "conflicts": len(ex["conflicts"]),
                          "search_calls": r["search_calls"], "fetch_calls": r["fetch_calls"],
                          "items": ex["items"], "sources": [s["domain"] for s in ex["sources"]],
                          "top_source_scored": ex["sources"][0]["trust"]["scored"] if ex["sources"] else None})
            prev = Turn(user=q_user, question=q, answer_type=at, entity=None, rewrite="NEW", result=r)
            if task["family"] == "chat":
                from .candidates import main_entity
                prev.entity = main_entity(q)
        worst = min(parts, key=lambda p: BANDS[::-1].index(p["band"]))
        rec = {"task_id": task["id"], "family": task["family"], "regime": task.get("regime", "seed"),
               "qtype_gold": task.get("qtype") or QTYPE_OF_REGIME.get(task.get("regime")),
               "gold": task.get("gold"), "value": parts[0]["value"] if len(parts) == 1 else [p["value"] for p in parts],
               "shown_correct": all(p["correct"] for p in parts), "band": worst["band"], "confidence": worst["confidence"],
               "faithful": all(p["faithful"] for p in parts), "fallback": any(p["fallback"] for p in parts),
               "qtype": parts[0]["qtype"], "tokens": {k: sum(p["tokens"][k] for p in parts) for k in ("in", "out")},
               "changed_by_personal": any(p["divergence"] for p in parts),
               "timing": {k: sum(p["timing"].get(k, 0.0) for p in parts) for k in parts[0]["timing"]},
               "search_calls": sum(p["search_calls"] for p in parts), "fetch_calls": sum(p["fetch_calls"] for p in parts),
               "parts": parts}
        append_jsonl(res_path, rec)
        records.append(rec)
        mark = "OK " if rec["shown_correct"] else ("-- " if rec["band"] == "none" else "XX ")
        print(f"[{i:>3}/{len(tasks)}] {mark} {task['id']:<12} {str(rec['value'])[:30]:<30} {rec['band']:<9} "
              f"{rec['confidence'] if rec['confidence'] is not None else '-'}/10  {'' if rec['faithful'] else 'UNFAITHFUL->template'}",
              flush=True)
    summ = summarize_r5(records)
    rate = prices["llm_per_mtok"][prices["llm_equivalent"]]
    summ.update({"reader": args.reader, "writer": ans.writer.name, "personal": args.personal, "tasks": args.tasks,
                 "search": web.backend, "search_health": web.search_health() if hasattr(web, "search_health") else None,
                 "wall_s": round(time.time() - t_start, 1),
                 "llm_usd_per_task_as_haiku": sum(r["tokens"]["in"] * rate["input"] + r["tokens"]["out"] * rate["output"]
                                                  for r in records) / 1e6 / max(len(records), 1)})
    write_json(out / "summary.json", summ)
    _print(summ)


def summarize_r5(records: list[dict]) -> dict:
    n = max(len(records), 1)
    bands = {}
    for b in BANDS:
        sub = [r for r in records if r["band"] == b]
        bands[b] = {"n": len(sub), "share": len(sub) / n,
                    "precision": (sum(r["shown_correct"] for r in sub) / len(sub)) if sub and b != "none" else None}
    prec = [bands[b]["precision"] for b in BANDS[:-1] if bands[b]["precision"] is not None and bands[b]["n"] >= 5]
    by_reg = {}
    for reg in sorted({r["regime"] for r in records}):
        sub = [r for r in records if r["regime"] == reg]
        by_reg[reg] = {"n": len(sub), "accuracy": sum(r["shown_correct"] for r in sub) / len(sub)}
    with_gold_qt = [r for r in records if r.get("qtype_gold")]
    model_written = [p for r in records for p in r["parts"] if p["band"] != "none"]
    t_keys = records[0]["timing"].keys() if records else []
    return {"n": len(records), "accuracy": sum(r["shown_correct"] for r in records) / n, "bands": bands,
            "bands_monotone": all(a >= b for a, b in zip(prec, prec[1:])), "by_regime": by_reg,
            "qtype_accuracy": (sum(r["qtype"] == r["qtype_gold"] for r in with_gold_qt) / len(with_gold_qt)) if with_gold_qt else None,
            "writer_faithful": (sum(p["faithful"] for p in model_written) / len(model_written)) if model_written else None,
            "writer_fallback": (sum(p["fallback"] for p in model_written) / len(model_written)) if model_written else None,
            "changed_by_personal": sum(r["changed_by_personal"] for r in records) / n,
            # registry coverage: how often the source an answer rests on was scored by hand (else 5, "unscored")
            "registry_coverage": (lambda ps: sum(ps) / len(ps) if ps else None)(
                [p["top_source_scored"] for r in records for p in r["parts"] if p.get("top_source_scored") is not None]),
            "tokens_per_task": sum(r["tokens"]["in"] + r["tokens"]["out"] for r in records) / n,
            "timing_ms_per_task": {k: sum(r["timing"][k] for r in records) / n for k in t_keys},
            "mean_search_calls": sum(r["search_calls"] for r in records) / n,
            "mean_fetch_calls": sum(r["fetch_calls"] for r in records) / n}


def _print(s):
    print(f"\n==== r5 summary: reader={s['reader']} writer={s['writer']} personal={s['personal']} search={s['search']} ====")
    print(f"  n {s['n']}  accuracy (value shown is right) {s['accuracy']:.3f}  bands monotone {s['bands_monotone']}")
    for b, d in s["bands"].items():
        pr = "" if d["precision"] is None else f" precision {d['precision']:.3f}"
        print(f"  band {b:<10} {d['share']:.3f} (n={d['n']}){pr}")
    for reg, d in s["by_regime"].items():
        print(f"  regime {reg:<13} accuracy {d['accuracy']:.3f} (n={d['n']})")
    print(f"  writer faithful {s['writer_faithful']}  fallback {s['writer_fallback']}  qtype accuracy {s['qtype_accuracy']}")
    print(f"  changed by personal profile {s['changed_by_personal']:.3f}  LLM tokens/task {s['tokens_per_task']:.0f}")
    print(f"  ms/task {json.dumps({k: round(v) for k, v in s['timing_ms_per_task'].items()})}")


# ── writer data + training ─────────────────────────────────────────────────
def collect_writer_data(args, build_web):
    """Gemma writes the answer text for TRAINING questions (disjoint from every benchmark)."""
    from .answer import Answerer
    from .readers import make_reader
    from .store import Store
    from .writer import LLMWriter, TemplateWriter, faithful, writer_prompt, writer_sources
    from .policies import make_policy
    tasks = [t for t in read_json(args.tasks)["tasks"] if t["family"] == "fact"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [t for t in tasks if t["id"] not in done][: args.limit or None]
    print(f"[writer] collecting {len(todo)} examples ({len(done)} done) -> {out}")
    web = build_web(args)
    reader = make_reader(args.reader, head=args.g2_head, checkpoint=args.planck_checkpoint,
                         tokenizer=args.planck_tokenizer, gemma_path=args.gemma_path)
    gem = LLMWriter(make_policy("gemma", gemma_path=args.gemma_path))
    ans = Answerer(web, Store(out.parent / "collect_store.sqlite"), reader, TemplateWriter(), store_answer=False)
    kept = 0
    for i, t in enumerate(todo, 1):
        r = ans.run(t["question"], t["answer_type"], task_id=t["id"])
        cons = {"value": r["explain"]["value"], "confidence": r["explain"]["confidence10"], "band": r["explain"]["band"],
                "families": [], "conflicts": r["explain"]["conflicts"]}
        if cons["value"] is None:
            append_jsonl(out, {"task_id": t["id"], "skipped": "no answer"})
            continue
        sources = r["writer_sources"]
        prompt = writer_prompt(t["question"], cons, sources)
        text = gem.generate(prompt) or ""
        ok, probs = faithful(text, t["question"], cons["value"], sources, t["answer_type"])
        kept += ok
        append_jsonl(out, {"task_id": t["id"], "question": t["question"], "band": cons["band"], "prompt": prompt,
                           "target": text, "faithful": ok, "problems": probs,
                           "value_correct": is_correct(cons["value"], t["gold"], t["answer_type"])})
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} ({kept} faithful)", flush=True)
    print(f"[writer] {kept}/{len(todo)} new examples faithful -> {out}")


def rescore(out_dir: Path, task_files: str):
    """Re-judge a finished run against (corrected) gold answers: no search, no model, values as recorded."""
    gold = {}
    for path in task_files.split(","):
        for t in read_json(path)["tasks"]:
            gold[t["id"]] = t
    recs = read_jsonl(out_dir / "results.jsonl")
    for r in recs:
        t = gold.get(r["task_id"])
        if not t:
            continue
        golds = [(t["gold"], t["answer_type"])] if t["family"] == "fact" else [(x["gold"], x.get("answer_type")) for x in t["turns"]]
        for p_, (g, at) in zip(r["parts"], golds):
            p_["correct"] = is_correct(p_["value"], g, at or "entity")
        r["shown_correct"] = all(p_["correct"] for p_ in r["parts"])
        r["gold"] = t.get("gold")
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = read_json(out_dir / "summary.json")
    summ.update(summarize_r5(recs))
    summ["rescored"] = time.strftime("%Y-%m-%d %H:%M")
    write_json(out_dir / "summary.json", summ)
    print(f"[r5] rescored {out_dir.name}: accuracy {summ['accuracy']:.3f}")


def report_lines(results: Path) -> list[str]:
    """REPORT.md section: one row per r5 run + paired comparisons on the same questions."""
    from .wikirace import mcnemar
    runs = {p.parent.name: read_json(p) for p in sorted(results.glob("r5_*/summary.json"))}
    if not runs:
        return []
    f3 = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
    lines = ["## Round 5: read -> weigh -> write (SETUP_planck_20261009.md)", "",
             "| run | n | value shown is right | high: n / precision | good | low | sceptical | none | monotone | writer faithful | fallback | LLM tok/task | read ms | write ms | changed by personal | question type acc |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in runs.items():
        b = s["bands"]
        cell = lambda k: f"{b[k]['n']} / {f3(b[k]['precision'])}"  # noqa: E731
        t = s.get("timing_ms_per_task", {})
        lines.append(f"| {name} | {s['n']} | {s['accuracy']:.3f} | {cell('high')} | {cell('good')} | {cell('low')} | {cell('sceptical')} | "
                     f"{b['none']['n']} | {s['bands_monotone']} | {f3(s['writer_faithful'])} | {f3(s['writer_fallback'])} | "
                     f"{s['tokens_per_task']:.0f} | {t.get('read_ms', 0):.0f} | {t.get('write_ms', 0):.0f} | {s['changed_by_personal']:.3f} | "
                     f"{f3(s.get('qtype_accuracy'))} |")
    regs = [f"| {n} | " + " | ".join(f"{k}: {v['accuracy']:.3f} (n={v['n']})" for k, v in s["by_regime"].items()) + " |"
            for n, s in runs.items()]
    lines += ["", "| run | accuracy per benchmark regime |", "|---|---|", *regs, ""]

    def recs(name):
        return {r["task_id"]: r for r in read_jsonl(results / name / "results.jsonl")}
    prod = next((n for n in ("r5_planck_hertz", "r5_planck_planck") if n in runs), None)   # the small-model product
    pairs = [(a, b) for a in runs for b in runs if prod and a != b and (
        (a == prod and b in ("r5_gemma_gemma", "r5_heuristic_template", "r5_planck_template", "r5_planck_planck"))
        or (a.startswith(prod + "_") and b == prod))]
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
    for name in ("writer_hertz", "writer_planck"):
        p_ = results / name / "train_log.json"
        if p_.exists():
            t = read_json(p_)
            lines.append(f"- {name}: {t['examples']} training examples ({t['train']} train / {t['val']} val), "
                         f"best val loss {t['best_val_loss']:.3f}, {t['params'] / 1e6:.0f}M params")
    return lines + [""]


def add_cli(sub, add_web_args, build_web):
    p = sub.add_parser("r5", help="round 5: read -> weigh -> write pipeline evaluation")
    p.add_argument("--tasks", default=str(REPO_ROOT / "scripts" / "assets" / "planck3_tasks_fresh.json"),
                   help="comma-separated task files")
    p.add_argument("--reader", default="heuristic", choices=["heuristic", "planck", "gemma", "bedrock"])
    p.add_argument("--writer", default="template", choices=["template", "hertz", "planck", "gemma", "bedrock"])
    p.add_argument("--writer-checkpoint", default=None)
    p.add_argument("--personal", default="none", choices=["none", "contrarian"])
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--store", default=None)
    p.add_argument("--prices", default=str(REPO_ROOT / "config" / "planck3_prices.json"))
    add_web_args(p)
    p.add_argument("--rescore", action="store_true", help="re-judge --out against --tasks gold (no search, no model)")
    p.set_defaults(fn=lambda a: rescore(Path(a.out), a.tasks) if a.rescore else run_eval(a, build_web))

    p = sub.add_parser("writer", help="the small writers: collect training text (Gemma) | train once")
    p.add_argument("stage", choices=["collect", "train"])
    p.add_argument("--tasks", default=str(REPO_ROOT / "scripts" / "assets" / "planck3_g2_train_tasks.json"))
    p.add_argument("--out", default=str(WRITER_DATA))
    p.add_argument("--reader", default="planck", choices=["heuristic", "planck"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--base", default="hertz", choices=["hertz", "planck"])
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--base-tokenizer", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8, help="Hertz (640M, full fine-tune): 4 on a 24 GB card")
    p.add_argument("--tag", default="", help="suffix for checkpoints/writer_<base><tag> (quick runs: _quick)")
    add_web_args(p)

    def run_writer(a):
        if a.stage == "collect":
            return collect_writer_data(a, build_web)
        from .writer import train_writer
        ck = a.base_checkpoint or ("checkpoints/hertz/best.pt" if a.base == "hertz" else "checkpoints/planck13/best.pt")
        tok = a.base_tokenizer or ("data/hertz12_data/tokenizer.model" if a.base == "hertz" else "data/wikipedia/tokenizer.model")
        meta = train_writer(Path(a.out), ck, tok, REPO_ROOT / "checkpoints" / f"writer_{a.base}{a.tag}", a.base,
                            epochs=a.epochs, batch_size=a.batch_size)
        write_json(RESULTS / f"writer_{a.base}{a.tag}" / "train_log.json", meta)
        print(f"[writer] {a.base}: {meta['examples']} examples, best val loss {meta['best_val_loss']:.3f}")
    p.set_defaults(fn=run_writer)
