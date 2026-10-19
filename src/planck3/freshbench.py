"""
Round 9's fresh benchmark (SETUP_planck_20261019.md): new questions, never used to choose anything.

Rounds 6-8 measured eight readers on the same 243 questions and picked Hertz's reading reader out of
four arms on them, so its 60.1% is a selected number. Round 9 confirms it on questions nobody has
looked at, built the same two ways as the old ones but from items not used before:

    wikidata  the fresh benchmark's 2026 templates (event / election / season winners, new heads of
              government), on 2026 items that are in NO earlier benchmark or training set (by QID),
              and not named in a benchmark question. Gold = Wikidata label + aliases (no model)
    news      Wikipedia's Current Events portal, 2026 days AFTER the old news benchmark's last event
              (2026-10-08), up to 15 items a day across its sections; Gemma turns each item into one question
              whose short answer the item states, through round 6's checks. Gold = the item's own words

Built ONCE, on the box, before any reader runs on it; the file is committed with its build manifest
(seed, counts, drops), so it is fixed before the first result exists. Readers never see it in training.
"""

import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from . import benchgen as bg
from .util import REPO_ROOT, read_json, write_json

FRESH9 = REPO_ROOT / "scripts" / "assets" / "planck3_tasks_fresh9.json"
NEWS9_RAW = REPO_ROOT / "data" / "planck3" / "fresh9_news_raw.json"
TRAIN_FILES = ("planck3_g2_train_tasks.json", "planck3_read_train_tasks.json")
NEWS_AFTER = date(2026, 10, 8)          # the old news benchmark's last event date
WD_PER = {"winner": 60, "election": 20, "season": 30, "headgov": 30}
NEWS_TARGET = 80


def used_qids_and_texts() -> tuple[set, list[str]]:
    """Every QID in a benchmark OR a training set, and the benchmark question texts (name check)."""
    from .readtasks import ASSETS, _bench
    qids, texts = _bench()
    for f in TRAIN_FILES:
        p = ASSETS / f
        if p.exists():
            qids |= {t.get("qid") for t in read_json(p)["tasks"]} | {q for t in read_json(p)["tasks"] for q in t.get("qids", [])}
    return qids - {None}, texts


def wikidata_fresh(seed: int = 9) -> tuple[list[dict], dict]:
    from .readtasks import named_in_bench
    rng = random.Random(seed)
    used, texts = used_qids_and_texts()
    tasks, stats = [], {}
    specs = (("winner", bg.q_event_winners(2026, 2027, 2500), lambda l, rs: f"Who won the {l}?"),
             ("election", bg.q_elections(2026, 2027, 2500), lambda l, rs: f"Who won the {l}?"),
             ("season", bg.q_season_champions(2026, 2027, 2500), lambda l, rs: f"Who won the {l}?"),
             ("headgov", bg.q_head_of_gov(2026, 2500),
              lambda l, rs: f"Who became head of government of {l} in {rs[0]['start'][:4]}?"))
    for name, query, qfn in specs:
        items = list(bg._group(bg.sparql(query)).items())
        rng.shuffle(items)
        st = Counter(pool=len(items))
        k = 0
        for uri, rs in items:
            if k >= WD_PER[name]:
                break
            label = rs[0].get("itemLabel", "")
            if not label or re.fullmatch(r"Q\d+", label):
                st["no label"] += 1
                continue
            if bg._qid(uri) in used:
                st["used before"] += 1
                continue
            if named_in_bench(label, texts):
                st["named in a benchmark question"] += 1
                continue
            if re.search(r"\b(Cabinet|Government|Ministry|Council of)\b", label):
                st["not a place"] += 1
                continue
            gold = bg._golds(rs)
            if not gold or not gold[0][:1].isupper():
                st["no proper-name answer"] += 1
                continue
            tasks.append(bg._task(f"f9{name}{k:03d}", "fresh", name, qfn(label, rs), "entity", gold, rs))
            k += 1
        st["kept"] = k
        stats[name] = dict(st)
        print(f"  [fresh9] wikidata {name:<9} kept {k:>3} (pool {len(items)}; {dict(st)})", flush=True)
    return tasks, stats


def portal_items_after(after: date, until: date, per_day: int = 15, seed: int = 9) -> list[dict]:
    """Up to per_day items per day, taken round-robin across the day's sections (a short window needs
    more than round 6's one item per section)."""
    from .readtasks import _wikitext, parse_portal
    rng = random.Random(seed)
    out, d = [], after + timedelta(days=1)
    while d <= until:
        title = f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}"
        try:
            items = parse_portal(_wikitext(title))
        except Exception as e:
            print(f"  [fresh9] {title}: {e.__class__.__name__}")
            items = []
        by_sec = {}
        for it in items:
            by_sec.setdefault(it["section"], []).append(it)
        secs = sorted(by_sec)
        rng.shuffle(secs)
        for v in by_sec.values():
            rng.shuffle(v)
        picked, r = [], 0
        while len(picked) < per_day and any(len(by_sec[x]) > r for x in secs):
            picked += [(x, r, by_sec[x][r]) for x in secs if len(by_sec[x]) > r][: per_day - len(picked)]
            r += 1
        for sec, r_, it in picked:
            out.append({"id": f"{d.isoformat()}_{sec}{r_}", "date": d.isoformat(), "section": sec, "text": it["text"]})
        d += timedelta(days=1)
    return out


def news_fresh(policy, until: date | None = None, seed: int = 9) -> tuple[list[dict], dict]:
    """Gemma writes the questions (resumable raw file); round 6's checks decide what is kept."""
    from .readtasks import write_news_questions
    until = until or date.today() - timedelta(days=1)
    items = portal_items_after(NEWS_AFTER, until, seed=seed)
    print(f"  [fresh9] portal: {len(items)} items, {NEWS_AFTER + timedelta(days=1)} to {until}", flush=True)
    stats = write_news_questions(items, policy, NEWS9_RAW)
    raw = read_json(NEWS9_RAW)["tasks"]
    rng = random.Random(seed)
    rng.shuffle(raw)
    kept = []
    for t in raw:
        if len(kept) >= NEWS_TARGET:
            break
        kept.append(dict(t, id=f"f9{t['id']}", regime="news", qtype="news"))
    return kept, {"portal_items": len(items), "from": str(NEWS_AFTER + timedelta(days=1)), "until": str(until),
                  "questions_written": len(raw), "kept": len(kept), **stats}


def build(gemma_path: str, out: Path = FRESH9, seed: int = 9, until: str | None = None, quick: bool = False) -> dict:
    """Once: if the file exists it is the benchmark of record and is never rebuilt."""
    if out.exists():
        d = read_json(out)
        print(f"[fresh9] {out.name} exists ({len(d['tasks'])} questions, built {d['created']}): kept as the benchmark of record")
        return d
    from .policies import make_policy
    global WD_PER, NEWS_TARGET, NEWS9_RAW
    if quick:                                  # shakedown: a handful of each, separate raw file
        WD_PER, NEWS_TARGET = {k: 3 for k in WD_PER}, 6
        NEWS9_RAW = NEWS9_RAW.with_name("fresh9_news_raw_quick.json")
        until = until or str(NEWS_AFTER + timedelta(days=2))
    wd, wst = wikidata_fresh(seed)
    news, nst = news_fresh(make_policy("gemma", gemma_path=gemma_path), date.fromisoformat(until) if until else None, seed)
    tasks = wd + news
    d = {"version": 1, "created": date.today().isoformat(), "split": "bench-fresh9", "seed": seed,
         "source": ("Wikidata SPARQL 2026 items unused by any earlier benchmark or training set (gold = Wikidata); "
                    "Wikipedia Current Events portal after 2026-10-08, questions by Gemma, round 6 checks (gold = the item's words)"),
         "note": "built before any round 9 reader ran on it; never used to train or choose a reader",
         "counts": dict(Counter(t["regime"] for t in tasks)), "build": {"wikidata": wst, "news": nst}, "tasks": tasks}
    write_json(out, d)
    print(f"[fresh9] {len(tasks)} questions {d['counts']} -> {out}")
    return d


def add_cli(sub):
    p = sub.add_parser("fresh9", help="round 9: build the fresh benchmark once (Wikidata 2026 + portal news via Gemma)")
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--until", default=None, help="last portal day (default: yesterday)")
    p.add_argument("--out", default=str(FRESH9))
    p.add_argument("--quick", action="store_true", help="shakedown: 3 per template, 6 news questions from 2 days")
    p.set_defaults(fn=lambda a: build(a.gemma_path, Path(a.out), a.seed, a.until, a.quick))
