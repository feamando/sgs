"""
Round 6 reader-training questions (SETUP_planck_20261012.md). Disjoint from every benchmark.

The round 5 reader was trained on 1,455 encyclopedic questions about 2012-2024 facts and lost to
Gemma mostly on news-style 2026 questions (35% vs 75%). Reading is a skill, so it is trained once
on BROADER questions, never on the benchmark's world:

    recent    Wikidata, the fresh benchmark's templates on 2025 events (the benchmark is 2026)
    info      Wikidata general-information facts: airport codes and cities served, stadium
              capacities, building floors, station opening dates (popular items only)
    news      Wikipedia's Current Events portal, 2025: one item per section per sampled day;
              Gemma turns each item into ONE question whose short answer the item states
              (on the box; the benchmark's news questions were drafted by Claude from
              2026-09/10 reports, so period and writer both differ)

Labels never come from a model: a candidate is right when it matches the gold answer
(Wikidata, or the portal item's own words). Items whose names appear in a benchmark question
are dropped, and so is every Wikidata item already in a benchmark or the round 3 training set.
"""

import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

from . import benchgen as bg
from .metrics import parse_date, parse_number
from .util import REPO_ROOT, norm_text, read_json, write_json

ASSETS = REPO_ROOT / "scripts" / "assets"
READ_TASKS = ASSETS / "planck3_read_train_tasks.json"
PORTAL_ITEMS = ASSETS / "planck3_portal_items_2025.json"
NEWS_TASKS = REPO_ROOT / "data" / "planck3" / "reader" / "news_tasks.json"
BENCH_FILES = ("planck3_tasks_fresh.json", "planck3_tasks_news.json", "planck3_tasks_geninfo.json",
               "planck3_tasks.json")
WIKI_API = "https://en.wikipedia.org/w/api.php"
SECTIONS = {"Armed conflicts and attacks": "conflict", "Disasters and accidents": "disaster",
            "Sports": "sport", "Politics and elections": "election", "Business and economy": "business",
            "Science and technology": "science", "Arts and culture": "culture",
            "International relations": "diplomacy", "Law and crime": "law", "Health and environment": "health"}


# ── Wikidata: recent (2025) and general-information templates ─────────────
def q_airport(prop: str, min_sl: int, limit: int) -> str:
    """International airports (Q644371) with an IATA code (P238) or a city served (P931)."""
    val = "?code" if prop == "P238" else "?ans"
    lab = "" if prop == "P238" else "?ansLabel (GROUP_CONCAT(DISTINCT ?al; separator=\"|\") AS ?aliases)"
    alias = "" if prop == "P238" else 'OPTIONAL { ?ans skos:altLabel ?al FILTER(LANG(?al) = "en") }'
    group = f"GROUP BY ?item ?itemLabel {val}" + (" ?ansLabel" if prop != "P238" else "")
    return f"""SELECT ?item ?itemLabel {val} {lab} WHERE {{
  ?item wdt:P31 wd:Q644371; wdt:{prop} {val}; wikibase:sitelinks ?sl. {bg.ENWIKI}
  FILTER(?sl >= {min_sl}) {alias}
  {bg.LANG} }} {group} LIMIT {limit}"""


def q_quantity(cls: str, prop: str, min_sl: int, limit: int) -> str:
    return f"""SELECT ?item ?itemLabel ?val WHERE {{
  ?item wdt:P31 wd:{cls}; wdt:{prop} ?val; wikibase:sitelinks ?sl. {bg.ENWIKI}
  FILTER(?sl >= {min_sl} && ?val > 0)
  {bg.LANG} }} LIMIT {limit}"""


def q_opened(cls: str, min_sl: int, limit: int) -> str:
    return f"""SELECT ?item ?itemLabel ?d WHERE {{
  ?item wdt:P31 wd:{cls}; p:P1619/psv:P1619 [ wikibase:timeValue ?d; wikibase:timePrecision 11 ];
        wikibase:sitelinks ?sl. {bg.ENWIKI}
  FILTER(?sl >= {min_sl} && YEAR(?d) >= 1900 && YEAR(?d) <= 2024)
  {bg.LANG} }} LIMIT {limit}"""


def _bench() -> tuple[set[str], list[str]]:
    """Benchmark QIDs, and benchmark question texts (normalized) for the name check."""
    qids, texts = set(), []
    for f in BENCH_FILES + ("planck3_g2_train_tasks.json",):
        p = ASSETS / f
        if not p.exists():
            continue
        for t in read_json(p)["tasks"]:
            qids |= {t.get("qid")} | set(t.get("qids", []))
            if f != "planck3_g2_train_tasks.json":
                texts.append(norm_text(t.get("question") or " ".join(x["user"] for x in t.get("turns", []))))
    return qids - {None}, texts


def named_in_bench(label: str, bench_texts: list[str]) -> bool:
    """Is this item (its distinctive words) the subject of a benchmark question?"""
    words = [w for w in norm_text(label).split() if len(w) > 3 and w not in
             {"international", "airport", "stadium", "station", "railway", "tower", "building", "park", "arena", "centre", "center"}]
    if not words:
        return False
    return any(all(re.search(rf"\b{re.escape(w)}\b", t) for w in words) for t in bench_texts)


def wikidata_tasks(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    bench_qids, bench_texts = _bench()
    tasks: list[dict] = []

    def add(group, name, answer_type, rows, qfn, gfn, n):
        items = list(bg._group(rows).items())
        rng.shuffle(items)
        k = 0
        for uri, rs in items:
            if k >= n:
                break
            label = rs[0].get("itemLabel", "")
            if not label or re.fullmatch(r"Q\d+", label) or bg._qid(uri) in bench_qids or named_in_bench(label, bench_texts):
                continue
            gold = gfn(rs)
            if not gold or (answer_type == "entity" and not gold[0][:1].isupper()):
                continue
            t = bg._task(f"{group}_{name}{k:03d}", group, name, qfn(label, rs), answer_type, gold, rs)
            tasks.append(t)
            k += 1
        print(f"  {group:<7} {name:<10} {k:>4} tasks (pool {len(items)})")

    y = (2025, 2026)
    add("recent", "winner", "entity", bg.sparql(bg.q_event_winners(*y, 2000)),
        lambda l, rs: f"Who won the {l}?", lambda rs: bg._golds(rs), 300)
    add("recent", "election", "entity", bg.sparql(bg.q_elections(*y, 1500)),
        lambda l, rs: f"Who won the {l}?", lambda rs: bg._golds(rs), 120)
    add("recent", "season", "entity", bg.sparql(bg.q_season_champions(*y, 1500)),
        lambda l, rs: f"Who won the {l}?", lambda rs: bg._golds(rs), 120)
    add("recent", "headgov", "entity", bg.sparql(bg.q_head_of_gov(2025, 1500)),
        lambda l, rs: f"Who became head of government of {l} in 2025?", lambda rs: bg._golds(rs), 60)
    add("info", "iata", "entity", bg.sparql(q_airport("P238", 10, 1500)),
        lambda l, rs: f"What is the IATA code of {l}?", lambda rs: [rs[0]["code"]], 120)
    add("info", "serves", "entity", bg.sparql(q_airport("P931", 10, 1500)),
        lambda l, rs: f"Which city does {l} serve?", lambda rs: bg._golds(rs), 100)
    add("info", "capacity", "number", bg.sparql(q_quantity("Q483110", "P1083", 8, 1500)),
        lambda l, rs: f"What is the seating capacity of {l}?", lambda rs: [str(round(float(rs[0]["val"])))], 100)
    add("info", "floors", "number", bg.sparql(q_quantity("Q11303", "P1101", 6, 1500)),
        lambda l, rs: f"How many floors does {l} have?", lambda rs: [str(round(float(rs[0]["val"])))], 80)
    add("info", "opened", "date", bg.sparql(q_opened("Q55488", 6, 1500)),
        lambda l, rs: f"When did {l} open?", lambda rs: bg._date_gold(rs[0]["d"]), 100)
    return tasks


# ── the Current Events portal (2025) ──────────────────────────────────────
def _wikitext(title: str) -> str:
    import requests
    r = requests.get(WIKI_API, params={"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                       "titles": title, "format": "json", "formatversion": 2},
                     headers={"User-Agent": bg.UA}, timeout=30)
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    return pages[0]["revisions"][0]["slots"]["main"]["content"] if pages and "revisions" in pages[0] else ""


def strip_wiki(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\[https?://\S+\s*([^\]]*)\]", "", s)                        # source links: (Reuters)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_portal(text: str) -> list[dict]:
    """Leaf bullet items with their section: [{section, text}]."""
    out, section = [], None
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^'''(.+?)'''\s*$", line.strip())
        if m:
            section = SECTIONS.get(strip_wiki(m.group(1)))
            continue
        b = re.match(r"^(\*+)\s*(.*)$", line)
        if not b or section is None:
            continue
        depth = len(b.group(1))
        nxt = re.match(r"^(\*+)", lines[i + 1]) if i + 1 < len(lines) else None
        if nxt and len(nxt.group(1)) > depth:
            continue                                   # a heading bullet ("*[[Russian invasion of Ukraine]]")
        t = strip_wiki(b.group(2))
        if 60 <= len(t) <= 420 and re.search(r"\d|[A-Z][a-z]+ [A-Z]", t):
            out.append({"section": section, "text": t})
    return out


def portal_items(year: int = 2025, per_day: int = 3, seed: int = 0, days: int = 0) -> list[dict]:
    """per_day items per sampled day, each from a different section (spread over the benchmark's mix)."""
    rng = random.Random(seed)
    d0 = date(year, 1, 1)
    all_days = [d0 + timedelta(days=i) for i in range((date(year + 1, 1, 1) - d0).days)]
    if days:
        all_days = sorted(rng.sample(all_days, days))
    out = []
    for d in all_days:
        title = f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}"
        try:
            items = parse_portal(_wikitext(title))
        except Exception as e:  # one missing day is not fatal
            print(f"  [portal] {title}: {e.__class__.__name__}")
            continue
        by_sec = {}
        for it in items:
            by_sec.setdefault(it["section"], []).append(it)
        secs = list(by_sec)
        rng.shuffle(secs)
        for s in secs[:per_day]:
            it = rng.choice(by_sec[s])
            out.append({"id": f"{d.isoformat()}_{s}", "date": d.isoformat(), "section": s, "text": it["text"]})
    return out


NEWSQ_SYSTEM = """You turn one news summary into one quiz question for a web search test.
The question must have a SHORT answer stated in the summary: a person or organisation's name, a place,
a number, or a date. Include the month and year of the event in the question (for example
"in March 2025"), so that it has one answer. Never put the answer in the question.
Copy the answer exactly as the summary writes it.
Reply with ONLY one JSON object: {"question": "...", "answer": "...", "answer_type": "entity|number|date"}"""


def check_news_question(obj: dict, item: dict) -> tuple[dict | None, str]:
    """A Gemma-written question becomes a task only if it passes every check (reasons are logged)."""
    q, a, at = (obj.get("question") or "").strip(), (obj.get("answer") or "").strip(), obj.get("answer_type")
    if at not in ("entity", "number", "date"):
        return None, "answer type"
    if not q.endswith("?") or not 6 <= len(q.split()) <= 35:
        return None, "question shape"
    if not a or len(a.split()) > 8:
        return None, "answer length"
    na, nq, nt = norm_text(a), norm_text(q), norm_text(item["text"])
    if na not in nt:
        return None, "answer not in the item"
    if re.search(rf"\b{re.escape(na)}\b", nq):
        return None, "answer in the question"
    if item["date"][:4] not in q:
        return None, "no year in the question"
    if at == "number" and parse_number(a) is None:
        return None, "number without a number"
    if at == "date" and parse_date(a) is None and not re.search(r"\b\d{4}\b", a):
        return None, "date without a date"
    return {"id": f"news_{item['id']}", "family": "fact", "regime": "news", "template": item["section"],
            "question": q, "answer_type": at, "gold": [a], "source": f"Portal:Current events {item['date']}",
            "evidence": item["text"], "gold_as_of": item["date"]}, "ok"


def write_news_questions(items: list[dict], policy, out: Path, limit: int = 0) -> dict:
    """Gemma writes one question per portal item (resumable: finished item ids are skipped)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_json(out) if out.exists() else {"tasks": [], "rejected": []}
    seen = {t["id"].removeprefix("news_") for t in done["tasks"]} | {r["id"] for r in done["rejected"]}
    _, bench_texts = _bench()
    todo = [it for it in items if it["id"] not in seen][: limit or None]
    old = getattr(policy, "max_new", None)
    if old is not None:
        policy.max_new = max(old, 120)
    print(f"[readtasks] Gemma writes questions for {len(todo)} portal items ({len(seen)} done) -> {out}")
    for i, it in enumerate(todo, 1):
        raw = policy._complete(NEWSQ_SYSTEM, f"DATE: {it['date']}\nSECTION: {it['section']}\nSUMMARY: {it['text']}")
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            obj = {}
        task, why = check_news_question(obj, it)
        if task and named_in_bench(task["gold"][0], bench_texts) and task["answer_type"] == "entity":
            task, why = None, "answer named in a benchmark question"
        if task:
            done["tasks"].append(task)
        else:
            done["rejected"].append({"id": it["id"], "why": why, "raw": raw[:300]})
        if i % 50 == 0 or i == len(todo):
            write_json(out, done)
            print(f"  {i}/{len(todo)}: {len(done['tasks'])} questions kept, {len(done['rejected'])} rejected", flush=True)
    if old is not None:
        policy.max_new = old
    write_json(out, done)
    from collections import Counter
    return {"kept": len(done["tasks"]), "rejected": dict(Counter(r["why"] for r in done["rejected"]))}


def build(seed: int = 0, per_day: int = 3, days: int = 0):
    """On any machine with internet: the Wikidata tasks and the portal items (both committed)."""
    tasks = wikidata_tasks(seed)
    write_json(READ_TASKS, {"version": 1, "created": date.today().isoformat(), "split": "reader-train",
                            "source": "Wikidata SPARQL (cached); disjoint by QID from every benchmark and the round 3 "
                                      "training set; items named in a benchmark question dropped", "tasks": tasks})
    items = portal_items(2025, per_day, seed, days)
    write_json(PORTAL_ITEMS, {"version": 1, "created": date.today().isoformat(),
                              "source": "Wikipedia Portal:Current events, 2025 (one item per section per day, seeded)",
                              "items": items})
    from collections import Counter
    print(f"[readtasks] {len(tasks)} Wikidata tasks {dict(Counter(t['regime'] for t in tasks))} -> {READ_TASKS}")
    print(f"[readtasks] {len(items)} portal items {dict(Counter(i['section'] for i in items))} -> {PORTAL_ITEMS}")


def add_cli(sub):
    p = sub.add_parser("readtasks", help="round 6 reader-training questions: build (Wikidata + portal) | newsq (Gemma)")
    p.add_argument("stage", choices=["build", "newsq"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-day", type=int, default=3)
    p.add_argument("--days", type=int, default=0, help="sample this many days of 2025 (0 = all)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=str(NEWS_TASKS))
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")

    def run(a):
        if a.stage == "build":
            return build(a.seed, a.per_day, a.days)
        from .policies import make_policy
        items = read_json(PORTAL_ITEMS)["items"]
        print(f"[readtasks] {write_news_questions(items, make_policy('gemma', gemma_path=a.gemma_path), Path(a.out), a.limit)}")
    p.set_defaults(fn=run)
