"""
Page-grounded benchmark + G2 training tasks from Wikidata (dated, sourced, reproducible).

Two regimes where base chat (an LLM answering from its weights) is weak:
    fresh      facts from 2026: event winners, election winners, season champions,
               new heads of government
               (after the teacher's training data, so closed-book cannot know them)
    long_tail  stable facts about OBSCURE items: entities with <= 3 Wikipedia language
               editions (sitelinks), still with an English article so the web has them

Every task carries: wikidata QID, enwiki URL, every acceptable gold label (label +
aliases), the template, and `gold_as_of`. The G2 training split uses the same
templates on OLDER / more popular items and is disjoint from the benchmark by QID.
Query results are cached on disk, so a rebuild is cheap and deterministic.
"""

import hashlib
import json
import random
import re
import time
from datetime import date
from pathlib import Path

from .util import REPO_ROOT, write_json

ENDPOINT = "https://query.wikidata.org/sparql"
UA = "Planck3-research/0.1 (+https://github.com/feamando/sgs; benchmark generation)"
CACHE = REPO_ROOT / "data" / "planck3" / "wikidata_cache"
LANG = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'
ENWIKI = "?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>."
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"]


def sparql(query: str, retries: int = 3) -> list[dict]:
    import requests
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(query.encode("utf-8")).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    for attempt in range(retries):
        try:
            r = requests.get(ENDPOINT, params={"query": query}, timeout=90,
                             headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            # strict=False: some Wikidata descriptions carry raw control characters
            data = json.loads(r.text, strict=False)
            rows = [{k: v["value"] for k, v in b.items()} for b in data["results"]["bindings"]]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            time.sleep(1.0)  # be polite to the public endpoint
            return rows
        except Exception as e:  # timeouts on big classes: caller narrows the scope
            print(f"  [wikidata] attempt {attempt + 1} failed: {e.__class__.__name__}: {str(e)[:120]}")
            time.sleep(3)
    return []


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _clean_desc(d: str | None) -> str | None:
    """Descriptions disambiguate the subject, but some leak the answer ('born 1987'): drop those."""
    if not d or re.search(r"\d", d) or len(d) > 60:
        return None
    return d


def _subject(label: str, desc: str | None) -> str:
    return f"{label}, {desc}," if desc else label


def _golds(rows, key="ans", alias_key="aliases") -> list[str]:
    out = []
    for r in rows:
        for v in [r.get(f"{key}Label")] + (r.get(alias_key) or "").split("|"):
            v = (v or "").strip()
            if v and not re.fullmatch(r"Q\d+", v) and v not in out:
                out.append(v)
    return out[:8]


# ── templates: (name, regime, answer_type, query builder, question builder) ──
def q_event_winners(y0: int, y1: int, limit: int) -> str:
    return f"""SELECT ?item ?itemLabel ?ans ?ansLabel (GROUP_CONCAT(DISTINCT ?al; separator="|") AS ?aliases) ?date WHERE {{
  ?item wdt:P1346 ?ans; wdt:P585 ?date. {ENWIKI}
  FILTER(?date >= "{y0}-01-01"^^xsd:dateTime && ?date < "{y1}-01-01"^^xsd:dateTime)
  OPTIONAL {{ ?ans skos:altLabel ?al FILTER(LANG(?al) = "en") }}
  {LANG} }} GROUP BY ?item ?itemLabel ?ans ?ansLabel ?date LIMIT {limit}"""


def q_elections(y0: int, y1: int, limit: int) -> str:
    """Elections (or any item) with a successful candidate (P991), dated in [y0, y1)."""
    return f"""SELECT ?item ?itemLabel ?ans ?ansLabel (GROUP_CONCAT(DISTINCT ?al; separator="|") AS ?aliases) WHERE {{
  ?item wdt:P991 ?ans; wdt:P585 ?date. {ENWIKI}
  FILTER(?date >= "{y0}-01-01"^^xsd:dateTime && ?date < "{y1}-01-01"^^xsd:dateTime)
  OPTIONAL {{ ?ans skos:altLabel ?al FILTER(LANG(?al) = "en") }}
  {LANG} }} GROUP BY ?item ?itemLabel ?ans ?ansLabel LIMIT {limit}"""


def q_season_champions(y0: int, y1: int, limit: int) -> str:
    """Seasons / tournaments with a winner (P1346) that ENDED (P582) in [y0, y1)."""
    return f"""SELECT ?item ?itemLabel ?ans ?ansLabel (GROUP_CONCAT(DISTINCT ?al; separator="|") AS ?aliases) WHERE {{
  ?item wdt:P1346 ?ans; wdt:P582 ?end. {ENWIKI}
  FILTER(?end >= "{y0}-01-01"^^xsd:dateTime && ?end < "{y1}-01-01"^^xsd:dateTime)
  OPTIONAL {{ ?ans skos:altLabel ?al FILTER(LANG(?al) = "en") }}
  {LANG} }} GROUP BY ?item ?itemLabel ?ans ?ansLabel LIMIT {limit}"""


def q_head_of_gov(y0: int, limit: int) -> str:
    return f"""SELECT ?item ?itemLabel ?ans ?ansLabel (GROUP_CONCAT(DISTINCT ?al; separator="|") AS ?aliases) ?start WHERE {{
  ?item p:P6 ?st. ?st ps:P6 ?ans; pq:P580 ?start. {ENWIKI}
  FILTER(?start >= "{y0}-01-01"^^xsd:dateTime && ?start < "{y0 + 1}-01-01"^^xsd:dateTime)
  FILTER NOT EXISTS {{ ?st pq:P582 ?end }}
  OPTIONAL {{ ?ans skos:altLabel ?al FILTER(LANG(?al) = "en") }}
  {LANG} }} GROUP BY ?item ?itemLabel ?ans ?ansLabel ?start LIMIT {limit}"""


# Long-tail queries are sliced by COUNTRY: filtering a whole class (all businesses, all
# footballers) by sitelink count times out on the public endpoint; one country is fast.
COUNTRIES = ["Q34", "Q20", "Q33", "Q35", "Q39", "Q40", "Q31", "Q55", "Q45", "Q28", "Q213", "Q37",
             "Q211", "Q191", "Q27", "Q664"]


def q_inception(cls: str, max_sl: int, min_sl: int, limit: int, country: str = "Q34") -> str:
    return f"""SELECT ?item ?itemLabel ?itemDescription ?inc WHERE {{
  ?item wdt:P31 wd:{cls}; wdt:P17 wd:{country}; p:P571/psv:P571 [ wikibase:timeValue ?inc; wikibase:timePrecision ?pr ];
        wikibase:sitelinks ?sl. {ENWIKI}
  FILTER(?pr >= 9 && ?sl <= {max_sl} && ?sl >= {min_sl} && YEAR(?inc) >= 1800)
  {LANG} }} LIMIT {limit}"""


def q_birth(occupation: str, year: int, max_sl: int, min_sl: int, limit: int, country: str = "Q34") -> str:
    return f"""SELECT ?item ?itemLabel ?itemDescription ?dob WHERE {{
  ?item wdt:P106 wd:{occupation}; wdt:P27 wd:{country}; p:P569/psv:P569 [ wikibase:timeValue ?dob; wikibase:timePrecision 11 ];
        wikibase:sitelinks ?sl. {ENWIKI}
  FILTER(YEAR(?dob) = {year} && ?sl <= {max_sl} && ?sl >= {min_sl})
  {LANG} }} LIMIT {limit}"""


def q_elevation(max_sl: int, min_sl: int, limit: int, country: str = "Q39") -> str:
    return f"""SELECT ?item ?itemLabel ?itemDescription ?elev WHERE {{
  ?item wdt:P31 wd:Q8502; wdt:P17 wd:{country}; p:P2044/psv:P2044 [ wikibase:quantityAmount ?elev; wikibase:quantityUnit wd:Q11573 ];
        wikibase:sitelinks ?sl. {ENWIKI}
  FILTER(?sl <= {max_sl} && ?sl >= {min_sl} && ?elev > 100)
  {LANG} }} LIMIT {limit}"""


def q_relation(cls: str, prop: str, max_sl: int, min_sl: int, limit: int, country: str = "Q34",
               country_prop: str = "P17") -> str:
    return f"""SELECT ?item ?itemLabel ?itemDescription ?ans ?ansLabel (GROUP_CONCAT(DISTINCT ?al; separator="|") AS ?aliases) WHERE {{
  ?item wdt:P31 wd:{cls}; wdt:{country_prop} wd:{country}; wdt:{prop} ?ans; wikibase:sitelinks ?sl. {ENWIKI}
  FILTER(?sl <= {max_sl} && ?sl >= {min_sl})
  OPTIONAL {{ ?ans skos:altLabel ?al FILTER(LANG(?al) = "en") }}
  {LANG} }} GROUP BY ?item ?itemLabel ?itemDescription ?ans ?ansLabel LIMIT {limit}"""


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    by = {}
    for r in rows:
        by.setdefault(r["item"], []).append(r)
    return by


def _task(tid, regime, template, question, answer_type, gold, item_rows, extra=None) -> dict:
    r0 = item_rows[0]
    t = {"id": tid, "family": "fact", "regime": regime, "template": template, "question": question,
         "answer_type": answer_type, "gold": gold, "qid": _qid(r0["item"]), "entity": r0.get("itemLabel"),
         "source": f"https://www.wikidata.org/wiki/{_qid(r0['item'])}", "gold_as_of": date.today().isoformat()}
    t.update(extra or {})
    return t


def _date_gold(iso: str) -> list[str]:
    d = iso[:10]
    y, m, dd = d.split("-")
    return [d, f"{int(dd)} {_MONTHS[int(m) - 1]} {y}", f"{_MONTHS[int(m) - 1]} {int(dd)}, {y}"]


def generate(split: str, seed: int = 0) -> list[dict]:
    """split: 'bench' (fresh + long-tail, ~140) or 'train' (older / popular, ~1,400, for G2)."""
    rng = random.Random(seed)
    tasks: list[dict] = []
    bench = split == "bench"

    def add(name, regime, answer_type, rows, qfn, gfn, n):
        items = list(_group(rows).items())
        rng.shuffle(items)
        k = 0
        for qid_uri, rs in items:
            if k >= n:
                break
            label = rs[0].get("itemLabel", "")
            if not label or re.fullmatch(r"Q\d+", label):
                continue
            gold = gfn(rs)
            if not gold:
                continue
            if answer_type == "entity" and not gold[0][:1].isupper():
                continue  # "independent politician" is a class, not a winner
            if re.search(r"\b(Cabinet|Government|Ministry|Council of)\b", label):
                continue  # head of government OF a cabinet is not a place question
            q = qfn(label, _clean_desc(rs[0].get("itemDescription")), rs)
            tasks.append(_task(f"{split[0]}{name}{k:03d}", regime, name, q, answer_type, gold, rs))
            k += 1
        print(f"  {name:<12} {regime:<9} {k:>4} tasks (pool {len(items)})")

    print(f"[benchgen] split={split}")
    # fresh (bench) / older (train): event winners
    y = (2026, 2027) if bench else (2015, 2025)
    add("winner", "fresh" if bench else "stable", "entity", sparql(q_event_winners(*y, 1500)),
        lambda l, d, rs: f"Who won the {l}?", lambda rs: _golds(rs), 30 if bench else 250)
    y = (2026, 2027) if bench else (2012, 2025)
    add("election", "fresh" if bench else "stable", "entity", sparql(q_elections(*y, 1500)),
        lambda l, d, rs: f"Who won the {l}?", lambda rs: _golds(rs), 10 if bench else 150)
    add("season", "fresh" if bench else "stable", "entity", sparql(q_season_champions(*y, 1500)),
        lambda l, d, rs: f"Who won the {l}?", lambda rs: _golds(rs), 10 if bench else 150)
    if bench:
        add("headgov", "fresh", "entity", sparql(q_head_of_gov(2026, 1500)),
            lambda l, d, rs: f"Who became head of government of {l} in {rs[0]['start'][:4]}?",
            lambda rs: _golds(rs), 10)
    # long-tail (bench: <= 3 sitelinks) / popular-ish (train: 4-60 sitelinks)
    sl = (3, 1) if bench else (60, 4)
    by_country = lambda qfn: [r for c in COUNTRIES for r in sparql(qfn(c))]  # noqa: E731
    inc_rows = by_country(lambda c: q_inception("Q4830453", *sl, 300, c))
    add("founded", "long_tail" if bench else "stable", "year", inc_rows,
        lambda l, d, rs: f"In what year was {_subject(l, d)} founded?".replace(", founded", " founded"),
        lambda rs: [rs[0]["inc"][:4]], 20 if bench else 300)
    birth_rows = []
    for i, (occ, yr) in enumerate([("Q937857", 1988), ("Q82955", 1961), ("Q33999", 1975), ("Q36834", 1949), ("Q2066131", 1993)]):
        for j, c in enumerate(COUNTRIES[i::5]):
            # a different year per slice: one shared gold year would be gameable ("always say 1988")
            y_ = yr + 7 * j - 10 + (0 if bench else 3)
            birth_rows += sparql(q_birth(occ, y_, *sl, 200, c))
    add("born", "long_tail" if bench else "stable", "year", birth_rows,
        lambda l, d, rs: f"In what year was {_subject(l, d)} born?".replace(", born", " born"),
        lambda rs: [rs[0]["dob"][:4]], 20 if bench else 300)
    elev_rows = [r for c in ["Q39", "Q40", "Q20", "Q38", "Q142", "Q183", "Q29", "Q16"] for r in sparql(q_elevation(*sl, 300, c))]
    add("elevation", "long_tail" if bench else "stable", "number", elev_rows,
        lambda l, d, rs: f"How high is {_subject(l, d)} in metres?".replace(", in metres", " in metres"),
        lambda rs: [str(round(float(rs[0]["elev"])))], 15 if bench else 150)
    add("founder", "long_tail" if bench else "stable", "entity", by_country(lambda c: q_relation("Q4830453", "P112", *sl, 300, c)),
        lambda l, d, rs: f"Who founded {_subject(l, d)}?".replace(",?", "?"), lambda rs: _golds(rs), 10 if bench else 150)
    add("author", "long_tail" if bench else "stable", "entity",
        by_country(lambda c: q_relation("Q7725634", "P50", *sl, 300, c, country_prop="P495")),
        lambda l, d, rs: f"Who wrote {_subject(l, d)}?".replace(",?", "?"), lambda rs: _golds(rs), 10 if bench else 150)
    if bench:
        tasks += _chat_tasks(birth_rows, inc_rows, rng)
    return tasks


def _chat_tasks(birth_rows, inc_rows, rng, n: int = 8) -> list[dict]:
    # birth_rows now span many years per slice, so the two turns rarely share a gold year
    """Two-turn conversations: '<question about A>' then 'And <B>?' (entity swap), long-tail subjects."""
    out = []
    for name, rows, tmpl, at, key in (("born", birth_rows, "In what year was {} born?", "year", "dob"),
                                      ("founded", inc_rows, "In what year was {} founded?", "year", "inc")):
        items = [rs for rs in _group(rows).values() if len(rs[0].get("itemLabel", "").split()) >= 2
                 and not re.search(r"\d", rs[0].get("itemLabel", ""))]
        rng.shuffle(items)
        for i in range(0, min(len(items) - 1, 2 * (n // 2)), 2):
            a, b = items[i][0], items[i + 1][0]
            out.append({"id": f"bchat_{name}{i // 2:02d}", "family": "chat", "regime": "long_tail", "template": f"chat_{name}",
                        "turns": [{"user": tmpl.format(a["itemLabel"]), "answer_type": at, "rewrite": "NEW", "gold": [a[key][:4]]},
                                  {"user": f"And {b['itemLabel']}?", "answer_type": at, "rewrite": "SWAP_ENTITY", "gold": [b[key][:4]]}],
                        "qids": [_qid(a["item"]), _qid(b["item"])], "gold_as_of": date.today().isoformat()})
            if len([t for t in out if t["template"] == f"chat_{name}"]) >= n // 2:
                break
    return out


def build(out_bench: Path, out_train: Path, seed: int = 0):
    bench = generate("bench", seed)
    bench_qids = {t.get("qid") for t in bench} | {q for t in bench for q in t.get("qids", [])}
    train = [t for t in generate("train", seed + 1) if t.get("qid") not in bench_qids]
    meta = {"created": date.today().isoformat(), "source": "Wikidata SPARQL (query.wikidata.org), cached",
            "note": ("fresh = 2026 facts (after the teacher's training data); long_tail = items with <= 3 Wikipedia "
                     "sitelinks; gold = Wikidata label + English aliases as of gold_as_of. Not tuned against any policy.")}
    write_json(out_bench, {"version": 1, **meta, "split": "bench", "tasks": bench})
    write_json(out_train, {"version": 1, **meta, "split": "train", "tasks": train})
    from collections import Counter
    print(f"[benchgen] bench {len(bench)} tasks {dict(Counter(t['regime'] for t in bench))} -> {out_bench}")
    print(f"[benchgen] train {len(train)} tasks (disjoint by QID) -> {out_train}")
