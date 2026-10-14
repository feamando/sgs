"""
Round 7 (SETUP_planck_20261014.md): reading taught as a SKILL from public reading-comprehension sets,
turned into the reader's own pools so training looks exactly like answering.

    sources   MRQA 2019 train (mrqa-workshop/mrqa): SQuAD, NewsQA (CNN news), TriviaQA-web,
              SearchQA (search-result snippets), NaturalQuestions; SQuAD 2.0 unanswerable questions
              (rajpurkar/squad_v2) for "the answer is not in here". HotpotQA (multi-hop) is left out.
    pools     the passage goes through the SAME candidate extractor as a fetched page
              (span_candidates; SearchQA's documents go through snippet_candidates, as search
              results), labels = is_correct against the dataset's answers, top 16 as served
    types     answer type from the question (chat.infer_answer_type); a "text" question takes the
              gold answer's type; still untyped -> dropped (the benchmark has no text questions)
    balance   per set: at most 40% of pools without a right candidate on offer
    disjoint  a question whose subject is named in a benchmark question, or whose answer is a
              benchmark answer (entity, > 4 letters), is dropped; counts in rc_points.health.json

The passages are 2019-and-earlier text, so nothing here is about 2026. They do carry old world facts;
the reader learns to find an answer in a passage, and the benchmark's fresh and news questions are
from 2026 (long-tail and general information are reported separately for that reason).
"""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .util import REPO_ROOT, norm_text, read_json, write_json

RC_POINTS = REPO_ROOT / "data" / "planck3" / "reader" / "rc_points.jsonl"
RAW_DIR = REPO_ROOT / "data" / "planck3" / "rc_raw"
MRQA_REPO, MRQA_SHARDS = "mrqa-workshop/mrqa", [f"plain_text/train-{i:05d}-of-00009.parquet" for i in range(9)]
SQUAD2_REPO, SQUAD2_FILE = "rajpurkar/squad_v2", "squad_v2/train-00000-of-00001.parquet"
SETS = {"SQuAD": "squad", "NewsQA": "newsqa", "TriviaQA-web": "triviaqa", "SearchQA": "searchqa",
        "NaturalQuestionsShort": "nq"}
NEG_GROUP = "squad2neg"
GROUPS = tuple(SETS.values()) + (NEG_GROUP,)
NO_RIGHT_MAX = 0.4
TYPED = ("year", "number", "date", "entity")
_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_YEAR_ONLY = re.compile(r"^\s*(1[5-9]\d\d|20\d\d)\s*$")


# ── answer types ─────────────────────────────────────────────────────────
def gold_type(g: str) -> str | None:
    from .candidates import RE_DATE
    from .metrics import parse_number
    g = (g or "").strip()
    if not g:
        return None
    if _YEAR_ONLY.match(g):
        return "year"
    if RE_DATE.search(g):
        return "date"
    if parse_number(g) is not None and re.fullmatch(r"[\d.,\s%$£€a-zA-Z]{1,24}", g) and re.search(r"\d", g):
        return "number"
    if g[:1].isupper():
        return "entity"
    return None


def answer_type_of(question: str, golds: list[str]) -> str | None:
    from .chat import infer_answer_type
    at = infer_answer_type(question)
    if at == "text":
        at = gold_type(golds[0]) if golds else None
    return at if at in TYPED else None


# ── benchmark disjointness ───────────────────────────────────────────────
_SKIP_NAME = {"international", "airport", "stadium", "station", "railway", "tower", "building", "park", "arena",
              "centre", "center", "the", "and"}


def bench_index() -> tuple[list[set], set]:
    """Token sets of the benchmark questions, and the benchmark's entity answers (normalized)."""
    from .readtasks import ASSETS, BENCH_FILES
    texts, golds = [], set()
    for f in BENCH_FILES:
        p = ASSETS / f
        if not p.exists():
            continue
        for t in read_json(p)["tasks"]:
            q = t.get("question") or " ".join(x["user"] for x in t.get("turns", []))
            texts.append(set(norm_text(q).split()))
            for g in t.get("gold") or []:
                if isinstance(g, str) and len(norm_text(g)) > 4 and not re.search(r"\d", g):
                    golds.add(norm_text(g))
    return texts, golds


def in_bench(question: str, golds: list[str], bench: tuple[list[set], set]) -> bool:
    from .candidates import main_entity
    texts, bgolds = bench
    subj = main_entity(question)
    if subj:
        ws = {w for w in norm_text(subj).split() if len(w) > 3 and w not in _SKIP_NAME}
        if ws and any(ws <= t for t in texts):
            return True
    return any(norm_text(g) in bgolds for g in golds)


# ── passages -> pools ────────────────────────────────────────────────────
def parse_docs(context: str) -> list[dict]:
    """MRQA '[DOC] [TLE] title [PAR] text ...' -> [{title, text}] (one doc when there are no markers)."""
    if "[DOC]" not in context:
        return [{"title": "", "text": clean(context)}]
    out = []
    for d in context.split("[DOC]"):
        d = d.strip()
        if not d:
            continue
        d = d.replace("[TLE]", "").strip()
        title, _, body = d.partition("[PAR]")
        out.append({"title": clean(title), "text": clean(body)})
    return out


def clean(s: str) -> str:
    s = _TAG.sub(" ", s or "")
    s = s.replace("[PAR]", "\n").replace("[SEP]", " ").replace("[TLE]", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\s*\n\s*", "\n", s).strip()


def pool_of(row: dict, group: str) -> dict | None:
    """One dataset row -> one pool in the reader's format (None when it cannot be used)."""
    from .candidates import snippet_candidates, span_candidates
    from .metrics import is_correct
    q, golds = row["question"].strip(), [g for g in row["answers"] if g]
    if not golds and group != NEG_GROUP:
        return None
    at = answer_type_of(q, golds)
    if not at:
        return None
    docs = parse_docs(row["context"])
    if group == "searchqa":
        results = [{"url": f"https://d{i}.searchqa.example/", "title": d["title"], "snippet": d["text"]}
                   for i, d in enumerate(docs)]
        cands, src = snippet_candidates(q, results, at), "snippet"
    else:
        text = "\n".join((d["title"] + ".\n" if d["title"] else "") + d["text"] for d in docs)
        cands, src = span_candidates(q, text, at, url=f"https://{group}.example/"), "page"
    if not cands:
        return None
    labels = [bool(golds) and is_correct(c["value"], golds, at) for c in cands]
    tid = f"{group}_{hashlib.sha1((row.get('qid') or q).encode('utf-8')).hexdigest()[:12]}"
    return {"task_id": tid, "question": q, "answer_type": at, "source": src, "search_backend": group,
            "cands": [{k: c.get(k) for k in ("value", "context", "score", "margin", "about", "domains", "support", "url")}
                      for c in cands], "labels": labels}


# ── download + build ─────────────────────────────────────────────────────
def _download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(RAW_DIR)))


def _rows_by_set(seed: int, take: dict[str, int]) -> dict[str, list[dict]]:
    """A seeded random sample of up to take[g] rows per set, read shard by shard (the full MRQA
    contexts would need several GB of memory; only the sampled rows are materialized)."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    files = [_download(MRQA_REPO, f) for f in MRQA_SHARDS]
    where: dict[str, list[tuple[int, int]]] = {g: [] for g in GROUPS}
    for fi, f in enumerate(files):
        for ri, s in enumerate(pq.read_table(f, columns=["subset"]).column("subset").to_pylist()):
            if s in SETS:
                where[SETS[s]].append((fi, ri))
    picked: dict[int, set] = {}
    for g in SETS.values():
        rng.shuffle(where[g])
        for fi, ri in where[g][:take[g]]:
            picked.setdefault(fi, set()).add(ri)
    by: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for fi, rows in picked.items():
        t = pq.read_table(files[fi], columns=["subset", "qid", "question", "context", "answers"]).take(sorted(rows))
        for r in t.to_pylist():
            by[SETS[r["subset"]]].append({"qid": r["qid"], "question": r["question"], "context": r["context"],
                                          "answers": list(r["answers"])})
    t = pq.read_table(_download(SQUAD2_REPO, SQUAD2_FILE), columns=["id", "question", "context", "answers"])
    for r in t.to_pylist():
        if not r["answers"]["text"]:
            by[NEG_GROUP].append({"qid": "sq2-" + r["id"], "question": r["question"], "context": r["context"], "answers": []})
    for g in by:
        rng.shuffle(by[g])
    by[NEG_GROUP] = by[NEG_GROUP][:take[NEG_GROUP]]
    return by


def build(out: Path = RC_POINTS, per_set: int = 40000, neg: int = 20000, seed: int = 0) -> dict:
    """Pools for every set (resumable: an existing file is kept). Returns the health summary."""
    if out.exists() and out.stat().st_size:
        print(f"[rc] {out} exists; keeping it (delete it to rebuild)")
        return read_json(out.with_suffix(".health.json")) if out.with_suffix(".health.json").exists() else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    bench = bench_index()
    # rows to read: about 3x the cap (typing, extraction and the 40% rule drop many)
    by = _rows_by_set(seed, {g: 3 * (neg if g == NEG_GROUP else per_set) for g in GROUPS})
    stats = {}
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for g, rows in by.items():
            cap = neg if g == NEG_GROUP else per_set
            st = Counter(rows=len(rows))
            keep_right, keep_none = 0, 0
            max_none = cap if g == NEG_GROUP else int(NO_RIGHT_MAX * cap)
            max_right = 0 if g == NEG_GROUP else cap - max_none
            for r in rows:
                if keep_right >= max_right and keep_none >= max_none:
                    break
                if in_bench(r["question"], r["answers"], bench):
                    st["dropped_benchmark"] += 1
                    continue
                p = pool_of(r, g)
                if p is None:
                    st["dropped_untyped_or_empty"] += 1
                    continue
                right = any(p["labels"][:16])
                if right and keep_right < max_right:
                    keep_right += 1
                elif not right and keep_none < max_none:
                    keep_none += 1
                else:
                    continue
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            st.update(kept=keep_right + keep_none, with_right_in_top16=keep_right, without=keep_none)
            stats[g] = dict(st)
            print(f"[rc] {g}: kept {keep_right + keep_none} ({keep_right} with a right candidate in the top 16) "
                  f"of {st['rows']} rows; {st['dropped_benchmark']} dropped as benchmark subjects", flush=True)
    tmp.replace(out)
    health = {"sets": stats, "per_set": per_set, "neg": neg, "seed": seed,
              "total": sum(s["kept"] for s in stats.values())}
    write_json(out.with_suffix(".health.json"), health)
    return health
