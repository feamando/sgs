"""
The knowledge store: where Planck 3.0 "gets smarter with use" (weights stay frozen).

    facts    every answered value + source + span + p + freshness TTL
    domains  per-domain Beta(a, b) trust prior: success = extraction corroborated,
             failure = opened but nothing usable / contradicted. Biases OPEN.
    watches  stored fact queries re-run on a schedule; notify on change

Lookup is lexical in v0 (exact entity/attribute, else fuzzy question match).
The DynamicBlobStore index (src/conversation_memory.py) plugs in once the
Planck encoder is in the loop (G3).
"""

import json
import sqlite3
import time
from difflib import SequenceMatcher
from pathlib import Path

from .util import content_words, norm_text

DAY = 86400.0

# attribute keyword -> TTL days. First match wins; None = never expires.
TTL_RULES = (
    (("price", "cost", "fee", "deal", "stock", "availability"), 1),
    (("hours", "open", "opening", "closing", "schedule", "weather"), 7),
    (("ceo", "president", "leader", "population", "rating", "version", "latest"), 90),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT, question_norm TEXT, entity TEXT, attribute TEXT, answer_type TEXT,
    value TEXT, source_url TEXT, domain TEXT, context TEXT,
    retrieved_at REAL, ttl_days REAL, p REAL, verified INTEGER, task_id TEXT
);
CREATE INDEX IF NOT EXISTS facts_ea ON facts(entity, attribute);
CREATE TABLE IF NOT EXISTS domains (domain TEXT PRIMARY KEY, a REAL, b REAL);
CREATE TABLE IF NOT EXISTS watches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT, answer_type TEXT, entity TEXT, attribute TEXT,
    condition TEXT, last_value TEXT, last_checked REAL, created REAL
);
"""


def ttl_for(attribute: str | None, question: str, override: float | None = None) -> float | None:
    if override is not None:
        return override
    hay = norm_text(f"{attribute or ''} {question}")
    for keys, days in TTL_RULES:
        if any(k in hay.split() for k in keys):
            return days
    return None


class Store:
    def __init__(self, path: str | Path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # the chat server serializes turns behind a lock, so cross-thread use is safe
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.clock = clock

    def close(self):
        self.db.close()

    # ── facts ────────────────────────────────────────────────────────────
    def add_fact(self, *, question, value, answer_type, source_url, domain, context,
                 p, verified=False, entity=None, attribute=None, ttl_days=None, task_id=None) -> int:
        cur = self.db.execute(
            "INSERT INTO facts (question, question_norm, entity, attribute, answer_type, value,"
            " source_url, domain, context, retrieved_at, ttl_days, p, verified, task_id)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (question, norm_text(question), norm_text(entity or "") or None,
             norm_text(attribute or "") or None, answer_type, value, source_url, domain,
             context, self.clock(), ttl_for(attribute, question, ttl_days), float(p),
             int(bool(verified)), task_id))
        self.db.commit()
        return cur.lastrowid

    def is_fresh(self, row) -> bool:
        ttl = row["ttl_days"]
        return ttl is None or (self.clock() - row["retrieved_at"]) < ttl * DAY

    def lookup(self, question: str, entity: str | None = None, attribute: str | None = None,
               answer_type: str | None = None, min_ratio: float = 0.85, limit: int = 5) -> list[dict]:
        """Fresh stored facts matching the query, best first."""
        if entity and attribute:
            rows = self.db.execute(
                "SELECT * FROM facts WHERE entity=? AND attribute=? ORDER BY retrieved_at DESC",
                (norm_text(entity), norm_text(attribute))).fetchall()
            scored = [(1.0, r) for r in rows]
        else:
            # Fuzzy path: the SAME content words (order/stopwords may differ). A near-identical
            # string with a different subject ("capital of Canada" vs "capital of Brazil")
            # must never hit; that is a confidently wrong answer served from memory.
            qn = norm_text(question)
            qset = set(content_words(question))
            rows = self.db.execute("SELECT * FROM facts ORDER BY retrieved_at DESC LIMIT 5000").fetchall()
            scored = []
            for r in rows:
                if set(content_words(r["question"])) != qset:
                    continue
                sim = SequenceMatcher(None, qn, r["question_norm"]).ratio()
                if sim >= min_ratio:
                    scored.append((sim, r))
        out = []
        for sim, r in sorted(scored, key=lambda x: (-x[0], -x[1]["p"])):
            if answer_type and r["answer_type"] != answer_type:
                continue
            if not self.is_fresh(r):
                continue
            d = dict(r)
            d["match"] = round(sim, 3)
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def n_facts(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # ── domain prior ─────────────────────────────────────────────────────
    def domain_prior(self, domain: str) -> float:
        row = self.db.execute("SELECT a, b FROM domains WHERE domain=?", (domain,)).fetchone()
        a, b = (row["a"], row["b"]) if row else (1.0, 1.0)
        return a / (a + b)

    def update_domain(self, domain: str, success: bool, weight: float = 1.0):
        row = self.db.execute("SELECT a, b FROM domains WHERE domain=?", (domain,)).fetchone()
        a, b = (row["a"], row["b"]) if row else (1.0, 1.0)
        a, b = (a + weight, b) if success else (a, b + weight)
        self.db.execute("INSERT OR REPLACE INTO domains (domain, a, b) VALUES (?,?,?)", (domain, a, b))
        self.db.commit()

    def top_domains(self, n: int = 10) -> list[tuple[str, float, float]]:
        rows = self.db.execute("SELECT domain, a, b FROM domains").fetchall()
        ranked = sorted(((r["domain"], r["a"] / (r["a"] + r["b"]), r["a"] + r["b"] - 2) for r in rows),
                        key=lambda x: (-x[1], -x[2]))
        return ranked[:n]

    # ── watches ──────────────────────────────────────────────────────────
    def add_watch(self, question, answer_type, condition: dict | None = None,
                  entity=None, attribute=None) -> int:
        cur = self.db.execute(
            "INSERT INTO watches (question, answer_type, entity, attribute, condition, last_value,"
            " last_checked, created) VALUES (?,?,?,?,?,?,?,?)",
            (question, answer_type, entity, attribute, json.dumps(condition or {}), None, None, self.clock()))
        self.db.commit()
        return cur.lastrowid

    def watches(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM watches ORDER BY id")]

    def update_watch(self, watch_id: int, value: str | None):
        self.db.execute("UPDATE watches SET last_value=?, last_checked=? WHERE id=?",
                        (value, self.clock(), watch_id))
        self.db.commit()
