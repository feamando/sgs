"""
Chat surface: the product bet is the behaviour shift from search boxes to chat,
done cheaply. Search already works; what a chat turn adds is (1) a direct answer
instead of links, (2) follow-ups that carry context ("and H&M?", "when was it
founded?"), (3) an answer phrased for a conversation, with its evidence.

All three are deterministic here (typed rules, no generation); each rule is a
small classification the Planck head can learn later (answer type, NEW vs
SWAP_ENTITY vs PRONOUN follow-up), which is exactly a System One decision.
"""

import re
from dataclasses import dataclass, field

from .harness import Harness
from .util import STOPWORDS
from .web import domain_of

_FOLLOW_LEAD = re.compile(r"^\s*(?:and|what about|how about|and what about|and for|same for|also)\b[\s,]*", re.I)
_PRONOUN = re.compile(r"\b(it|its|they|their|them|he|his|she|her|this|that one)\b", re.I)
_WH = re.compile(r"^\s*(who|what|when|where|which|how|in what|in which|is|are|was|were|did|does|do)\b", re.I)


def infer_answer_type(question: str) -> str:
    q = question.lower().strip()
    if re.search(r"\b(what|which) (date|day)\b|\bwhen exactly\b|\bon what date\b", q):
        return "date"
    if re.search(r"\b(what|which|in what|in which) year\b|^\s*when\b|\bwhat year\b", q):
        return "year"
    if re.search(r"\bhow (many|much|tall|high|long|far|big|large|heavy|old|fast|deep|wide)\b", q) or \
       re.search(r"\b(height|population|number|price|cost|size|length|weight|speed|atomic number|"
                 r"capacity|area|distance|temperature|percentage)\b", q):
        return "number"
    if re.search(r"^\s*(who|whom)\b|\bwho (is|was|were|founded|wrote|directed|painted|invented|developed|created)\b", q) or \
       re.search(r"^\s*where\b|\b(which|what) (city|country|town|company|planet|state|continent|person|author|director)\b", q) or \
       re.search(r"\b(capital|headquarter|headquarters|founder|author|director|ceo|inventor)\b", q):
        return "entity"
    return "text"


def main_entity(question: str) -> str | None:
    """Longest capitalized span that is not the sentence-initial wh-word (the question's subject)."""
    toks = re.findall(r"[\w&'’.-]+", question)
    spans, cur = [], []
    for i, t in enumerate(toks):
        t = t.strip(".")
        ok = t[:1].isupper() and not (i == 0 and t.lower() in STOPWORDS) and t.lower() not in ("i",)
        if ok or (cur and t.lower() in ("of", "the", "de", "&") and i + 1 < len(toks) and toks[i + 1][:1].isupper()):
            cur.append(t)
        elif cur:
            spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    return max(spans, key=len) if spans else None


@dataclass
class Turn:
    user: str
    question: str          # the standalone question actually asked (after follow-up rewrite)
    answer_type: str
    entity: str | None
    rewrite: str           # NEW | SWAP_ENTITY | PRONOUN
    result: dict = field(default_factory=dict)


def resolve_followup(utterance: str, prev: Turn | None) -> tuple[str, str, str | None]:
    """-> (standalone question, rewrite kind, entity)."""
    u = utterance.strip()
    if prev is None:
        return u, "NEW", main_entity(u)
    lead = _FOLLOW_LEAD.match(u)
    words = re.findall(r"\w+", u)
    # "and H&M?" / "what about Zara" / bare "Zara?" -> same question, new subject
    if lead or (len(words) <= 3 and not _WH.match(u)):
        new_ent = _FOLLOW_LEAD.sub("", u).strip(" ?.!")
        if new_ent and prev.entity and re.search(re.escape(prev.entity), prev.question, re.I):
            q = re.sub(re.escape(prev.entity), new_ent, prev.question, count=1, flags=re.I)
            return q, "SWAP_ENTITY", new_ent
    # "when was it founded?" -> substitute the carried subject
    if prev.entity and _PRONOUN.search(u) and not main_entity(u):
        q = _PRONOUN.sub(lambda m: prev.entity + ("'s" if m.group(1).lower() in ("its", "their", "his", "her") else ""),
                         u, count=1)
        return q, "PRONOUN", prev.entity
    return u, "NEW", main_entity(u)


def chat_answer(turn: Turn) -> str:
    """A chat reply: direct answer first, then the evidence and source. No generation."""
    r = turn.result
    if not r.get("answered"):
        return ("I couldn't find an answer I'd trust for that. "
                f"(searched for: \"{turn.question}\")")
    conf = r["p"]
    hedge = "" if conf >= 0.75 else "Probably " if conf >= 0.5 else "Not sure, but possibly "
    src = domain_of(r["source_url"] or "")
    extra = ", and a second source agrees" if r.get("verified") else ""
    mem = " (from memory)" if r.get("from_store") else ""
    ctx = (r.get("context") or "").strip()
    return (f"{hedge}**{r['value']}**{mem}.\n"
            f"> {ctx[:220]}\n"
            f"Source: {src}{extra} · confidence {conf:.0%}")


class ChatSession:
    def __init__(self, harness: Harness):
        self.h = harness
        self.turns: list[Turn] = []

    def ask(self, utterance: str) -> Turn:
        prev = self.turns[-1] if self.turns else None
        q, kind, ent = resolve_followup(utterance, prev)
        at = infer_answer_type(q)
        t = Turn(user=utterance, question=q, answer_type=at, entity=ent, rewrite=kind)
        t.result = self.h.run_fact(q, at, entity=None, attribute=None)
        self.turns.append(t)
        return t
