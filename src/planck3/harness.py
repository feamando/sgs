"""
The agent loop (plan section 5):

    query -> LOOKUP store -> (miss) SEARCH -> OPEN k -> EXTRACT k -> [VERIFY]
          -> ANSWER card | ABSTAIN

The policy only ever sees an observation and returns a typed Decision; every
side effect (search, fetch, extraction, store writes) happens here, in code.
Each step is logged so successful teacher trajectories can be distilled (G1/G2).
"""

import time

from .actions import PHASES, Decision, InvalidDecision, fallback, validate
from .candidates import span_candidates
from .metrics import is_correct, parse_number
from .util import content_words, norm_text
from .web import domain_of, extract_jsonld, extract_main_text

MAX_STEPS = 8


class Harness:
    def __init__(self, policy, web, store, max_steps: int = MAX_STEPS, use_store: bool = True):
        self.policy = policy
        self.web = web
        self.store = store
        self.max_steps = max_steps
        self.use_store = use_store

    # ── one fact question ────────────────────────────────────────────────
    def run_fact(self, question: str, answer_type: str, entity: str | None = None,
                 attribute: str | None = None, ttl_days: float | None = None,
                 task_id: str | None = None, store_answer: bool = True) -> dict:
        st = {"phase": "start", "results": [], "spans": [], "store": [], "held": None,
              "verified": None, "opened": set(), "history": []}
        calls0 = self.web.calls["search"] + self.web.calls["fetch"]
        search0 = self.web.calls["search"]
        usage0 = dict(getattr(self.policy, "usage", {}))
        steps, ms, invalid, trajectory = 0, [], 0, []
        outcome = {"answered": False, "value": None, "p": 0.0, "source_url": None,
                   "context": None, "verified": None, "reason": "step_cap", "from_store": False}

        def finish(**kw):
            outcome.update(kw)

        while steps < self.max_steps:
            obs = self._observation(question, answer_type, st)
            t0 = time.perf_counter()
            d = self.policy.decide(obs)
            ms.append((time.perf_counter() - t0) * 1000)
            valid = True
            try:
                d = validate(d, st["phase"], {k: len(st[k]) for k in ("results", "spans", "store")})
            except InvalidDecision as e:
                invalid += 1
                valid = False
                bad = d
                d = fallback(st["phase"])
                d.meta["invalid"] = f"{bad.action} k={bad.k}: {e}"
            trajectory.append({"step": steps, "phase": st["phase"], "obs": _compact(obs),
                               "decision": d.to_dict(), "valid": valid, "raw": d.raw[:300]})
            steps += 1
            st["history"].append(d.action + (f" {d.k}" if d.k is not None else ""))
            done = self._apply(d, st, question, answer_type, entity, attribute, finish)
            if done:
                break

        if outcome["answered"] and store_answer and not outcome["from_store"]:
            self.store.add_fact(question=question, value=outcome["value"], answer_type=answer_type,
                                source_url=outcome["source_url"], domain=domain_of(outcome["source_url"] or ""),
                                context=outcome["context"], p=outcome["p"], verified=bool(outcome["verified"]),
                                entity=entity, attribute=attribute, ttl_days=ttl_days, task_id=task_id)
        usage = getattr(self.policy, "usage", {})
        outcome.update(steps=steps, decision_ms=ms, invalid=invalid, trajectory=trajectory,
                       web_calls=self.web.calls["search"] + self.web.calls["fetch"] - calls0,
                       search_calls=self.web.calls["search"] - search0,
                       usage={k: usage.get(k, 0) - usage0.get(k, 0) for k in usage})
        return outcome

    def _observation(self, question, answer_type, st) -> dict:
        spec = PHASES[st["phase"]]
        return {"question": question, "answer_type": answer_type, "phase": st["phase"],
                "allowed": list(spec["allowed"]), "pointer": dict(spec["pointer"]),
                "lists": {"results": st["results"], "spans": st["spans"], "store": st["store"]},
                "held": st["held"], "verified": st["verified"], "history": list(st["history"]),
                "opened": set(st["opened"]), "store_size": self.store.n_facts() if self.use_store else 0}

    def _apply(self, d: Decision, st, question, answer_type, entity, attribute, finish) -> bool:
        a = d.action
        if a == "LOOKUP":
            hits = self.store.lookup(question, entity, attribute, answer_type) if self.use_store else []
            if hits:
                st["store"], st["phase"] = hits, "store_hit"
                return False
            return self._search(st, question, finish)
        if a == "SEARCH":
            return self._search(st, question, finish)
        if a == "OPEN":
            return self._open(d.k, st, question, answer_type, finish)
        if a == "EXTRACT":
            span = dict(st["spans"][d.k])
            span["domain"] = domain_of(span["url"])
            st["held"], st["verified"], st["phase"] = span, None, "extracted"
            self.store.update_domain(span["domain"], True, weight=0.5)
            return False
        if a == "VERIFY":
            st["verified"] = self._verify(st, question, answer_type)
            st["phase"] = "verified"
            if st["verified"]:
                self.store.update_domain(st["held"]["domain"], True)
            return False
        if a == "ANSWER":
            if st["phase"] == "store_hit":
                f = st["store"][d.k]
                finish(answered=True, value=f["value"], p=d.p, source_url=f["source_url"],
                       context=f["context"], verified=bool(f["verified"]), reason="store", from_store=True)
            else:
                h = st["held"]
                finish(answered=True, value=h["value"], p=d.p, source_url=h["url"],
                       context=h["context"], verified=st["verified"], reason="answer")
            return True
        finish(answered=False, p=d.p, reason="abstain")
        return True

    def _search(self, st, question, finish) -> bool:
        results = self.web.search(question)
        for r in results:
            r["trust"] = round(self.store.domain_prior(r["domain"]), 3)
        st["results"], st["opened"], st["phase"] = results, set(), "results"
        if not results:
            finish(answered=False, reason="no_results")
            return True
        return False

    def _open(self, k, st, question, answer_type, finish) -> bool:
        r = st["results"][k]
        st["opened"].add(k)
        spans = self._page_spans(r["url"], question, answer_type)
        if not spans:
            self.store.update_domain(r["domain"], False, weight=0.5)
            if len(st["opened"]) >= len(st["results"]):
                finish(answered=False, reason="exhausted")
                return True
            st["spans"], st["phase"] = [], "results"
            return False
        st["spans"], st["held"], st["verified"], st["phase"] = spans, None, None, "page"
        return False

    def _page_spans(self, url, question, answer_type):
        page = self.web.fetch(url)
        if not page.get("html"):
            return []
        text = extract_main_text(page["html"], url)
        return span_candidates(question, text, answer_type, url=url,
                               extra_lines=extract_jsonld(page["html"]))

    def _verify(self, st, question, answer_type) -> bool:
        """Deterministic: open the most query-relevant unopened result, look for the held value."""
        q = set(content_words(question))
        best, best_s = None, -1.0
        for i, r in enumerate(st["results"]):
            if i in st["opened"] or domain_of(r["url"]) == st["held"]["domain"]:
                continue
            s = len(q & set(norm_text(f"{r['title']} {r['snippet']}").split())) + r.get("trust", 0.5)
            if s > best_s:
                best, best_s = i, s
        if best is None:
            return False
        st["opened"].add(best)
        spans = self._page_spans(st["results"][best]["url"], question, answer_type)
        held = st["held"]["value"]
        return any(is_correct(s["value"], [held], answer_type) for s in spans[:10])

    # ── compare = composed fact cards ────────────────────────────────────
    def run_compare(self, task: dict) -> dict:
        cells, table = [], {}
        for item in task["items"]:
            table[item] = {}
            for att in task["attributes"]:
                q = att["template"].format(item=item)
                res = self.run_fact(q, att["answer_type"], entity=item, attribute=att["name"],
                                    task_id=task["id"])
                res["item"], res["attribute"], res["question"] = item, att["name"], q
                table[item][att["name"]] = res
                cells.append(res)
        return {"cells": cells, "table": table}

    # ── watch = re-run a fact with the network forced, compare to last value ──
    def run_watch(self, watch: dict) -> dict:
        res = self.run_fact(watch["question"], watch["answer_type"], entity=watch["entity"],
                            attribute=watch["attribute"], store_answer=True)
        new, old = res["value"], watch["last_value"]
        cond = __import__("json").loads(watch["condition"] or "{}")
        fired = False
        if res["answered"]:
            if cond.get("op") in ("lt", "gt"):
                v = parse_number(new)
                if v is not None:
                    fired = v < cond["value"] if cond["op"] == "lt" else v > cond["value"]
            else:
                fired = old is not None and norm_text(new) != norm_text(old)
            self.store.update_watch(watch["id"], new)
        return {"watch_id": watch["id"], "question": watch["question"], "old": old, "new": new,
                "fired": fired, "result": res}


def _compact(obs: dict) -> dict:
    """What gets logged per step: enough to re-render the prompt / re-train a head."""
    lists = {}
    for name in set(obs["pointer"].values()):
        lists[name] = obs["lists"].get(name, [])
    return {"question": obs["question"], "answer_type": obs["answer_type"], "allowed": obs["allowed"],
            "pointer": obs["pointer"], "lists": lists, "held": obs["held"], "verified": obs["verified"],
            "history": obs["history"], "opened": sorted(obs["opened"])}


# ── answer cards (deterministic renderers, no generation) ─────────────────
def render_card(question: str, res: dict) -> str:
    if not res["answered"]:
        return f"**{question}**\n  I couldn't verify an answer ({res['reason']})."
    src = domain_of(res["source_url"] or "")
    tag = "verified by a 2nd source" if res.get("verified") else "single source"
    store = ", from memory" if res.get("from_store") else ""
    return (f"**{question}**\n  **{res['value']}**  (confidence {res['p']:.0%}, {tag}{store})\n"
            f"  \"{(res['context'] or '')[:200]}\"\n  Source: {res['source_url']} ({src})")


def render_table(task: dict, out: dict) -> str:
    atts = [a["name"] for a in task["attributes"]]
    lines = ["| item | " + " | ".join(atts) + " |", "|---" * (len(atts) + 1) + "|"]
    for item, row in out["table"].items():
        cells = []
        for a in atts:
            r = row[a]
            cells.append(f"{r['value']} ({domain_of(r['source_url'] or '')})" if r["answered"] else "n/a")
        lines.append(f"| {item} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
