"""
Scoring + calibration.

Correctness rules (per answer_type), documented so the gate is auditable:
    year     exact 4-digit year match
    number   numeric value within 0.5% of any gold (units ignored)
    date     normalized string equal, or same year+day digits
    entity   normalized gold == answer, or one contains the other with the
             shorter side >= 1 content word (so "Miyazaki" ~ "Hayao Miyazaki")
    text     every gold content word appears in the answer
"""

import math
import re

from .util import content_words, norm_text

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(s: str) -> float | None:
    s = str(s).replace(",", "").replace(" ", "").replace(" ", "").replace(" ", "")
    m = _NUM.search(s)
    return float(m.group(0)) if m else None


def is_correct(answer: str | None, gold: list[str], answer_type: str) -> bool:
    if answer is None or not gold:
        return False
    for g in gold:
        if answer_type == "year":
            m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", str(answer))
            if m and m.group(1) == str(g).strip():
                return True
        elif answer_type == "number":
            a, b = parse_number(answer), parse_number(g)
            if a is not None and b is not None and abs(a - b) <= 0.005 * max(abs(b), 1e-9):
                return True
        elif answer_type == "date":
            if norm_text(answer) == norm_text(g):
                return True
            da, dg = re.findall(r"\d+", str(answer)), re.findall(r"\d+", str(g))
            if da and sorted(da) == sorted(dg):
                return True
        elif answer_type == "entity":
            a, b = norm_text(answer), norm_text(g)
            if a == b:
                return True
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            if content_words(short) and re.search(rf"\b{re.escape(short)}\b", long_):
                return True
        else:
            if all(w in norm_text(answer).split() for w in content_words(g)):
                return True
    return False


def ece(probs: list[float], correct: list[bool], n_bins: int = 10) -> float | None:
    """Expected calibration error over answered items (equal-width bins)."""
    if not probs:
        return None
    bins = [[] for _ in range(n_bins)]
    for p, c in zip(probs, correct):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, c))
    total = len(probs)
    err = 0.0
    for b in bins:
        if b:
            conf = sum(p for p, _ in b) / len(b)
            acc = sum(1 for _, c in b if c) / len(b)
            err += len(b) / total * abs(conf - acc)
    return err


def fit_temperature(logits_per_item: list[list[float]], gold_index: list[int]) -> float:
    """1-D grid search for the NLL-minimising softmax temperature (held-out data only)."""
    best_t, best_nll = 1.0, math.inf
    for t in [0.25 * 1.15 ** i for i in range(40)]:
        nll = 0.0
        for logits, g in zip(logits_per_item, gold_index):
            m = max(logits)
            z = sum(math.exp((x - m) / t) for x in logits)
            nll -= (logits[g] - m) / t - math.log(z)
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def summarize(records: list[dict]) -> dict:
    """records: [{family, answered, correct, p, steps, web_calls, decision_ms}] per scored unit."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    answered = [r for r in records if r["answered"]]
    right = [r for r in answered if r["correct"]]
    probs = [r["p"] for r in answered]
    out = {
        "n": n,
        "success": len(right) / n,
        "answered_rate": len(answered) / n,
        "wrong_when_answered": (len(answered) - len(right)) / len(answered) if answered else 0.0,
        "abstain_rate": 1 - len(answered) / n,
        "ece": ece(probs, [r["correct"] for r in answered]),
        "mean_steps": sum(r["steps"] for r in records) / n,
        "mean_web_calls": sum(r["web_calls"] for r in records) / n,
    }
    ms = [m for r in records for m in r.get("decision_ms", [])]
    out["mean_decision_ms"] = sum(ms) / len(ms) if ms else None
    by_fam = {}
    for fam in sorted({r["family"] for r in records}):
        sub = [r for r in records if r["family"] == fam]
        by_fam[fam] = {"n": len(sub), "success": sum(1 for r in sub if r["answered"] and r["correct"]) / len(sub)}
    out["by_family"] = by_fam
    return out
