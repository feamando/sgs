"""
Writers (round 5): turn the consolidated answer into the short AI answer the user reads.

The writer gets ONLY the question, the chosen value, its confidence band, and the numbered
sources (with any conflicting one). It never decides the answer, so writing is a SKILL that is
trained once and does not need retraining as the world changes: the world arrives in the sources.

    template   deterministic sentences per band (no model; faithful by construction)
    gemma      Gemma 4 E4B (local teacher; also writes the one-time training set for the small writers)
    hertz      Hertz 1.2 (640M) fine-tuned once on Gemma-written, faithfulness-filtered examples
    planck     Planck 1.3 (100M), same

Every model answer passes `faithful()`: it must contain the chosen value, cite only listed sources,
and every number and name in it must appear in the question or the sources. A failing answer is
replaced by the template answer and the fallback is counted, so the card never shows an invention.
"""

import json
import re
import time
from pathlib import Path

from .candidates import _entities
from .metrics import is_correct
from .util import STOPWORDS, norm_text, read_jsonl

MAX_SOURCES = 3
SNIPPET_CHARS = 220
ALLOWED_WORDS = {"according", "possibly", "probably", "unclear", "however", "although", "while", "sources", "source",
                 "both", "reports", "report", "reported", "confirm", "confirmed", "cannot", "be", "the", "a", "an",
                 "low", "high", "confidence", "trust", "says", "say", "state", "states", "stated", "it", "this",
                 "answer", "unverified", "only", "one", "other", "others", "january", "february", "march", "april",
                 "may", "june", "july", "august", "september", "october", "november", "december"}


def source_name(domain: str) -> str:
    """bbc.com -> BBC, meduza.io -> Meduza, en.wikipedia.org -> Wikipedia."""
    d = (domain or "").lower().removeprefix("www.")
    parts = d.split(".")
    core = parts[-3] if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov", "ac", "net") else (
        parts[-2] if len(parts) >= 2 else d)
    if len(parts) >= 3 and parts[-2] in ("wikipedia", "wikimedia"):
        core = parts[-2]
    return core.upper() if len(core) <= 4 else core.capitalize()


def writer_sources(cons: dict, snippets: dict[str, str]) -> list[dict]:
    """Up to MAX_SOURCES supporting families (most trusted first) + the strongest conflict."""
    out = []
    for f in cons.get("families", [])[: MAX_SOURCES - (1 if cons.get("conflicts") else 0)]:
        ctx = f.get("context") or snippets.get(f.get("url"), "")
        out.append({"domain": f["domain"], "name": source_name(f["domain"]), "url": f.get("url"),
                    "trust": f["trust"]["effective"], "context": ctx[:SNIPPET_CHARS], "says": None})
    for c in cons.get("conflicts", [])[:1]:
        ctx = c.get("context") or snippets.get(c.get("url"), "")
        out.append({"domain": c["domain"], "name": source_name(c["domain"]), "url": c.get("url"),
                    "trust": c["trust"]["effective"], "context": ctx[:SNIPPET_CHARS], "says": c["value"]})
    return out


def writer_prompt(question: str, cons: dict, sources: list[dict]) -> str:
    lines = [f"QUESTION: {question}", f"ANSWER: {cons['value']}",
             f"CONFIDENCE: {cons['confidence']:.0f}/10 {cons['band']}", "SOURCES:"]
    for i, s in enumerate(sources, 1):
        says = f" says {s['says']}" if s["says"] else ""
        lines.append(f"[{i}] {s['name']} ({s['trust']:.0f}/10){says}: {s['context']}")
    lines.append("ANSWER TEXT:")
    return "\n".join(lines)


def faithful(text: str, question: str, value: str, sources: list[dict], answer_type: str) -> tuple[bool, list[str]]:
    problems = []
    t = (text or "").strip()
    if not t:
        return False, ["empty"]
    hay_raw = " ".join([question, value] + [f"{s['name']} {s['domain']} {s['context']} {s.get('says') or ''}" for s in sources])
    if not (norm_text(value) in norm_text(t) or is_correct(t, [value], answer_type)):
        problems.append(f"does not state the answer {value!r}")
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", t)]
    if any(c < 1 or c > len(sources) for c in cites):
        problems.append("cites a source that is not listed")
    body = re.sub(r"\[\d+\]", " ", t)
    have_nums = set(re.findall(r"\d+", hay_raw))
    for n in re.findall(r"\d+", body):
        if n not in have_nums:
            problems.append(f"number {n} is not in the sources")
    hay_words = set(norm_text(hay_raw).split())
    for sent in re.split(r"(?<=[.!?])\s+", body):
        for ent, _off in _entities(sent):
            for w in norm_text(ent).split():
                if w not in hay_words and w not in STOPWORDS and w not in ALLOWED_WORDS:
                    problems.append(f"name {ent!r} is not in the sources")
                    break
    if len(t.split()) > 60:
        problems.append("longer than 60 words")
    return not problems, problems


def template_text(cons: dict, sources: list[dict]) -> str:
    v = cons["value"]
    sup = [(i, s) for i, s in enumerate(sources, 1) if not s["says"]]
    con = [(i, s) for i, s in enumerate(sources, 1) if s["says"]]
    names = " and ".join(f"{s['name']} [{i}]" for i, s in sup[:2])
    tail = f" {con[0][1]['name']} [{con[0][0]}] says {con[0][1]['says']}." if con else ""
    b = cons["band"]
    if b == "high":
        return f"{v}. {names} report this.{tail}"
    if b == "good":
        return f"{v}, according to {names}.{tail}"
    if b == "low":
        return f"Possibly {v}, according to {names}; I could not confirm it well.{tail}"
    return f"Unclear. The sources I found say {v} ({names}), which I cannot confirm.{tail}"


class TemplateWriter:
    name = "template"

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def generate(self, prompt: str) -> str | None:
        return None

    def write(self, question: str, answer_type: str, cons: dict, snippets: dict[str, str]) -> dict:
        t0 = time.perf_counter()
        sources = writer_sources(cons, snippets)
        templ = template_text(cons, sources)
        out = {"writer": self.name, "sources": sources, "template": templ}
        gen = self.generate(writer_prompt(question, cons, sources))
        if gen is None:
            out.update(text=templ, faithful=True, problems=[], fallback=False)
        else:
            ok, probs = faithful(gen, question, cons["value"], sources, answer_type)
            out.update(text=gen if ok else templ, generated=gen, faithful=ok, problems=probs, fallback=not ok)
        out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out


WRITER_SYSTEM = """You write the final answer of a search assistant from the given answer and sources only.
One or two sentences, under 45 words. Start with the answer. Cite sources as [n].
Match the confidence: high -> state it plainly; good -> "according to"; low -> start with "Possibly";
sceptical -> start with "Unclear:" and say it cannot be confirmed. If a source is listed as saying
something else, mention it briefly. Never add any fact, number or name that is not in the sources."""


class LLMWriter(TemplateWriter):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.name = policy.name
        self.usage = policy.usage

    def generate(self, prompt: str) -> str | None:
        return self.policy._complete(WRITER_SYSTEM, prompt.rsplit("\nANSWER TEXT:", 1)[0]).strip().split("\n")[0]


class SGSWriter(TemplateWriter):
    """A fine-tuned SGSLanguageModel (Planck 1.3 / Hertz 1.2): greedy decoding, stops at the end token."""

    def __init__(self, checkpoint: str, tokenizer: str | None = None, device: str | None = None, max_new: int = 48):
        super().__init__()
        import sentencepiece as spm
        import torch
        from scripts.generate import infer_arch
        from src.sgs_lm import SGSLanguageModel, migrate_state_dict
        self.torch = torch
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tok = tokenizer or ck.get("tokenizer")
        self.sp = spm.SentencePieceProcessor(model_file=str(tok))
        state = migrate_state_dict(ck["model"] if "model" in ck else ck)
        arch = infer_arch(state)
        self.max_len = arch["max_len"]
        self.model = SGSLanguageModel(**arch)
        self.model.load_state_dict(state)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_new = min(max_new, self.max_len // 3)
        self.name = ck.get("writer_name", "sgs-writer")
        self.eos = END_MARK

    def generate(self, prompt: str) -> str | None:
        torch = self.torch
        ids = fit_prompt(self.sp, prompt, self.max_len - self.max_new)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = []
        with torch.no_grad():
            for _ in range(self.max_new):
                logits = self.model(x[:, -self.max_len:])[0, -1]
                nid = int(logits.argmax())
                if nid == self.sp.eos_id():
                    break
                out.append(nid)
                x = torch.cat([x, torch.tensor([[nid]], device=self.device)], dim=1)
                text = self.sp.decode(out)
                if END_MARK in text:
                    break
        text = self.sp.decode(out)
        return text.split(END_MARK)[0].strip()


END_MARK = "<END>"


def fit_prompt(sp, prompt: str, budget: int) -> list[int]:
    """Shorten source snippets (never the question or answer) until the prompt fits the context."""
    ids = sp.encode(prompt, out_type=int)
    if len(ids) <= budget:
        return ids
    lines = prompt.split("\n")
    for cut in (160, 110, 70, 40):
        short = [re.sub(r"^(\[\d+\][^:]*: )(.*)$", lambda m: m.group(1) + m.group(2)[:cut], l) for l in lines]
        ids = sp.encode("\n".join(short), out_type=int)
        if len(ids) <= budget:
            return ids
    return ids[-budget:]


def make_writer(name: str, **kw):
    if name == "template":
        return TemplateWriter()
    if name in ("gemma", "bedrock"):
        from .policies import make_policy
        return LLMWriter(make_policy(name, **{k: v for k, v in kw.items() if k in ("gemma_path", "model_id", "region", "profile")}))
    if name in ("hertz", "planck"):
        path = kw.get("checkpoint") or f"checkpoints/writer_{name}/best.pt"
        if not Path(path).exists():
            raise SystemExit(f"no fine-tuned {name} writer at {path}: run `planck3.py writer train --base {name}` first")
        w = SGSWriter(path)
        w.name = f"{name}-writer"
        return w
    raise ValueError(f"unknown writer {name}")


# ── one-time training of the small writers ────────────────────────────────
def train_writer(data: Path, base_ckpt: str, tokenizer: str, out_dir: Path, name: str, epochs: int = 4,
                 lr: float = 5e-5, batch_size: int = 8, seed: int = 0) -> dict:
    """
    Full fine-tune of an SGSLanguageModel on (prompt, answer) pairs; the loss covers the answer
    tokens only (the prompt is context). Best checkpoint by validation loss.
    """
    import random
    import sentencepiece as spm
    import torch
    import torch.nn.functional as F
    from scripts.generate import infer_arch
    from src.sgs_lm import SGSLanguageModel, migrate_state_dict
    from .util import write_json
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    state = migrate_state_dict(ck["model"] if "model" in ck else ck)
    arch = infer_arch(state)
    if arch["vocab_size"] != sp.get_piece_size():
        raise SystemExit(f"tokenizer vocab {sp.get_piece_size()} != checkpoint vocab {arch['vocab_size']} ({tokenizer})")
    model = SGSLanguageModel(**arch)
    model.load_state_dict(state)
    model.to(device)
    rows = [r for r in read_jsonl(data) if r.get("faithful")]
    rng.shuffle(rows)
    n_val = max(1, len(rows) // 10)
    pad = sp.pad_id() if sp.pad_id() >= 0 else 0

    ans_budget = min(70, arch["max_len"] // 3)

    def encode(r):
        p = fit_prompt(sp, r["prompt"], arch["max_len"] - ans_budget)
        a = sp.encode(" " + r["target"] + " " + END_MARK, out_type=int)[:ans_budget]
        return p + a, len(p)
    enc = [encode(r) for r in rows]
    val, tr = enc[:n_val], enc[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(tr) // batch_size + 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))

    def batch_loss(batch):
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), L - 1))
        for i, (x, plen) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, plen - 1:len(x) - 1] = 1.0           # predict answer tokens only
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(ids[:, :-1]).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none")
        return (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, log = float("inf"), []
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        t0, tot, nb = time.time(), 0.0, 0
        for s in range(0, len(tr), batch_size):
            loss = batch_loss(tr[s:s + batch_size])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        model.eval()
        with torch.no_grad():
            vl = sum(batch_loss(val[s:s + batch_size]).item() for s in range(0, len(val), batch_size)) / max(1, (len(val) + batch_size - 1) // batch_size)
        log.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_loss": vl, "s": round(time.time() - t0, 1)})
        print(f"  [writer:{name}] epoch {ep}: train {tot / max(nb, 1):.3f} val {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "tokenizer": str(tokenizer), "writer_name": f"{name}-writer",
                        "base": str(base_ckpt), "val_loss": vl}, out_dir / "best.pt")
    meta = {"name": name, "base": str(base_ckpt), "examples": len(rows), "train": len(tr), "val": len(val),
            "best_val_loss": best, "epochs": log, "params": sum(p.numel() for p in model.parameters())}
    write_json(out_dir / "train_log.json", meta)
    return meta
