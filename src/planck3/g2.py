"""
G2, redesigned (2026-10-07): learn the decisions from KNOWN ANSWERS, not from a teacher.

The first two box runs showed the Gemma teacher is now worse than the deterministic
heuristic on G0 (85.1% vs 93.6%): its errors are subject echoes and abstains, so
imitating it would teach Planck those errors. Instead:

    collect   for each training question (Wikidata, disjoint from every benchmark):
              SEARCH -> snippet candidates (decision point 1), then read the most
              on-subject page -> page candidates (decision point 2). Each candidate is
              labelled right/wrong against the gold answer. No model in the loop.
    embed     encode question + "value. context" texts (hash control | Planck)
    train     CandidateScorer: per candidate [question, candidate, features] -> logit,
              plus a NULL logit ("none of these is right"), multi-positive NLL,
              temperature-calibrated on held-out questions
    policy    PlanckPolicy plugs into the harness: EXTRACT the top candidate when its
              calibrated p >= 0.5, otherwise read a page / abstain; ANSWER with that p
"""

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from .actions import Decision
from .candidates import snippet_candidates, span_candidates, title_aboutness
from .metrics import is_correct
from .policies import HeuristicPolicy, Policy
from .util import REPO_ROOT, content_words, norm_text, read_json, read_jsonl, write_json
from .web import extract_jsonld, extract_main_text

G2_DIR = REPO_ROOT / "data" / "planck3" / "g2"
ANSWER_TYPES = ("year", "number", "date", "entity", "text")
N_FEAT = 9 + len(ANSWER_TYPES)
EXTRACT_THRESHOLD = 0.5
MAX_PAGES = 2  # fetch budget per question (declared 2026-10-07, before the box run): unsure after 2 pages -> abstain


# ── features (identical at train and decision time) ──────────────────────
def cand_text(c: dict) -> str:
    return f"{c['value']}. {c.get('context', '')}"[:400]


def features(c: dict, question: str, answer_type: str, i: int, n: int, from_snippet: bool) -> list[float]:
    q = set(content_words(question))
    v = set(norm_text(c["value"]).split())
    def num(k, default):  # stored candidates may carry None for fields their source lacks
        v = c.get(k)
        return float(default if v is None else v)
    f = [num("score", 0.0), num("margin", 0.0), num("about", 0.5),
         math.log1p(len(c.get("domains") or [1])), math.log1p(num("support", 1)), i / max(n - 1, 1),
         1.0 if v and v <= q else 0.0, 1.0 if from_snippet else 0.0, 1.0 / (1 + i)]
    return f + [1.0 if answer_type == t else 0.0 for t in ANSWER_TYPES]


# ── collect ──────────────────────────────────────────────────────────────
def collect(tasks: list[dict], web, out_path: Path, limit: int = 0):
    """Two labelled decision points per question. Resumable: skips task ids already collected."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [t for t in tasks if t["family"] == "fact" and t["id"] not in done][: limit or None]
    print(f"[g2] collecting {len(todo)} questions ({len(done)} already done) -> {out_path}")
    t0, empty = time.time(), 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, t in enumerate(todo, 1):
            q, at, gold = t["question"], t["answer_type"], t["gold"]
            results = web.search(q)
            if not results:
                empty += 1
            points = []
            snip = snippet_candidates(q, results, at)
            points.append(("snippet", snip))
            # the page a sensible policy would read: most on-subject title, then rank
            ranked = sorted(enumerate(results), key=lambda x: (-title_aboutness(q, x[1].get("title", "")), x[0]))
            if ranked:
                r = ranked[0][1]
                page = web.fetch(r["url"])
                if page.get("html"):
                    text = extract_main_text(page["html"], r["url"])
                    points.append(("page", span_candidates(q, text, at, url=r["url"],
                                                           extra_lines=extract_jsonld(page["html"]))))
            for src, cands in points:
                if not cands:
                    continue
                labels = [is_correct(c["value"], gold, at) for c in cands]
                f.write(json.dumps({"task_id": t["id"], "question": q, "answer_type": at, "source": src,
                                    "cands": [{k: c.get(k) for k in ("value", "context", "score", "margin", "about",
                                                                     "domains", "support", "url")} for c in cands],
                                    "labels": labels}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)} ({rate:.1f}s/question, ~{rate * (len(todo) - i) / 60:.0f} min left; "
                      f"{empty} empty searches)", flush=True)
    if todo and empty / len(todo) > 0.2:
        print(f"  ! {empty}/{len(todo)} searches came back empty: the search engine may be rate-limiting")


# ── embed ────────────────────────────────────────────────────────────────
def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed(points_path: Path, encoder_name: str, out_dir: Path, **enc_kw):
    from .encoders import make_encoder
    pts = read_jsonl(points_path)
    texts = sorted({p["question"] for p in pts} | {cand_text(c) for p in pts for c in p["cands"]})
    enc = make_encoder(encoder_name, **enc_kw)
    t0 = time.time()
    E = enc.encode(texts, progress=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"emb_{encoder_name}.npy", E)
    write_json(out_dir / f"emb_{encoder_name}_index.json", {_key(t): i for i, t in enumerate(texts)})
    print(f"[g2] emb_{encoder_name}: {E.shape} in {time.time() - t0:.0f}s")


# ── model ────────────────────────────────────────────────────────────────
def make_scorer(dim: int, hidden: int = 128, dropout: float = 0.2):
    import torch.nn as nn

    class CandidateScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pq = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden))
            self.pc = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden))
            self.pf = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.GELU())
            self.mlp = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.null = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        def forward(self, e_q, e_c, feats):
            """e_q [B,D], e_c [B,C,D], feats [B,C,F] -> logits [B, C+1] (last = NULL: none is right)."""
            import torch
            q = self.pq(e_q)
            c = self.pc(e_c)
            x = torch.cat([c, q.unsqueeze(1) * c, (q.unsqueeze(1) - c).abs(), self.pf(feats)], -1)
            return torch.cat([self.mlp(x).squeeze(-1), self.null(q)], -1)

    return CandidateScorer()


def _tensors(batch, E, index, device):
    import torch
    C = max(len(p["cands"]) for p in batch)
    B, D = len(batch), E.shape[1]
    e_q = torch.zeros(B, D)
    e_c = torch.zeros(B, C, D)
    ft = torch.zeros(B, C, N_FEAT)
    mask = torch.zeros(B, C + 1, dtype=torch.bool)
    pos = torch.zeros(B, C + 1, dtype=torch.bool)
    for r, p in enumerate(batch):
        e_q[r] = torch.from_numpy(E[index[_key(p["question"])]].astype(np.float32))
        n = len(p["cands"])
        for i, c in enumerate(p["cands"]):
            e_c[r, i] = torch.from_numpy(E[index[_key(cand_text(c))]].astype(np.float32))
            ft[r, i] = torch.tensor(features(c, p["question"], p["answer_type"], i, n, p["source"] == "snippet"))
        mask[r, :n] = True
        mask[r, C] = True                      # NULL is always a legal choice
        lab = p["labels"]
        if any(lab):
            pos[r, :n] = torch.tensor(lab)
        else:
            pos[r, C] = True                   # nothing on offer is right
    return e_q.to(device), e_c.to(device), ft.to(device), mask.to(device), pos.to(device)


def _nll(logits, pos, mask):
    import torch
    neg = torch.finfo(logits.dtype).min
    a = logits.masked_fill(~mask, neg)
    g = logits.masked_fill(~(pos & mask), neg)
    return (torch.logsumexp(a, -1) - torch.logsumexp(g, -1)).mean()


def train(points_path: Path, emb_dir: Path, encoder_name: str, out_dir: Path, seed: int = 0,
          epochs: int = 12, lr: float = 1e-3, patience: int = 3):
    import torch
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = read_jsonl(points_path)
    E = np.load(emb_dir / f"emb_{encoder_name}.npy")
    index = read_json(emb_dir / f"emb_{encoder_name}_index.json")
    tids = sorted({p["task_id"] for p in pts})
    rng.shuffle(tids)
    val_ids = set(tids[: max(1, len(tids) // 10)])          # held-out QUESTIONS, not points
    tr = [p for p in pts if p["task_id"] not in val_ids]
    va = [p for p in pts if p["task_id"] in val_ids]
    model = make_scorer(E.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    best, bad, log = -1.0, 0, []
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        tot = 0.0
        for s in range(0, len(tr), 64):
            e_q, e_c, ft, m, pos = _tensors(tr[s:s + 64], E, index, device)
            loss = _nll(model(e_q, e_c, ft), pos, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        acc = _val_top1(model, va, E, index, device)
        log.append({"epoch": ep, "train_nll": tot / max(1, len(tr) // 64), "val_top1": acc})
        print(f"  epoch {ep}: train_nll={log[-1]['train_nll']:.4f} val_top1={acc:.4f}", flush=True)
        if acc <= best:
            bad += 1
            if bad >= patience:
                break
            continue
        best, bad = acc, 0
        torch.save({"model": model.state_dict(), "dim": E.shape[1], "encoder": encoder_name, "seed": seed}, out_dir / "head.pt")
    ck = torch.load(out_dir / "head.pt", map_location=device)
    model.load_state_dict(ck["model"])
    T = _fit_temperature(model, va, E, index, device)
    ck["temperature"] = T
    torch.save(ck, out_dir / "head.pt")
    cal = _val_calibration(model, va, E, index, device, T)
    cal.update(_val_choice(model, va, E, index, device))
    base = _heuristic_val_top1(va)
    write_json(out_dir / "train_log.json", {"epochs": log, "best_val_top1": best, "temperature": T,
                                            "val_heuristic_top1": base, **cal, "encoder": encoder_name, "seed": seed,
                                            "n_train_points": len(tr), "n_val_points": len(va)})
    print(f"[g2] head:{encoder_name} s{seed}: choice accuracy {cal['val_choice_acc']:.3f} vs deterministic ranker "
          f"{base:.3f} on {cal['val_points_with_gold']} val points with a right candidate; 'none' correct "
          f"{cal['val_none_acc']:.3f} on {cal['val_points_without_gold']}; T={T:.2f}, ECE {cal['val_ece']:.3f} -> {out_dir}")


def _scores(model, pts, E, index, device, T=1.0):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(pts), 128):
            batch = pts[s:s + 128]
            e_q, e_c, ft, m, pos = _tensors(batch, E, index, device)
            lg = model(e_q, e_c, ft).masked_fill(~m, -1e4) / T
            pr = torch.softmax(lg, -1)
            for r, p in enumerate(batch):
                n = len(p["cands"])
                C = lg.shape[1] - 1
                out.append((pr[r, list(range(n)) + [C]].cpu().numpy(), p))
    return out


def _val_top1(model, va, E, index, device) -> float:
    ok = 0
    for pr, p in _scores(model, va, E, index, device):
        k = int(pr.argmax())
        ok += (k == len(p["cands"]) and not any(p["labels"])) or (k < len(p["cands"]) and p["labels"][k])
    return ok / max(len(va), 1)


def _heuristic_val_top1(va) -> float:
    """The deterministic ranker's first candidate, on points where a right candidate EXISTS."""
    has = [p for p in va if any(p["labels"])]
    return sum(1 for p in has if p["labels"][0]) / max(len(has), 1)


def _val_choice(model, va, E, index, device) -> dict:
    """Apples to apples with the ranker: argmax over CANDIDATES only, on points with a right one;
    plus how often 'none of these' is chosen correctly where nothing on offer is right."""
    has_ok = has_n = none_ok = none_n = 0
    for pr, p in _scores(model, va, E, index, device):
        n = len(p["cands"])
        if any(p["labels"]):
            has_n += 1
            has_ok += bool(p["labels"][int(pr[:n].argmax())])
        else:
            none_n += 1
            none_ok += int(pr.argmax()) == n
    return {"val_choice_acc": has_ok / max(has_n, 1), "val_points_with_gold": has_n,
            "val_none_acc": none_ok / max(none_n, 1), "val_points_without_gold": none_n}


def _fit_temperature(model, va, E, index, device) -> float:
    best_t, best = 1.0, math.inf
    raw = _scores(model, va, E, index, device, T=1.0)
    logits = [np.log(np.clip(pr, 1e-12, 1)) for pr, _ in raw]
    for t in [0.3 * 1.12 ** i for i in range(40)]:
        nll = 0.0
        for lg, (_, p) in zip(logits, raw):
            z = lg / t
            z = z - z.max()
            lp = z - np.log(np.exp(z).sum())
            posi = [i for i, y in enumerate(p["labels"]) if y] or [len(p["cands"])]
            nll -= np.log(np.exp(lp[posi]).sum())
        if nll < best:
            best_t, best = t, nll
    return best_t


def _val_calibration(model, va, E, index, device, T) -> dict:
    from .metrics import ece
    probs, corr = [], []
    for pr, p in _scores(model, va, E, index, device, T):
        n = len(p["cands"])
        k = int(pr[:n].argmax()) if n else 0
        probs.append(float(pr[k]))
        corr.append(bool(n and p["labels"][k]))
    return {"val_ece": ece(probs, corr) or 0.0,
            "val_answer_acc_at_0.5": (sum(c for p_, c in zip(probs, corr) if p_ >= 0.5) /
                                      max(sum(1 for p_ in probs if p_ >= 0.5), 1))}


# ── the policy ───────────────────────────────────────────────────────────
class PlanckPolicy(Policy):
    """Learned candidate choice + calibrated confidence; everything else as the heuristic."""
    name = "planck"
    kind = "local"

    def __init__(self, head_path: str, checkpoint: str | None = None, tokenizer: str | None = None):
        super().__init__()
        import torch
        from .encoders import make_encoder
        self.torch = torch
        ck = torch.load(head_path, map_location="cpu", weights_only=False)
        self.encoder_name = ck["encoder"]
        self.name = f"planck-g2:{self.encoder_name}"
        self.enc = make_encoder(self.encoder_name, checkpoint=checkpoint, tokenizer=tokenizer)
        self.model = make_scorer(ck["dim"])
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.T = float(ck.get("temperature", 1.0))
        self.fallback = HeuristicPolicy()
        self._held_p = 0.5

    def _probs(self, obs, spans, from_snippet):
        torch = self.torch
        q, at = obs["question"], obs["answer_type"]
        E = self.enc.encode([q] + [cand_text(c) for c in spans]).astype(np.float32)
        e_q = torch.from_numpy(E[:1])
        e_c = torch.from_numpy(E[1:]).unsqueeze(0)
        ft = torch.tensor([[features(c, q, at, i, len(spans), from_snippet) for i, c in enumerate(spans)]])
        with torch.no_grad():
            pr = torch.softmax(self.model(e_q, e_c, ft)[0] / self.T, -1).numpy()
        return pr  # [n + 1], last = none is right

    def decide(self, obs):
        phase = obs["phase"]
        spans = obs["lists"].get("spans", [])
        if phase in ("results_snip", "page") and spans:
            pr = self._probs(obs, spans, phase == "results_snip")
            k = int(pr[:-1].argmax())
            if pr[k] >= EXTRACT_THRESHOLD:
                self._held_p = float(pr[k])
                return Decision("EXTRACT", k=k, p=self._held_p)
            nxt = self.fallback._best_unopened(obs)
            if nxt is not None and len(obs["opened"]) < MAX_PAGES:
                return Decision("OPEN", k=nxt, p=float(pr[k]))
            return Decision("ABSTAIN", p=float(pr[k]))
        if phase == "extracted":
            p = self._held_p
            if p < 0.7 and self.fallback._best_unopened(obs) is not None:
                return Decision("VERIFY", p=p)
            return Decision("ANSWER", p=p)
        if phase == "verified":
            p = min(0.97, self._held_p + 0.2) if obs["verified"] else self._held_p * 0.8
            return Decision("ANSWER", p=p)
        return self.fallback.decide(obs)  # start / store_hit / results without snippet values


# ── CLI ──────────────────────────────────────────────────────────────────
def add_cli(sub, add_web_args, build_web):
    p = sub.add_parser("g2", help="G2: learn decisions from known answers (collect | embed | train)")
    p.add_argument("stage", choices=["collect", "embed", "train"])
    p.add_argument("--tasks", default=str(REPO_ROOT / "scripts" / "assets" / "planck3_g2_train_tasks.json"))
    p.add_argument("--points", default=str(G2_DIR / "points.jsonl"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--encoder", default="hash", choices=["hash", "planck"])
    p.add_argument("--checkpoint", default="checkpoints/planck13/best.pt")
    p.add_argument("--tokenizer", default="data/wikipedia/tokenizer.model")
    p.add_argument("--seed", type=int, default=0)
    add_web_args(p)

    def run(args):
        if args.stage == "collect":
            collect(read_json(args.tasks)["tasks"], build_web(args), Path(args.points), args.limit)
        elif args.stage == "embed":
            embed(Path(args.points), args.encoder, G2_DIR, checkpoint=args.checkpoint, tokenizer=args.tokenizer)
        else:
            train(Path(args.points), G2_DIR, args.encoder,
                  REPO_ROOT / "results" / "planck3" / f"g2_head_{args.encoder}_s{args.seed}", seed=args.seed)
    p.set_defaults(fn=run)
