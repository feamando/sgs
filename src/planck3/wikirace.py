"""
G1: offline Wikiracing (Jev's own demo), with free perfect labels.

    build   download Simple English Wikipedia -> link graph (CSR) + title/lead text
    tasks   sample (start, target) pairs, reverse-BFS distances -> gold next-links
            (split by TARGET so test targets are never seen in training)
    embed   encode every article "Title. lead" with an encoder (hash | planck)
    train   pointer head on cached embeddings (fast; GPU optional)
    eval    rollouts on held-out pairs: random / lexical / head:<enc> / gemma teacher
            + step top-1, ECE, per-decision latency -> G1 verdict

Simple English (~250k articles) keeps the whole pipeline to minutes on the 4090
box and needs no Wikipedia API. Pass --lang en for the full graph (hours, ~25 GB).
"""

import bz2
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from .util import REPO_ROOT, read_json, read_jsonl, write_json

WR_DIR = REPO_ROOT / "data" / "planck3" / "wikirace"
DUMP_URL = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"
SKIP_NS = {"file", "image", "category", "wikipedia", "wp", "template", "help", "portal", "user", "talk",
           "media", "special", "wikt", "wiktionary", "commons", "draft", "module", "mediawiki", "s",
           "q", "n", "b", "v", "voy", "species", "meta", "mw", "d", "t", "c", "simple"}
RE_LINK = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")
G1_REL, G1_MS = 0.80, 100.0
MAX_TEACHER_LINKS = 255


# ── build ────────────────────────────────────────────────────────────────
def download(url: str, dest: Path):
    """Streaming download with resume (re-run after a dropped connection)."""
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "Planck3-research/0.1 (+https://github.com/feamando/sgs)"}
    head = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
    total = int(head.headers.get("content-length", 0))
    if total and have >= total:
        print(f"[wikirace] dump present: {dest} ({have / 1e6:.0f} MB)")
        return
    if have:
        headers["Range"] = f"bytes={have}-"
    print(f"[wikirace] downloading {url} -> {dest} ({total / 1e6:.0f} MB, resume at {have / 1e6:.0f} MB)")
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        done = have if mode == "ab" else 0
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % (50 << 20) < (1 << 20):
                    print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)


def norm_title(t: str) -> str:
    t = re.sub(r"[\s_]+", " ", t).strip()
    return t[:1].upper() + t[1:] if t else t


def _is_article_link(target: str) -> bool:
    if ":" in target:
        prefix = target.split(":", 1)[0].strip().lower()
        if prefix in SKIP_NS or (len(prefix) <= 3 and prefix.isalpha()):
            return False
    return bool(target.strip())


def clean_lead(wikitext: str, max_chars: int = 300) -> str:
    s = re.sub(r"(?s)<!--.*?-->", "", wikitext)
    s = re.sub(r"(?is)<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", s)
    for _ in range(6):  # peel nested templates / tables innermost-first
        s2 = re.sub(r"\{\{[^{}]*\}\}", "", s)
        s2 = re.sub(r"(?s)\{\|[^{}]*?\|\}", "", s2)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\[\[(?:File|Image|Category):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", s, flags=re.I)
    s = re.sub(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    for para in s.split("\n"):
        para = para.strip()
        if len(para) >= 40 and not para.startswith(("=", "*", "#", "|", "!", "{", "}")):
            return re.sub(r"\s+", " ", para)[:max_chars]
    return ""


def iter_pages(dump_path: Path):
    import xml.etree.ElementTree as ET
    with bz2.open(dump_path, "rb") as f:
        title = ns = redirect = text = None
        for event, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = el.text or ""
            elif tag == "ns":
                ns = el.text
            elif tag == "redirect":
                redirect = el.get("title")
            elif tag == "text":
                text = el.text or ""
            elif tag == "page":
                yield title, ns, redirect, text
                title = ns = redirect = text = None
                el.clear()


def build_graph(dump_path: Path, out_dir: Path, max_pages: int = 0):
    t0 = time.time()
    redirects, raw_links, leads, titles = {}, [], [], []
    for i, (title, ns, redirect, text) in enumerate(iter_pages(dump_path)):
        if ns != "0" or not title:
            continue
        t = norm_title(title)
        if redirect:
            redirects[t] = norm_title(redirect)
            continue
        if text.lstrip().lower().startswith("#redirect"):
            continue
        links = []
        for m in RE_LINK.finditer(text):
            tgt = m.group(1)
            if _is_article_link(tgt):
                links.append(norm_title(tgt))
        titles.append(t)
        raw_links.append(links)
        leads.append(clean_lead(text))
        if len(titles) % 50000 == 0:
            print(f"  parsed {len(titles):,} articles ({time.time() - t0:.0f}s)", flush=True)
        if max_pages and len(titles) >= max_pages:
            break

    index = {t: i for i, t in enumerate(titles)}

    def resolve(t):
        for _ in range(3):
            if t in index:
                return index[t]
            t = redirects.get(t)
            if t is None:
                return None
        return None

    indptr, indices = [0], []
    for i, links in enumerate(raw_links):
        seen = set()
        for l in links:
            j = resolve(l)
            if j is not None and j != i and j not in seen:
                seen.add(j)
                indices.append(j)
        indptr.append(len(indices))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "graph.npz", indptr=np.array(indptr, dtype=np.int64),
                        indices=np.array(indices, dtype=np.int32))
    write_json(out_dir / "titles.json", titles)
    write_json(out_dir / "leads.json", leads)
    deg = np.diff(indptr)
    info = {"n_articles": len(titles), "n_edges": len(indices), "n_redirects": len(redirects),
            "mean_outdeg": float(deg.mean()), "median_outdeg": float(np.median(deg)),
            "max_outdeg": int(deg.max()), "with_lead": sum(1 for l in leads if l),
            "dump": str(dump_path), "build_s": round(time.time() - t0, 1)}
    write_json(out_dir / "graph_info.json", info)
    print(f"[wikirace] graph: {info}")


class Graph:
    def __init__(self, d: Path = WR_DIR):
        g = np.load(d / "graph.npz")
        self.indptr, self.indices = g["indptr"], g["indices"]
        self.titles = read_json(d / "titles.json")
        self.leads = read_json(d / "leads.json")
        self.n = len(self.titles)

    def out(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def text(self, u: int) -> str:
        return f"{self.titles[u]}. {self.leads[u]}" if self.leads[u] else self.titles[u]

    def reverse_csr(self):
        from scipy.sparse import csr_matrix
        m = csr_matrix((np.ones(len(self.indices), dtype=np.int8), self.indices, self.indptr),
                       shape=(self.n, self.n))
        return m.T.tocsr()


# ── tasks ────────────────────────────────────────────────────────────────
def make_tasks(d: Path, n_targets: int, starts_per_target: int, offpath_per_target: int,
               seed: int, min_dist: int = 2, max_dist: int = 6, min_indeg: int = 5, td: Path | None = None):
    td = Path(td or d)  # tasks dir: the graph is shared, task sets (full / quick) are not
    td.mkdir(parents=True, exist_ok=True)
    from scipy.sparse.csgraph import shortest_path
    g = Graph(d)
    rng = random.Random(seed)
    rev = g.reverse_csr()
    indeg = np.diff(rev.indptr)
    pool = [u for u in range(g.n) if indeg[u] >= min_indeg and g.leads[u]]
    rng.shuffle(pool)
    targets = pool[:n_targets]
    n_val = n_test = max(1, len(targets) // 10)
    split_of = {t: "test" for t in targets[:n_test]}
    split_of.update({t: "val" for t in targets[n_test:n_test + n_val]})
    for t in targets[n_test + n_val:]:
        split_of[t] = "train"
    pairs = {s: [] for s in ("train", "val", "test")}
    steps = {s: [] for s in ("train", "val", "test")}
    t0 = time.time()
    B = 64
    for bstart in range(0, len(targets), B):
        batch = targets[bstart:bstart + B]
        dist = shortest_path(rev, method="D", unweighted=True, directed=True, indices=batch)
        for bi, tgt in enumerate(batch):
            dv = dist[bi]
            split = split_of[tgt]
            by_d = {}
            for k in range(min_dist, max_dist + 1):
                cand = np.flatnonzero(dv == k)
                if len(cand):
                    by_d[k] = cand
            if not by_d:
                continue
            ks = sorted(by_d)
            for s in range(starts_per_target):
                k = ks[s % len(ks)]
                start = int(by_d[k][rng.randrange(len(by_d[k]))])
                pairs[split].append({"target": tgt, "start": start, "dist": k})
                u = start
                while dv[u] > 0:
                    nb = g.out(u)
                    gold = [int(v) for v in nb if dv[v] == dv[u] - 1]
                    steps[split].append({"target": tgt, "node": int(u), "dist": int(dv[u]), "gold": gold,
                                         "cand_dist": _cand_dist(dv, nb)})
                    u = rng.choice(gold)
            finite = np.flatnonzero(np.isfinite(dv) & (dv >= 1) & (dv <= 8))
            for j in range(min(offpath_per_target, len(finite))):
                u = int(finite[rng.randrange(len(finite))])
                nb = g.out(u)
                if len(nb) == 0:
                    continue
                gold = [int(v) for v in nb if dv[v] == dv[u] - 1]
                steps[split].append({"target": tgt, "node": int(u), "dist": int(dv[u]), "gold": gold,
                                     "cand_dist": _cand_dist(dv, nb)})
        print(f"  targets {min(bstart + B, len(targets))}/{len(targets)} ({time.time() - t0:.0f}s)", flush=True)
    for s in pairs:
        with open(td / f"pairs_{s}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in pairs[s])
        with open(td / f"steps_{s}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in steps[s])
    info = {s: {"pairs": len(pairs[s]), "steps": len(steps[s])} for s in pairs}
    info.update({"n_targets": len(targets), "seed": seed, "format": 2})  # 2 = steps carry cand_dist
    write_json(td / "tasks_info.json", info)
    print(f"[wikirace] tasks: {info}")


def _cand_dist(dv, nb) -> list[int]:
    """BFS distance-to-target of every candidate link (aligned with g.out(node)); -1 = unreachable."""
    cd = dv[nb]
    return np.where(np.isfinite(cd), cd, -1).astype(int).tolist()


# ── embed ────────────────────────────────────────────────────────────────
def embed(d: Path, encoder_name: str, **enc_kw):
    from .encoders import make_encoder
    g = Graph(d)
    enc = make_encoder(encoder_name, **enc_kw)
    t0 = time.time()
    E = enc.encode([g.text(u) for u in range(g.n)], progress=True)
    np.save(d / f"emb_{encoder_name}.npy", E)
    print(f"[wikirace] emb_{encoder_name}: {E.shape} in {time.time() - t0:.0f}s")


# ── train ────────────────────────────────────────────────────────────────
def _batches(g, steps, E, max_cands, rng, device, train=True):
    import torch
    B = 64
    order = list(range(len(steps)))
    if train:
        rng.shuffle(order)
    for s in range(0, len(order), B):
        chunk = [steps[i] for i in order[s:s + B]]
        cand_lists, gold_sets, dist_maps = [], [], []
        for st in chunk:
            nb = g.out(st["node"]).tolist()
            gold = set(st["gold"])
            dist_maps.append(dict(zip(nb, st["cand_dist"])) if "cand_dist" in st else {})
            if len(nb) > max_cands:
                neg = [v for v in nb if v not in gold]
                nb = list(gold) + rng.sample(neg, max_cands - len(gold)) if len(gold) < max_cands else list(gold)[:max_cands]
            cand_lists.append(nb)
            gold_sets.append(gold)
        C = max(len(c) for c in cand_lists)
        idx = torch.zeros((len(chunk), C), dtype=torch.long)
        cm = torch.zeros((len(chunk), C), dtype=torch.bool)
        gm = torch.zeros((len(chunk), C), dtype=torch.bool)
        dm = torch.full((len(chunk), C), -1.0)
        for r, (cl, gs, dmap) in enumerate(zip(cand_lists, gold_sets, dist_maps)):
            idx[r, :len(cl)] = torch.tensor(cl)
            cm[r, :len(cl)] = True
            gm[r, :len(cl)] = torch.tensor([v in gs for v in cl])
            if dmap:
                dm[r, :len(cl)] = torch.tensor([float(dmap.get(v, -1)) for v in cl])
        t_idx = torch.tensor([st["target"] for st in chunk])
        n_idx = torch.tensor([st["node"] for st in chunk])
        yield (E[t_idx.to(device)], E[n_idx.to(device)], E[idx.to(device)],
               gm.to(device), cm.to(device), cand_lists, gold_sets, dm.to(device))


def train(d: Path, encoder_name: str, out_dir: Path, epochs: int, lr: float, seed: int, td: Path | None = None,
          max_cands: int = 256, hidden: int = 128, dropout: float = 0.3, weight_decay: float = 0.05,
          patience: int = 2, objective: str = "nll"):
    """
    objective: nll   multi-positive NLL on the gold set (a link exactly one step closer)
               rank  listwise soft targets over EVERY candidate from its BFS distance
                     (q ~ exp(-(d - d_min) / 0.5)), so "two steps away" beats "lost"; the
                     candidate fix for heads that win single steps but lose whole races
    """
    import torch
    from .heads import PointerHead, fit_temperature, multi_positive_nll
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = Graph(d)
    E = torch.from_numpy(np.load(d / f"emb_{encoder_name}.npy").astype(np.float32)).to(device)
    td = Path(td or d)
    tr = [s for s in read_jsonl(td / "steps_train.jsonl") if s["gold"]]
    va = [s for s in read_jsonl(td / "steps_val.jsonl") if s["gold"]]
    if objective == "rank" and "cand_dist" not in tr[0]:
        raise SystemExit("objective=rank needs tasks with cand_dist (format 2): re-run `wikirace tasks`")
    head = PointerHead(E.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, bad, log = -1.0, 0, []
    for ep in range(epochs):
        head.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for e_t, e_n, e_c, gm, cm, _, _, dm in _batches(g, tr, E, max_cands, rng, device):
            logits = head(e_t, e_n, e_c).masked_fill(~cm, -1e4)
            loss = multi_positive_nll(logits, gm, cm) if objective == "nll" else rank_loss(logits, dm, cm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        acc = step_accuracy(head, g, va, E, device)
        log.append({"epoch": ep, "train_nll": tot / max(nb, 1), "val_top1": acc, "s": round(time.time() - t0, 1)})
        print(f"  epoch {ep}: train_nll={tot / max(nb, 1):.4f} val_top1={acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc <= best:
            bad += 1
            if bad > patience:
                print(f"  early stop (val top1 not improving for {bad} epochs)")
                break
            continue
        best, bad = acc, 0
        torch.save({"head": head.state_dict(), "dim": E.shape[1], "hidden": hidden,
                    "encoder": encoder_name, "seed": seed, "objective": objective}, out_dir / "head.pt")
    ck = torch.load(out_dir / "head.pt", map_location=device)
    head.load_state_dict(ck["head"])
    head.eval()
    lg, gl = _collect_logits(head, g, va, E, device)
    T = fit_temperature(lg, gl)
    head.log_temp.data = torch.tensor(float(np.log(T)))
    ck["head"] = head.state_dict()
    torch.save(ck, out_dir / "head.pt")
    write_json(out_dir / "train_log.json", {"epochs": log, "best_val_top1": best, "temperature": T,
                                            "encoder": encoder_name, "seed": seed, "objective": objective})
    print(f"[wikirace] trained head:{encoder_name} seed={seed} best val top1={best:.4f} T={T:.3f} -> {out_dir}")


def rank_loss(logits, dist, cand_mask, tau: float = 0.5):
    """Listwise CE against soft targets from BFS distance (unreachable = worst reachable + 2)."""
    import torch
    reach = (dist >= 0) & cand_mask
    worst = torch.where(reach, dist, torch.full_like(dist, -1)).max(-1, keepdim=True).values
    d = torch.where(reach, dist, worst + 2)
    dmin = torch.where(cand_mask, d, torch.full_like(d, 1e9)).min(-1, keepdim=True).values
    q = torch.exp(-(d - dmin) / tau).masked_fill(~cand_mask, 0)
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
    logp = torch.log_softmax(logits.masked_fill(~cand_mask, -1e4), -1)
    return -(q * logp).sum(-1).mean()


def _collect_logits(head, g, steps, E, device):
    import torch
    lg, gl = [], []
    with torch.no_grad():
        for e_t, e_n, e_c, gm, cm, cls, _, _ in _batches(g, steps, E, 10**9, random.Random(0), device, train=False):
            logits = head(e_t, e_n, e_c)
            for r in range(len(cls)):
                n = len(cls[r])
                lg.append(logits[r, :n].float().cpu())
                gl.append(gm[r, :n].cpu())
    return lg, gl


def step_accuracy(head, g, steps, E, device) -> float:
    head.eval()
    lg, gl = _collect_logits(head, g, steps, E, device)
    return sum(bool(gm[int(l.argmax())]) for l, gm in zip(lg, gl)) / max(len(lg), 1)


# ── policies for rollouts ────────────────────────────────────────────────
class RandomWR:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def pick(self, g, target, node, cands):
        return self.rng.randrange(len(cands)), 1.0 / len(cands)


class EmbedWR:
    """Cosine to the target (lexical when E = hash embeddings); no training."""
    def __init__(self, E, name="lexical"):
        self.E, self.name = E.astype(np.float32), name

    def pick(self, g, target, node, cands):
        s = self.E[cands] @ self.E[target]
        e = np.exp((s - s.max()) * 10)
        k = int(s.argmax())
        return k, float(e[k] / e.sum())


class HeadWR:
    def __init__(self, head_path, E, device="cpu"):
        import torch
        from .heads import PointerHead
        self.torch = torch
        ck = torch.load(head_path, map_location=device)
        self.head = PointerHead(ck["dim"], hidden=ck["hidden"]).to(device)
        self.head.load_state_dict(ck["head"])
        self.head.eval()
        self.E = torch.from_numpy(E.astype(np.float32)).to(device)
        obj = ck.get("objective", "nll")
        self.name = f"head:{ck['encoder']}" + ("" if obj == "nll" else f"-{obj}")
        self.device = device

    def pick(self, g, target, node, cands):
        torch = self.torch
        with torch.no_grad():
            c = self.E[torch.as_tensor(cands, device=self.device)].unsqueeze(0)
            logits = self.head(self.E[target].unsqueeze(0), self.E[node].unsqueeze(0), c)[0]
            p = torch.softmax(logits / self.head.temperature, -1)
        k = int(p.argmax())
        return k, float(p[k])


class GemmaWR:
    name = "gemma"

    def __init__(self, model_path, E_rank=None):
        from .policies import GemmaPolicy
        self.llm = GemmaPolicy(model_path, max_new=8)
        self.E = E_rank.astype(np.float32) if E_rank is not None else None

    def pick(self, g, target, node, cands):
        order = list(range(len(cands)))
        if len(cands) > MAX_TEACHER_LINKS and self.E is not None:  # pre-select like Jev does
            s = self.E[cands] @ self.E[target]
            order = list(np.argsort(-s)[:MAX_TEACHER_LINKS])
        links = "\n".join(f"[{i}] {g.titles[cands[j]]}" for i, j in enumerate(order))
        system = ("You are playing Wikiracing: reach the TARGET article by clicking links. "
                  "Reply with ONLY the number of the link that gets you closest to the target.")
        user = (f"TARGET: {g.titles[target]}: {g.leads[target][:200]}\n"
                f"CURRENT ARTICLE: {g.titles[node]}\nLINKS:\n{links}\nNumber:")
        m = re.search(r"\d+", self.llm._complete(system, user))
        i = int(m.group(0)) if m else 0
        i = i if 0 <= i < len(order) else 0
        return int(order[i]), 0.5


def rollout(g, policy, start, target, max_steps):
    u, visited, ms = start, {start}, []
    for step in range(max_steps):
        cands = [int(v) for v in g.out(u) if int(v) not in visited]
        if not cands:
            return False, step, ms
        t0 = time.perf_counter()
        k, _ = policy.pick(g, target, u, cands)
        ms.append((time.perf_counter() - t0) * 1000)
        u = cands[k]
        visited.add(u)
        if u == target:
            return True, step + 1, ms
    return False, max_steps, ms


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired outcomes (same races): b01 = only B wins, b10 = only A wins."""
    import math
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b10 + b01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b10, b01) + 1)) / 2 ** n)
    return {"a_only": b10, "b_only": b01, "p": p}


def discover_heads(results: Path, seed: int, tag: str) -> dict[str, Path]:
    """results/planck3/g1_head_<key>_s<seed><tag>/head.pt -> {key: path}, key = encoder[-objective]."""
    out = {}
    for p in sorted(results.glob(f"g1_head_*_s{seed}{tag}/head.pt")):
        key = p.parent.name[len("g1_head_"):-len(f"_s{seed}{tag}")]
        if key:
            out[key] = p
    return out


def evaluate(d: Path, policies: list[str], out_dir: Path, limit: int, gemma_path: str,
             teacher_limit: int, heads: dict[str, Path], latency_ckpt: str | None, latency_tok: str | None,
             td: Path | None = None, teacher_dir: Path | None = None, primary: str = "head:planck"):
    from .metrics import ece
    g = Graph(d)
    td = Path(td or d)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(td / "pairs_test.jsonl")[: limit or None]
    test_steps = read_jsonl(td / "steps_test.jsonl")
    embs = {p.stem.replace("emb_", ""): np.load(p) for p in d.glob("emb_*.npy")}
    if "heads" in policies:  # expand to every trained head for this seed/tag
        i = policies.index("heads")
        policies = policies[:i] + [f"head:{k}" for k in heads] + policies[i + 1:]
    results, outcomes = {}, {}
    for name in policies:
        cached = None
        if name == "random":
            pol, n_pairs = RandomWR(), pairs
        elif name == "lexical":
            pol, n_pairs = EmbedWR(embs["hash"], "lexical"), pairs
        elif name.startswith("head:"):
            key = name.split(":", 1)[1]
            enc = key.split("-")[0]
            if key not in heads or enc not in embs:
                print(f"[wikirace] skip {name}: missing head or emb_{enc}.npy")
                continue
            pol, n_pairs = HeadWR(heads[key], embs[enc]), pairs
        elif name == "gemma":
            n_pairs = pairs[:teacher_limit]
            cache = (teacher_dir or out_dir) / "pairs_gemma.jsonl"
            if cache.exists():  # the teacher is deterministic and the test races are fixed: run it once
                rows = read_jsonl(cache)
                if len(rows) >= len(n_pairs) and all(r["start"] == p["start"] and r["target"] == p["target"]
                                                     for r, p in zip(rows, n_pairs)):
                    cached = rows[: len(n_pairs)]
                    print(f"  gemma: reusing {len(cached)} cached teacher races from {cache}")
            pol = None if cached else GemmaWR(gemma_path, embs.get("hash"))
        else:
            raise ValueError(name)
        rows, ms, t0 = [], [], time.time()
        if cached is not None:
            rows = cached
        else:
            for i, pr in enumerate(n_pairs):
                ok, n, m = rollout(g, pol, pr["start"], pr["target"], max_steps=2 * pr["dist"])
                ms += m
                rows.append({"i": i, "start": pr["start"], "target": pr["target"], "dist": pr["dist"],
                             "ok": bool(ok), "steps": n, "ms": round(float(np.mean(m)), 3) if m else None})
                if name == "gemma" and (i + 1) % 25 == 0:
                    print(f"    gemma {i + 1}/{len(n_pairs)} success so far {sum(r['ok'] for r in rows) / (i + 1):.3f}", flush=True)
            if name == "gemma":
                (teacher_dir or out_dir).mkdir(parents=True, exist_ok=True)
                with open((teacher_dir or out_dir) / "pairs_gemma.jsonl", "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(out_dir / f"pairs_{name.replace(':', '_')}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        outcomes[name] = [r["ok"] for r in rows]
        succ = [r for r in rows if r["ok"]]
        ms_all = ms or [r["ms"] for r in rows if r.get("ms") is not None]
        r = {"n_pairs": len(rows), "rollout_success": len(succ) / max(len(rows), 1),
             "mean_steps_over_optimal": float(np.mean([x["steps"] / x["dist"] for x in succ])) if succ else None,
             "median_decision_ms": float(np.median(ms_all)) if ms_all else None, "wall_s": round(time.time() - t0, 1),
             "cached": cached is not None}
        if name != "gemma":  # step-level top-1 + calibration on held-out steps
            top1, probs, corr = 0, [], []
            for st in test_steps[: limit * 4 if limit else None]:
                cands = [int(v) for v in g.out(st["node"])]
                if not cands:
                    continue
                k, p = pol.pick(g, st["target"], st["node"], cands)
                ok = cands[k] in set(st["gold"])
                top1 += ok
                probs.append(p)
                corr.append(ok)
            r["step_top1"] = top1 / max(len(corr), 1)
            r["step_ece"] = ece(probs, corr)
        results[name] = r
        print(f"  {name:<18} success={r['rollout_success']:.3f} steps/opt={r['mean_steps_over_optimal']} "
              f"top1={r.get('step_top1')} ece={r.get('step_ece')} ms={r['median_decision_ms']}", flush=True)
    paired = paired_stats(outcomes)
    if latency_ckpt:
        results["latency_cpu_cold"] = cold_latency(g, pairs, latency_ckpt, latency_tok)
    verdict = g1_verdict(results, paired, primary=primary)
    verdict["primary"] = primary
    summary = {"results": results, "paired": paired, "gate": verdict, "graph": read_json(d / "graph_info.json"),
               "tasks": read_json(td / "tasks_info.json")}
    write_json(out_dir / "summary.json", summary)
    print(f"\n  GATE G1: {verdict['verdict']}\n  {verdict.get('note', '')}\n  -> {out_dir / 'summary.json'}")


def paired_stats(outcomes: dict[str, list[bool]]) -> dict:
    """Every head vs the teacher on the SAME races (the teacher's subset), plus head-vs-control pairs."""
    out = {}
    teacher = outcomes.get("gemma")
    for name, oc in outcomes.items():
        if not name.startswith("head:"):
            continue
        if teacher:
            n = len(teacher)
            sub = oc[:n]
            t_rate = sum(teacher) / n
            out[f"{name} vs gemma"] = {"n": n, "head": sum(sub) / n, "teacher": t_rate,
                                       "ratio": (sum(sub) / n) / t_rate if t_rate else None,
                                       **mcnemar(sub, teacher)}
        for ctrl in ("head:hash", "lexical"):
            if ctrl in outcomes and ctrl != name:
                out[f"{name} vs {ctrl}"] = {"n": len(oc), "head": sum(oc) / len(oc),
                                            "control": sum(outcomes[ctrl]) / len(oc), **mcnemar(oc, outcomes[ctrl])}
    for a, b in (("head:planck-rank", "head:planck"), ("head:hertz", "head:planck"), ("head:hertz-rank", "head:planck-rank")):
        if a in outcomes and b in outcomes:
            out[f"{a} vs {b}"] = {"n": len(outcomes[a]), "head": sum(outcomes[a]) / len(outcomes[a]),
                                  "control": sum(outcomes[b]) / len(outcomes[b]), **mcnemar(outcomes[a], outcomes[b])}
    return out


def cold_latency(g, pairs, ckpt, tok, n=30):
    """On-device cost when nothing is cached: encode target+node+all candidate TITLES on CPU."""
    from .encoders import PlanckEncoder
    enc = PlanckEncoder(ckpt, tok, device="cpu")
    ms = []
    for pr in pairs[:n]:
        cands = [int(v) for v in g.out(pr["start"])]
        texts = [g.text(pr["target"]), g.titles[pr["start"]]] + [g.titles[c] for c in cands]
        t0 = time.perf_counter()
        enc.encode(texts, batch_size=512)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)),
            "mean_candidates": float(np.mean([len(g.out(p["start"])) for p in pairs[:n]]))}


def g1_verdict(res: dict, paired: dict | None = None, primary: str = "head:planck") -> dict:
    """
    Pre-registered: primary head >= 0.8x the teacher's rollout success ON THE SAME RACES,
    beats head:hash, < 100 ms warm. Secondary arms (-rank, hertz) are reported with the same
    rule, flagged exploratory (two arms -> read their p-values with that in mind).
    """
    paired = paired or {}
    head = res.get(primary)
    if head is None:
        return {"verdict": f"INCOMPLETE (no {primary} result)"}
    lat = head["median_decision_ms"]
    fast = lat is not None and lat < G1_MS
    notes, arms = [], {}
    hashr = res.get("head:hash")
    if hashr:
        delta = head["rollout_success"] - hashr["rollout_success"]
        pv = paired.get(f"{primary} vs head:hash", {}).get("p")
        notes.append(f"{primary}-vs-hash {delta:+.3f}" + (f" (McNemar p={pv:.3g})" if pv is not None else ""))
    if "gemma" not in res:
        return {"verdict": "INCOMPLETE (run the gemma teacher)", "note": "; ".join(notes)}
    for name in [k for k in res if k.startswith("head:")]:
        pr = paired.get(f"{name} vs gemma")
        if pr and pr["ratio"] is not None:
            arms[name] = {"paired_ratio": pr["ratio"], "n": pr["n"], "p_vs_teacher": pr["p"],
                          "pass": pr["ratio"] >= G1_REL and res[name]["median_decision_ms"] < G1_MS}
    pr = paired.get(f"{primary} vs gemma")
    rel = pr["ratio"] if pr else head["rollout_success"] / max(res["gemma"]["rollout_success"], 1e-9)
    beats_hash = hashr is None or head["rollout_success"] > hashr["rollout_success"]
    ok = rel >= G1_REL and fast and beats_hash
    notes.append(f"paired head/teacher = {rel:.2f} on {pr['n'] if pr else '?'} shared races (pass >= {G1_REL}); "
                 f"warm decision {lat:.1f} ms (pass < {G1_MS})")
    extra = [f"{k} {v['paired_ratio']:.2f}" for k, v in arms.items() if k != primary]
    if extra:
        notes.append("secondary arms (exploratory): " + ", ".join(extra))
    v = "PASS" if ok else ("FAIL (latency)" if not fast else "FAIL")
    return {"verdict": v, "relative_to_teacher": rel, "arms": arms, "note": "; ".join(notes)}


def aggregate(results_dir: Path, seeds: list[int], tag: str, out_dir: Path, primary: str = "head:planck"):
    """Across seeds: mean, sd, 95% t-interval per policy; verdict on the mean paired ratio."""
    import statistics
    T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    runs = {}
    for sd in seeds:
        p = results_dir / f"g1_eval_s{sd}{tag}" / "summary.json"
        if p.exists():
            runs[sd] = read_json(p)
    if not runs:
        raise SystemExit("no g1_eval summaries found for those seeds")
    pols = sorted({k for r in runs.values() for k in r["results"] if "rollout_success" in r["results"][k]})
    table = {}
    for pol in pols:
        xs = [r["results"][pol]["rollout_success"] for r in runs.values() if pol in r["results"]]
        m = statistics.mean(xs)
        s_ = statistics.stdev(xs) if len(xs) > 1 else 0.0
        h = T95.get(len(xs), 2.0) * s_ / len(xs) ** 0.5 if len(xs) > 1 else None
        table[pol] = {"n_seeds": len(xs), "mean": m, "sd": s_, "ci95": [m - h, m + h] if h is not None else None, "per_seed": xs}
    ratios = {}
    for arm in [p for p in pols if p.startswith("head:")]:
        rs = [r["paired"][f"{arm} vs gemma"]["ratio"] for r in runs.values()
              if f"{arm} vs gemma" in r.get("paired", {}) and r["paired"][f"{arm} vs gemma"]["ratio"] is not None]
        if rs:
            ratios[arm] = {"mean": statistics.mean(rs), "per_seed": rs}
    beats_hash_every_seed = all(r["results"].get(primary, {}).get("rollout_success", 0) >
                                r["results"].get("head:hash", {}).get("rollout_success", 1) for r in runs.values())
    prim = ratios.get(primary, {}).get("mean")
    ok = prim is not None and prim >= G1_REL and beats_hash_every_seed
    verdict = {"verdict": "PASS" if ok else ("INCOMPLETE" if prim is None else "FAIL"), "primary": primary,
               "note": f"mean paired {primary}/teacher = {prim:.2f} over {len(runs)} seeds (pass >= {G1_REL}); "
                       f"beats head:hash in every seed: {beats_hash_every_seed}" if prim is not None else "no paired teacher ratio"}
    summ = {"seeds": sorted(runs), "policies": table, "paired_ratio_vs_teacher": ratios, "gate": verdict}
    write_json(out_dir / "summary.json", summ)
    print(f"[wikirace] aggregate over seeds {sorted(runs)}:")
    for pol, t in table.items():
        ci = f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]" if t["ci95"] else ""
        print(f"  {pol:<18} {t['mean']:.3f} ± {t['sd']:.3f} {ci}")
    print(f"  GATE G1 (seeds): {verdict['verdict']}  {verdict['note']}")


def add_cli(sub):
    p = sub.add_parser("wikirace", help="G1 offline Wikiracing pipeline")
    p.add_argument("stage", choices=["build", "tasks", "embed", "train", "eval", "aggregate"])
    p.add_argument("--dir", default=str(WR_DIR), help="graph + embeddings (shared by every task set)")
    p.add_argument("--tag", default="", help="task-set tag, e.g. _quick: tasks in <dir>/tasks<tag>, results suffixed")
    p.add_argument("--lang", default="simple")
    p.add_argument("--dump", default=None)
    p.add_argument("--max-pages", type=int, default=0, help="debug: stop parsing after N articles")
    p.add_argument("--targets", type=int, default=3000)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--offpath", type=int, default=6)
    p.add_argument("--encoder", default="hash", choices=["hash", "planck", "hertz"])
    p.add_argument("--objective", default="nll", choices=["nll", "rank"])
    p.add_argument("--checkpoint", default="checkpoints/planck13/best.pt")
    p.add_argument("--tokenizer", default="data/wikipedia/tokenizer.model")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="0,1,2", help="aggregate: comma list")
    p.add_argument("--policies", default="random,lexical,heads,gemma", help="'heads' = every trained head")
    p.add_argument("--limit", type=int, default=0, help="eval: cap test pairs")
    p.add_argument("--teacher-limit", type=int, default=300)
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")
    p.add_argument("--no-latency", action="store_true")
    p.add_argument("--primary", default="head:planck", help="the pre-registered primary arm for the verdict")
    p.set_defaults(fn=_cli)


def tasks_dir(d: Path, tag: str) -> Path:
    return d if not tag else d / f"tasks{tag}"


def _cli(args):
    d = Path(args.dir)
    td = tasks_dir(d, args.tag)
    results = REPO_ROOT / "results" / "planck3"
    if args.stage == "build":
        dump = Path(args.dump) if args.dump else d / f"{args.lang}wiki-latest-pages-articles.xml.bz2"
        if not args.dump:
            download(DUMP_URL.format(lang=args.lang), dump)
        build_graph(dump, d, args.max_pages)
    elif args.stage == "tasks":
        make_tasks(d, args.targets, args.starts, args.offpath, args.seed, td=td)
    elif args.stage == "embed":
        embed(d, args.encoder, checkpoint=args.checkpoint, tokenizer=args.tokenizer)
    elif args.stage == "train":
        key = args.encoder + ("" if args.objective == "nll" else f"-{args.objective}")
        train(d, args.encoder, results / f"g1_head_{key}_s{args.seed}{args.tag}", args.epochs, args.lr,
              args.seed, td=td, objective=args.objective)
    elif args.stage == "eval":
        heads = discover_heads(results, args.seed, args.tag)
        lat = None if args.no_latency or not Path(args.checkpoint).exists() else args.checkpoint
        evaluate(d, args.policies.split(","), results / f"g1_eval_s{args.seed}{args.tag}", args.limit,
                 args.gemma_path, args.teacher_limit, heads, lat, args.tokenizer, td=td,
                 teacher_dir=results / f"g1_teacher{args.tag}", primary=args.primary)
    elif args.stage == "aggregate":
        aggregate(results, [int(x) for x in args.seeds.split(",")], args.tag, results / f"g1_aggregate{args.tag}",
                  primary=args.primary)
