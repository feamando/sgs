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
               seed: int, min_dist: int = 2, max_dist: int = 6, min_indeg: int = 5):
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
                    steps[split].append({"target": tgt, "node": int(u), "dist": int(dv[u]), "gold": gold})
                    u = rng.choice(gold)
            finite = np.flatnonzero(np.isfinite(dv) & (dv >= 1) & (dv <= 8))
            for j in range(min(offpath_per_target, len(finite))):
                u = int(finite[rng.randrange(len(finite))])
                nb = g.out(u)
                if len(nb) == 0:
                    continue
                gold = [int(v) for v in nb if dv[v] == dv[u] - 1]
                steps[split].append({"target": tgt, "node": int(u), "dist": int(dv[u]), "gold": gold})
        print(f"  targets {min(bstart + B, len(targets))}/{len(targets)} ({time.time() - t0:.0f}s)", flush=True)
    for s in pairs:
        with open(d / f"pairs_{s}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in pairs[s])
        with open(d / f"steps_{s}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in steps[s])
    info = {s: {"pairs": len(pairs[s]), "steps": len(steps[s])} for s in pairs}
    info.update({"n_targets": len(targets), "seed": seed})
    write_json(d / "tasks_info.json", info)
    print(f"[wikirace] tasks: {info}")


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
        cand_lists, gold_sets = [], []
        for st in chunk:
            nb = g.out(st["node"]).tolist()
            gold = set(st["gold"])
            if len(nb) > max_cands:
                neg = [v for v in nb if v not in gold]
                nb = list(gold) + rng.sample(neg, max_cands - len(gold)) if len(gold) < max_cands else list(gold)[:max_cands]
            cand_lists.append(nb)
            gold_sets.append(gold)
        C = max(len(c) for c in cand_lists)
        idx = torch.zeros((len(chunk), C), dtype=torch.long)
        cm = torch.zeros((len(chunk), C), dtype=torch.bool)
        gm = torch.zeros((len(chunk), C), dtype=torch.bool)
        for r, (cl, gs) in enumerate(zip(cand_lists, gold_sets)):
            idx[r, :len(cl)] = torch.tensor(cl)
            cm[r, :len(cl)] = True
            gm[r, :len(cl)] = torch.tensor([v in gs for v in cl])
        t_idx = torch.tensor([st["target"] for st in chunk])
        n_idx = torch.tensor([st["node"] for st in chunk])
        yield (E[t_idx.to(device)], E[n_idx.to(device)], E[idx.to(device)],
               gm.to(device), cm.to(device), cand_lists, gold_sets)


def train(d: Path, encoder_name: str, out_dir: Path, epochs: int, lr: float, seed: int,
          max_cands: int = 256, hidden: int = 128, dropout: float = 0.3, weight_decay: float = 0.05,
          patience: int = 2):
    import torch
    from .heads import PointerHead, fit_temperature, multi_positive_nll
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = Graph(d)
    E = torch.from_numpy(np.load(d / f"emb_{encoder_name}.npy").astype(np.float32)).to(device)
    tr = [s for s in read_jsonl(d / "steps_train.jsonl") if s["gold"]]
    va = [s for s in read_jsonl(d / "steps_val.jsonl") if s["gold"]]
    head = PointerHead(E.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, bad, log = -1.0, 0, []
    for ep in range(epochs):
        head.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for e_t, e_n, e_c, gm, cm, _, _ in _batches(g, tr, E, max_cands, rng, device):
            loss = multi_positive_nll(head(e_t, e_n, e_c).masked_fill(~cm, -1e4), gm, cm)
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
                    "encoder": encoder_name, "seed": seed}, out_dir / "head.pt")
    ck = torch.load(out_dir / "head.pt", map_location=device)
    head.load_state_dict(ck["head"])
    head.eval()
    lg, gl = _collect_logits(head, g, va, E, device)
    T = fit_temperature(lg, gl)
    head.log_temp.data = torch.tensor(float(np.log(T)))
    ck["head"] = head.state_dict()
    torch.save(ck, out_dir / "head.pt")
    write_json(out_dir / "train_log.json", {"epochs": log, "best_val_top1": best, "temperature": T,
                                            "encoder": encoder_name, "seed": seed})
    print(f"[wikirace] trained head:{encoder_name} seed={seed} best val top1={best:.4f} T={T:.3f} -> {out_dir}")


def _collect_logits(head, g, steps, E, device):
    import torch
    lg, gl = [], []
    with torch.no_grad():
        for e_t, e_n, e_c, gm, cm, cls, _ in _batches(g, steps, E, 10**9, random.Random(0), device, train=False):
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
        self.name = f"head:{ck['encoder']}"
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


def evaluate(d: Path, policies: list[str], out_dir: Path, limit: int, gemma_path: str,
             teacher_limit: int, heads: dict[str, Path], latency_ckpt: str | None, latency_tok: str | None):
    from .metrics import ece
    g = Graph(d)
    pairs = read_jsonl(d / "pairs_test.jsonl")[: limit or None]
    test_steps = read_jsonl(d / "steps_test.jsonl")
    embs = {p.stem.replace("emb_", ""): np.load(p) for p in d.glob("emb_*.npy")}
    results = {}
    for name in policies:
        if name == "random":
            pol, n_pairs = RandomWR(), pairs
        elif name == "lexical":
            pol, n_pairs = EmbedWR(embs["hash"], "lexical"), pairs
        elif name.startswith("head:"):
            enc = name.split(":", 1)[1]
            if enc not in heads or enc not in embs:
                print(f"[wikirace] skip {name}: missing head or emb_{enc}.npy")
                continue
            pol, n_pairs = HeadWR(heads[enc], embs[enc]), pairs
        elif name == "gemma":
            pol, n_pairs = GemmaWR(gemma_path, embs.get("hash")), pairs[:teacher_limit]
        else:
            raise ValueError(name)
        succ, steps_ratio, ms = 0, [], []
        t0 = time.time()
        for i, pr in enumerate(n_pairs):
            ok, n, m = rollout(g, pol, pr["start"], pr["target"], max_steps=2 * pr["dist"])
            succ += ok
            ms += m
            if ok:
                steps_ratio.append(n / pr["dist"])
            if name == "gemma" and (i + 1) % 10 == 0:
                print(f"    gemma {i + 1}/{len(n_pairs)} success so far {succ / (i + 1):.3f}", flush=True)
        r = {"n_pairs": len(n_pairs), "rollout_success": succ / max(len(n_pairs), 1),
             "mean_steps_over_optimal": float(np.mean(steps_ratio)) if steps_ratio else None,
             "median_decision_ms": float(np.median(ms)) if ms else None, "wall_s": round(time.time() - t0, 1)}
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
        print(f"  {name:<14} success={r['rollout_success']:.3f} steps/opt={r['mean_steps_over_optimal']} "
              f"top1={r.get('step_top1')} ece={r.get('step_ece')} ms={r['median_decision_ms']}", flush=True)
    if latency_ckpt:
        results["latency_cpu_cold"] = cold_latency(g, pairs, latency_ckpt, latency_tok)
    verdict = g1_verdict(results)
    summary = {"results": results, "gate": verdict, "graph": read_json(d / "graph_info.json"),
               "tasks": read_json(d / "tasks_info.json")}
    write_json(out_dir / "summary.json", summary)
    print(f"\n  GATE G1: {verdict['verdict']}\n  {verdict.get('note', '')}\n  -> {out_dir / 'summary.json'}")


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


def g1_verdict(res: dict) -> dict:
    head = res.get("head:planck")
    if head is None:
        return {"verdict": "INCOMPLETE (no head:planck result)"}
    teacher = res.get("gemma")
    lat = head["median_decision_ms"]
    fast = lat is not None and lat < G1_MS
    notes = []
    hashr = res.get("head:hash")
    if hashr:
        delta = head["rollout_success"] - hashr["rollout_success"]
        notes.append(f"planck-vs-hash head delta {delta:+.3f} (<=0 means Planck features add nothing)")
    if teacher is None:
        return {"verdict": "INCOMPLETE (run the gemma teacher)", "note": "; ".join(notes)}
    rel = head["rollout_success"] / max(teacher["rollout_success"], 1e-9)
    ok = rel >= G1_REL and fast
    notes.append(f"head/teacher = {rel:.2f} (pass >= {G1_REL}); warm decision {lat:.1f} ms (pass < {G1_MS})")
    return {"verdict": "PASS" if ok else ("FAIL (try the Hertz encoder)" if not ok and fast else "FAIL (latency)"),
            "relative_to_teacher": rel, "note": "; ".join(notes)}


# ── CLI ──────────────────────────────────────────────────────────────────
def add_cli(sub):
    p = sub.add_parser("wikirace", help="G1 offline Wikiracing pipeline")
    p.add_argument("stage", choices=["build", "tasks", "embed", "train", "eval"])
    p.add_argument("--dir", default=str(WR_DIR))
    p.add_argument("--lang", default="simple")
    p.add_argument("--dump", default=None)
    p.add_argument("--max-pages", type=int, default=0, help="debug: stop parsing after N articles")
    p.add_argument("--targets", type=int, default=3000)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--offpath", type=int, default=6)
    p.add_argument("--encoder", default="hash", choices=["hash", "planck"])
    p.add_argument("--checkpoint", default="checkpoints/planck13/best.pt")
    p.add_argument("--tokenizer", default="data/wikipedia/tokenizer.model")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policies", default="random,lexical,head:hash,head:planck,gemma")
    p.add_argument("--limit", type=int, default=0, help="eval: cap test pairs")
    p.add_argument("--teacher-limit", type=int, default=100)
    p.add_argument("--gemma-path", default="models/gemma-4-e4b-it")
    p.add_argument("--no-latency", action="store_true")
    p.set_defaults(fn=_cli)


def _cli(args):
    d = Path(args.dir)
    results = REPO_ROOT / "results" / "planck3"
    if args.stage == "build":
        dump = Path(args.dump) if args.dump else d / f"{args.lang}wiki-latest-pages-articles.xml.bz2"
        if not args.dump:
            download(DUMP_URL.format(lang=args.lang), dump)
        build_graph(dump, d, args.max_pages)
    elif args.stage == "tasks":
        make_tasks(d, args.targets, args.starts, args.offpath, args.seed)
    elif args.stage == "embed":
        embed(d, args.encoder, checkpoint=args.checkpoint, tokenizer=args.tokenizer)
    elif args.stage == "train":
        train(d, args.encoder, results / f"g1_head_{args.encoder}_s{args.seed}", args.epochs, args.lr, args.seed)
    elif args.stage == "eval":
        heads = {e: results / f"g1_head_{e}_s{args.seed}" / "head.pt" for e in ("hash", "planck")}
        heads = {e: p for e, p in heads.items() if p.exists()}
        lat = None if args.no_latency or not Path(args.checkpoint).exists() else args.checkpoint
        evaluate(d, args.policies.split(","), results / f"g1_eval_s{args.seed}", args.limit,
                 args.gemma_path, args.teacher_limit, heads, lat, args.tokenizer)
