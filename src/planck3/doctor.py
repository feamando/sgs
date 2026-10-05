"""
Preflight for the 4090 box: catch what would kill a long run in minute 90, in minute 1.

    FAIL  blocks every stage (python too old, core package missing, no network)
    WARN  degrades a stage (no Gemma -> no teacher; no Planck ckpt -> no head:planck; no Docker)
    OK

--deep also loads the models and MEASURES them: Gemma emits valid typed decisions
(the G0 run is worthless if it can't), and Planck embedding throughput; both feed a
per-stage ETA so you know whether `all` is a coffee or an afternoon.
"""

import importlib
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .util import REPO_ROOT, write_json

OK, WARN, FAIL = "OK", "WARN", "FAIL"
N_ARTICLES = 284_749          # Simple English Wikipedia graph built 2026-09-29
G0_DECISIONS = 250            # ~47 tasks x ~5.2 decisions (Mac heuristic run)
G0_CLOSED_BOOK_Q = 62         # questions incl. compare cells and chat turns
G1_TEACHER_DECISIONS = 400    # 100 pairs x ~4 steps; short replies (a link number), ~prefill-bound


class Doctor:
    def __init__(self):
        self.rows, self.measured = [], {}

    def add(self, status, check, detail, fix=""):
        self.rows.append({"status": status, "check": check, "detail": detail, "fix": fix})

    # ── environment ──────────────────────────────────────────────────────
    def python_and_packages(self):
        v = sys.version_info
        self.add(OK if v >= (3, 10) else FAIL, "python", platform.python_version(),
                 "" if v >= (3, 10) else "needs Python 3.10+ (py -3.12 -m venv .venv)")
        for mod, need, fix in (("numpy", FAIL, ""), ("scipy", FAIL, ""), ("requests", FAIL, ""),
                               ("torch", FAIL, "pip install torch --index-url https://download.pytorch.org/whl/cu124"),
                               ("sentencepiece", WARN, "pip install sentencepiece (needed for head:planck)"),
                               ("trafilatura", WARN, "pip install trafilatura (fallback extractor is cruder)"),
                               ("transformers", WARN, "pip install -U transformers (needed for the Gemma teacher)")):
            try:
                m = importlib.import_module(mod)
                self.add(OK, mod, getattr(m, "__version__", "ok"))
            except Exception as e:
                msg = str(e).splitlines()[0][:140] if str(e) else ""
                hint = fix or f"pip install {mod}"
                if mod == "trafilatura" and "html" in msg.lower() and "clean" in msg.lower():
                    hint = "pip install lxml_html_clean  (lxml >= 5.2 split html.clean out)"
                self.add(need, mod, f"import failed: {e.__class__.__name__}: {msg}", hint)

    def gpu(self):
        try:
            import torch
        except Exception:
            return
        if not torch.cuda.is_available():
            self.add(WARN, "cuda", "not available (CPU only)",
                     "install the CUDA build of torch; Gemma + Planck embedding are very slow on CPU")
            return
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 2**30
        self.add(OK if gb >= 15 else WARN, "gpu", f"{p.name}, {gb:.0f} GB",
                 "" if gb >= 15 else "Gemma 4 E4B in bf16 needs ~16 GB")

    def disk(self):
        free = shutil.disk_usage(REPO_ROOT).free / 2**30
        self.add(OK if free >= 5 else (WARN if free >= 2 else FAIL), "disk free", f"{free:.0f} GB",
                 "" if free >= 5 else "G1 needs ~3 GB (dump 356 MB, graph, embeddings ~600 MB)")

    def git(self):
        try:
            br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
            subprocess.run(["git", "fetch", "-q", "origin"], cwd=REPO_ROOT, capture_output=True, timeout=30)
            behind = subprocess.run(["git", "rev-list", "--count", "HEAD..origin/main"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip()
            n = int(behind or 0)
            self.add(OK if n == 0 else WARN, "git", f"branch {br}, {n} commits behind origin/main",
                     "" if n == 0 else "git pull")
        except Exception as e:
            self.add(WARN, "git", f"could not check ({e.__class__.__name__})")

    def network(self, searxng_url):
        import requests
        ua = {"User-Agent": "Planck3-research/0.1 (+https://github.com/feamando/sgs)"}
        for name, url, need in (("wikipedia api", "https://en.wikipedia.org/w/api.php?action=query&format=json", FAIL),
                                ("wikimedia dumps", "https://dumps.wikimedia.org/simplewiki/latest/", WARN)):
            try:
                r = requests.get(url, headers=ua, timeout=10)
                self.add(OK if r.status_code == 200 else need, name, f"HTTP {r.status_code}")
            except Exception as e:
                self.add(need, name, f"unreachable ({e.__class__.__name__})", "check network / VPN / proxy")
        try:
            r = requests.get(f"{searxng_url}/search", params={"q": "test", "format": "json"}, timeout=5)
            ok = r.status_code == 200 and "results" in r.json()
            self.add(OK if ok else WARN, "searxng", f"{searxng_url} {'up' if ok else 'answered but not JSON'}",
                     "" if ok else "check config/searxng/settings.yml has json in search.formats")
        except Exception:
            docker = shutil.which("docker")
            self.add(WARN, "searxng", "not running" + ("" if docker else ", Docker not installed"),
                     ".\\scripts\\planck3.ps1 searxng" if docker else
                     "optional: install Docker Desktop; without it search uses the Wikipedia API")

    # ── models ───────────────────────────────────────────────────────────
    def planck(self, ckpt, tok, deep):
        if not Path(ckpt).exists():
            self.add(WARN, "planck checkpoint", f"{ckpt} missing", "copy checkpoints/planck13/best.pt onto the box")
            return
        if not Path(tok).exists():
            self.add(WARN, "planck tokenizer", f"{tok} missing", "copy data/wikipedia/tokenizer.model")
            return
        try:
            import sentencepiece as spm
            import torch
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = sd["model"] if "model" in sd else sd
            vocab = state["tok_mu.weight"].shape[0]
            pieces = spm.SentencePieceProcessor(model_file=str(tok)).get_piece_size()
            same = vocab == pieces
            self.add(OK if same else FAIL, "planck ckpt vs tokenizer", f"vocab {vocab} vs {pieces} pieces",
                     "" if same else "tokenizer does not belong to this checkpoint (head:planck would be garbage)")
        except Exception as e:
            self.add(FAIL, "planck checkpoint", f"load failed: {e}", "re-copy the checkpoint")
            return
        if deep:
            from .encoders import PlanckEncoder
            enc = PlanckEncoder(ckpt, tok)
            # lead-length texts (~60 tokens, like "Title. lead" in the graph), not short stubs
            lead = ("is a town in the south of the country, known for its old castle, its river port and a "
                    "university founded in the fourteenth century; it has about forty thousand inhabitants.")
            texts = [f"Example article {i}. Example article {i} {lead}" for i in range(2048)]
            enc.encode(texts[:256])  # warm-up
            t0 = time.perf_counter()
            enc.encode(texts)
            per = (time.perf_counter() - t0) / len(texts)
            self.measured["planck_embed_s_per_text"] = per
            self.add(OK, "planck embed speed", f"{1 / per:,.0f} texts/s on {enc.device} "
                     f"({_dur(per * N_ARTICLES)} for the G1 graph)")
            del enc

    def gemma(self, path, deep):
        if not Path(path).exists():
            self.add(WARN, "gemma teacher", f"{path} missing",
                     f"huggingface-cli download google/gemma-4-E4B-it --local-dir {path}")
            return
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(path)
            self.add(OK, "gemma config", path)
        except Exception as e:
            self.add(FAIL, "gemma config", f"unreadable: {e.__class__.__name__}", "pip install -U transformers")
            return
        if not deep:
            return
        from .actions import InvalidDecision, validate
        from .policies import GemmaPolicy
        pol = GemmaPolicy(path)
        valid, ms = 0, []
        for obs in _probe_observations():
            t0 = time.perf_counter()
            d = pol.decide(obs)
            ms.append((time.perf_counter() - t0) * 1000)
            try:
                validate(d, obs["phase"], {k: len(v) for k, v in obs["lists"].items()})
                valid += 1
            except InvalidDecision:
                pass
        t0 = time.perf_counter()
        cb = pol.answer_closed_book("What year was IKEA founded?")
        cb_ms = (time.perf_counter() - t0) * 1000
        per = sum(ms) / len(ms)
        self.measured["gemma_ms_per_decision"] = per
        self.measured["gemma_ms_closed_book"] = cb_ms
        n = len(_probe_observations())
        self.add(OK if valid == n else (WARN if valid else FAIL), "gemma typed decisions",
                 f"{valid}/{n} valid, {per:.0f} ms/decision",
                 "" if valid == n else "teacher emits invalid actions: inspect policies.SYSTEM_PROMPT / parse_decision")
        self.add(OK if "1943" in cb else WARN, "gemma closed-book", f"'{cb[:40]}' in {cb_ms:.0f} ms")
        del pol

    # ── ETA ──────────────────────────────────────────────────────────────
    def eta(self) -> list[tuple[str, str]]:
        m = self.measured
        g = m.get("gemma_ms_per_decision")
        e = m.get("planck_embed_s_per_text")
        rows = [("setup + smoke", "~1 min"),
                ("g0 teacher (Gemma on tools)",
                 _dur(G0_DECISIONS * g / 1000 + 47 * 3) if g else "~15-25 min (run doctor -Deep to measure)"),
                ("g0 closed-book (base-chat rival)",
                 _dur(G0_CLOSED_BOOK_Q * m["gemma_ms_closed_book"] / 1000) if g else "~2-5 min"),
                ("g0 heuristic floor", "~2 min (cached pages)"),
                ("g1 build + tasks + hash embed", "~5 min (356 MB download)"),
                ("g1 planck embed", _dur(e * N_ARTICLES) if e else "unknown (run doctor -Deep)"),
                ("g1 train heads", "~5 min on GPU"),
                # a race decision replies with one number, so it costs ~a closed-book answer, not a
                # G0 decision (first box run: 272 ms measured vs the old 3x-G0 estimate of ~6 s)
                ("g1 eval incl. Gemma teacher",
                 _dur(G1_TEACHER_DECISIONS * m["gemma_ms_closed_book"] * 1.5 / 1000 + 120) if g else "~5-15 min")]
        return rows

    def report(self) -> int:
        w = max(len(r["check"]) for r in self.rows)
        for r in self.rows:
            print(f"  [{r['status']:<4}] {r['check']:<{w}}  {r['detail']}" + (f"\n         -> {r['fix']}" if r["fix"] else ""))
        print("\n  Estimated time per stage:")
        for stage, t in self.eta():
            print(f"    {stage:<34} {t}")
        n_fail = sum(r["status"] == FAIL for r in self.rows)
        n_warn = sum(r["status"] == WARN for r in self.rows)
        print(f"\n  {'READY' if not n_fail else 'NOT READY'}: {n_fail} fail, {n_warn} warn")
        write_json(REPO_ROOT / "results" / "planck3" / "doctor.json",
                   {"rows": self.rows, "measured": self.measured, "eta": self.eta()})
        return 1 if n_fail else 0


def _dur(seconds: float) -> str:
    return f"~{seconds:.0f} s" if seconds < 90 else f"~{seconds / 60:.0f} min"


def _probe_observations() -> list[dict]:
    """Three fixed decision points: results, page spans, held value (what G0 asks the teacher)."""
    results = [{"title": "IKEA - Wikipedia", "url": "https://en.wikipedia.org/wiki/IKEA", "domain": "en.wikipedia.org",
                "snippet": "IKEA is a Swedish multinational conglomerate founded in 1943.", "trust": 0.5},
               {"title": "Cheap furniture deals", "url": "https://spam.example/ikea", "domain": "spam.example",
                "snippet": "Buy now!", "trust": 0.5}]
    spans = [{"value": "1943", "context": "IKEA was founded in 1943 by Ingvar Kamprad.", "score": 1.4, "url": results[0]["url"]},
             {"value": "2008", "context": "the world's largest furniture retailer since 2008", "score": 1.3, "url": results[0]["url"]}]
    held = dict(spans[0], domain="en.wikipedia.org")
    base = {"question": "What year was IKEA founded?", "answer_type": "year", "history": [], "store_size": 0}
    return [
        base | {"phase": "results", "allowed": ["OPEN", "ABSTAIN"], "pointer": {"OPEN": "results"},
                "lists": {"results": results, "spans": [], "store": []}, "held": None, "verified": None, "opened": set()},
        base | {"phase": "page", "allowed": ["EXTRACT", "OPEN", "ABSTAIN"], "pointer": {"EXTRACT": "spans", "OPEN": "results"},
                "lists": {"results": results, "spans": spans, "store": []}, "held": None, "verified": None, "opened": {0}},
        base | {"phase": "extracted", "allowed": ["ANSWER", "VERIFY", "OPEN", "ABSTAIN"], "pointer": {"OPEN": "results"},
                "lists": {"results": results, "spans": spans, "store": []}, "held": held, "verified": None, "opened": {0}},
    ]


def run(deep: bool, ckpt: str, tok: str, gemma: str, searxng_url: str) -> int:
    d = Doctor()
    print(f"Planck 3.0 doctor ({'deep: loads and measures the models' if deep else 'quick'})\n")
    d.python_and_packages()
    d.gpu()
    d.disk()
    d.git()
    d.network(searxng_url)
    d.planck(ckpt, tok, deep)
    d.gemma(gemma, deep)
    return d.report()
