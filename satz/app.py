"""
Satz demo v0.2: local web app with a model selector (Planck <-> Hertz).

Loads a trained SGS LM checkpoint and serves a FastAPI endpoint that generates
text. For models that have a blob store (Planck), it also shows which blobs are
retrieved for a prompt. Hertz has no blob store, so it runs blob-free and the
UI greys the blob panel.

Run (default model = planck):
    python -m satz.app --model planck

Switch the default at launch:
    python -m satz.app --model hertz

Paths default from the registry below; override per-flag if your layout differs.
Then open http://localhost:8001 in a browser.
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.sgs_lm import SGSLanguageModel, migrate_state_dict
from src.blob_store import BlobStore


# ── Model registry ─────────────────────────────────────────────────────────
# Arch MUST match the checkpoint or load_state_dict fails on shape mismatch.
# blobs_dir=None means the model runs blob-free (blob panel greyed in the UI).
MODELS = {
    "planck": {
        "label": "Planck 1.3 (~100M, blobs)",
        "checkpoint": "checkpoints/planck/best.pt",
        "tokenizer": "data/wikipedia/tokenizer.model",
        "blobs_dir": "data/blobs/wikipedia",
        "arch": dict(d_s=128, d_f=1000, n_passes=3, n_heads=4,
                     context_len=512, ffn_mult=4),
    },
    "hertz": {
        "label": "Hertz 1.2 (0.64B, blob-free)",
        "checkpoint": "checkpoints/hertz12/best.pt",
        "tokenizer": "data/hertz12_data/tokenizer.model",
        "blobs_dir": None,
        "arch": dict(d_s=256, d_f=3700, n_passes=3, n_heads=4,
                     context_len=512, ffn_mult=4),
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Satz demo server")
    p.add_argument("--model", default="planck", choices=list(MODELS.keys()),
                   help="Model to load at startup (default: planck)")
    p.add_argument("--k", type=int, default=8,
                   help="Top-k blobs to retrieve (default 8)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-new", type=int, default=200,
                   help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    # Per-model path overrides (optional; default from the registry).
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    p.add_argument("--tokenizer", default=None, help="Override tokenizer path")
    p.add_argument("--blobs-dir", default=None, help="Override blob directory")
    # Conversation logging (on by default; one JSONL line per generation).
    p.add_argument("--log-file", default="runs/satz_conversations.jsonl",
                   help="JSONL file to append each generation to")
    p.add_argument("--no-log", action="store_true",
                   help="Disable conversation logging")
    return p.parse_args()


class ConversationLogger:
    """Appends one JSON line per generation for later analysis.

    JSONL (one self-contained record per line) is append-only, crash-safe, and
    trivial to load with pandas / jq. Writes are lock-guarded so concurrent
    requests don't interleave. A failed write never breaks generation.
    """

    def __init__(self, log_file: str | None):
        self.path = None
        self._lock = Lock()
        if not log_file:
            print("[satz] conversation logging: OFF")
            return
        p = Path(log_file)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        print(f"[satz] conversation log: {self.path}")

    def log(self, record: dict) -> None:
        if self.path is None:
            return
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:  # logging must never break a request
            print(f"[satz] WARN: failed to write conversation log: {e}")


def _resolve(path_str: str) -> Path:
    """Resolve a path relative to cwd, then to the project root."""
    p = Path(path_str)
    if p.exists():
        return p
    alt = ROOT / path_str
    return alt if alt.exists() else p


class SatzRuntime:
    """Loads one SGS LM checkpoint (+ optional blob store) and generates text."""

    def __init__(self, name: str, spec: dict, k: int, max_new: int,
                 temperature: float):
        self.name = name
        self.label = spec["label"]
        self.arch = spec["arch"]
        self.max_new = max_new
        self.temperature = temperature
        self.k = k
        self.has_blobs = spec.get("blobs_dir") is not None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Tokenizer ──
        tok_path = _resolve(spec["tokenizer"])
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer not found: {spec['tokenizer']}")
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tok_path))
        self.vocab_size = self.sp.get_piece_size()
        print(f"[satz:{name}] tokenizer: {tok_path} ({self.vocab_size} tokens)")

        # ── Blob store (optional) ──
        self.blob_store = None
        self.n_blobs = 0
        if self.has_blobs:
            self._load_blobs(spec["blobs_dir"])

        # ── Model ──
        ckpt_path = _resolve(spec["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {spec['checkpoint']}")
        print(f"[satz:{name}] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = migrate_state_dict(state)

        ckpt_vocab_size = state["tok_mu.weight"].shape[0]
        self.model = SGSLanguageModel(
            vocab_size=ckpt_vocab_size,
            d_s=self.arch["d_s"],
            d_f=self.arch["d_f"],
            n_passes=self.arch["n_passes"],
            n_heads=self.arch["n_heads"],
            max_len=self.arch["context_len"],
            ffn_mult=self.arch["ffn_mult"],
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[satz:{name}] model: {self.model.count_parameters()/1e6:.1f}M "
              f"params on {self.device} | blobs={'yes' if self.has_blobs else 'no'}")
        print(f"[satz:{name}] ready.")

    def _load_blobs(self, blobs_dir_str: str):
        blobs_dir = _resolve(blobs_dir_str)
        blobs_pt = blobs_dir / "blobs.pt"
        if not blobs_pt.exists():
            raise FileNotFoundError(
                f"blobs.pt not found in {blobs_dir}. Run scripts/build_blobs.py first.")
        print(f"[satz:{self.name}] loading blobs from {blobs_dir} ...")
        blob_data = torch.load(blobs_pt, map_location="cpu", weights_only=False)
        self.n_blobs = blob_data["mu"].shape[0]
        print(f"[satz:{self.name}] blob store: {self.n_blobs:,} blobs, "
              f"d_s={blob_data['mu'].shape[1]}, d_f={blob_data['features'].shape[1]}")
        self.blob_store = BlobStore(
            n_blobs=self.n_blobs,
            d_s=self.arch["d_s"],
            d_f=self.arch["d_f"],
            k=self.k,
        )
        self.blob_store.init_from_clusters(
            mu=blob_data["mu"],
            log_var=blob_data["log_var"],
            alpha=blob_data["raw_alpha"],
            features=blob_data["features"],
        )
        self.blob_store.to(self.device).eval()

    def _tokenize(self, text: str) -> torch.Tensor:
        ids = self.sp.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def _compute_query(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute blob query from input tokens (mean of token mu's)."""
        with torch.no_grad():
            mu = self.model.tok_mu(token_ids)  # [B, L, d_s]
            pos = torch.arange(token_ids.shape[1], device=token_ids.device)
            mu = mu + self.model.pos_mu(pos).unsqueeze(0)
            return mu.mean(dim=1)  # [B, d_s]

    def _retrieve_blobs(self, prompt_ids: torch.Tensor, k: int) -> list:
        """Retrieve top-k blobs for the prompt. Empty list if blob-free."""
        if not self.has_blobs:
            return []
        self.blob_store.k = k
        query = self._compute_query(prompt_ids)
        top_idx, top_scores = self.blob_store.retrieve(query)
        blob_indices = top_idx[0].cpu().tolist()
        blob_scores = top_scores[0].cpu().tolist()
        max_score = max(blob_scores) if blob_scores else 1.0
        blobs_info = []
        for i, (idx, score) in enumerate(zip(blob_indices, blob_scores)):
            feat_norm = float(self.blob_store.features[idx].norm().item())
            score_norm = score / max_score if max_score > 0 else 0.0
            blobs_info.append({
                "rank": i + 1,
                "index": idx,
                "score": round(score, 6),
                "score_normalized": round(score_norm, 4),
                "feature_norm": round(feat_norm, 4),
            })
        return blobs_info

    @torch.no_grad()
    def generate(self, prompt: str, k: int | None = None, max_new: int | None = None,
                 temperature: float | None = None, top_k: int | None = None,
                 top_p: float | None = None, repetition_penalty: float | None = None,
                 no_repeat_ngram: int | None = None) -> dict:
        """Generate text and (if this model has blobs) retrieve blobs.

        Decoding defaults are tuned to suppress the base-model degeneration
        (verbatim loops) that a small 1-epoch LM falls into under plain top-k:
        a repetition penalty, no-repeat n-gram blocking, and nucleus (top-p)
        sampling on top of top-k.
        """
        k = self.k if k is None else k
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        top_k = 50 if top_k is None else top_k
        top_p = 0.92 if top_p is None else top_p
        repetition_penalty = 1.3 if repetition_penalty is None else repetition_penalty
        no_repeat_ngram = 3 if no_repeat_ngram is None else no_repeat_ngram

        prompt_ids = self._tokenize(prompt)  # [1, L]
        prompt_len = prompt_ids.shape[1]

        blobs_info = self._retrieve_blobs(prompt_ids, k)

        t0 = time.perf_counter()
        ids = prompt_ids.clone()
        generated_tokens = []
        for _ in range(max_new):
            ctx = ids[:, -self.model.max_len:]
            logits = self.model.forward(ctx)[:, -1, :]  # [1, V]

            # ── Repetition penalty (CTRL-style): divide logits of already-seen
            #    tokens so they're less likely to be picked again. ──
            if repetition_penalty and repetition_penalty != 1.0:
                seen = torch.unique(ids[0])
                sel = logits[0, seen]
                logits[0, seen] = torch.where(
                    sel > 0, sel / repetition_penalty, sel * repetition_penalty)

            # ── No-repeat n-gram: hard-ban any token that would complete an
            #    n-gram already generated (kills exact phrase loops). ──
            if no_repeat_ngram and no_repeat_ngram > 0 and len(generated_tokens) >= no_repeat_ngram - 1:
                seq = generated_tokens
                prefix = tuple(seq[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else tuple()
                banned = set()
                for i in range(len(seq) - no_repeat_ngram + 1):
                    if tuple(seq[i:i + no_repeat_ngram - 1]) == prefix:
                        banned.add(seq[i + no_repeat_ngram - 1])
                for tok in banned:
                    logits[0, tok] = float("-inf")

            logits = logits / max(temperature, 1e-8)

            # ── top-k ──
            if top_k and top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # ── top-p (nucleus): keep the smallest set whose cumulative prob
            #    exceeds top_p; drop the long improbable tail. ──
            if top_p and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits[0, sorted_idx[0, remove[0]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            generated_tokens.append(int(next_id[0, 0].item()))
        gen_seconds = time.perf_counter() - t0

        generated_text = self._decode(generated_tokens)
        n_gen = len(generated_tokens)
        return {
            "model": self.name,
            "model_label": self.label,
            "has_blobs": self.has_blobs,
            "prompt": prompt,
            "generated_text": generated_text,
            "prompt_tokens": prompt_len,
            "generated_tokens": n_gen,
            "k": k,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram": no_repeat_ngram,
            "gen_seconds": round(gen_seconds, 3),
            "tokens_per_sec": round(n_gen / gen_seconds, 1) if gen_seconds > 0 else None,
            "blobs": blobs_info,
        }


class RuntimeManager:
    """Lazily loads runtimes and tracks the active model. One per model name."""

    def __init__(self, args):
        self.args = args
        self.runtimes: dict[str, SatzRuntime] = {}
        self.active_name: str | None = None
        # Startup path overrides apply only to the initially-loaded model.
        self._overrides = {
            "checkpoint": args.checkpoint,
            "tokenizer": args.tokenizer,
            "blobs_dir": args.blobs_dir,
        }
        self.load(args.model, apply_overrides=True)

    def load(self, name: str, apply_overrides: bool = False) -> SatzRuntime:
        if name not in MODELS:
            raise KeyError(f"unknown model '{name}'")
        if name not in self.runtimes:
            spec = dict(MODELS[name])
            if apply_overrides:
                for key, val in self._overrides.items():
                    if val is not None:
                        spec[key] = val
                # An explicit --blobs-dir override implies blobs are present.
                if self._overrides["blobs_dir"] is not None:
                    spec["blobs_dir"] = self._overrides["blobs_dir"]
            self.runtimes[name] = SatzRuntime(
                name, spec, self.args.k, self.args.max_new, self.args.temperature)
        self.active_name = name
        return self.runtimes[name]

    @property
    def active(self) -> SatzRuntime:
        return self.runtimes[self.active_name]


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Satz demo v0.2")
manager: RuntimeManager | None = None
conversation_log: ConversationLogger | None = None

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    k: int | None = None
    max_new: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram: int | None = None
    session_id: str | None = None  # optional client-supplied conversation id


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/models")
def models():
    """Registry keys + whether each has blobs + which is active."""
    return {
        "active": manager.active_name if manager else None,
        "models": [
            {"name": name, "label": spec["label"],
             "has_blobs": spec.get("blobs_dir") is not None,
             "loaded": name in manager.runtimes if manager else False}
            for name, spec in MODELS.items()
        ],
    }


@app.get("/health")
def health():
    rt = manager.active if manager else None
    return {
        "ok": rt is not None,
        "model": rt.name if rt else None,
        "model_label": rt.label if rt else None,
        "device": str(rt.device) if rt else "uninitialised",
        "model_params": f"{rt.model.count_parameters()/1e6:.1f}M" if rt else None,
        "has_blobs": rt.has_blobs if rt else None,
        "n_blobs": rt.n_blobs if rt else None,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    if manager is None:
        raise HTTPException(status_code=503, detail="runtime not initialised")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    # Switch model if requested (lazy-loads on first use).
    if req.model and req.model != manager.active_name:
        if req.model not in MODELS:
            raise HTTPException(status_code=400, detail=f"unknown model '{req.model}'")
        try:
            manager.load(req.model)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"cannot load {req.model}: {e}")
    try:
        result = manager.active.generate(
            prompt, k=req.k, max_new=req.max_new, temperature=req.temperature,
            top_k=req.top_k, top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            no_repeat_ngram=req.no_repeat_ngram)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Log the conversation turn (best-effort; never blocks the response) ──
    if conversation_log is not None:
        record = {
            "id": uuid.uuid4().hex,
            "session_id": req.session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": result["model"],
            "prompt": result["prompt"],
            "generated_text": result["generated_text"],
            "prompt_tokens": result["prompt_tokens"],
            "generated_tokens": result["generated_tokens"],
            "k": result["k"],
            "temperature": result["temperature"],
            "gen_seconds": result.get("gen_seconds"),
            "tokens_per_sec": result.get("tokens_per_sec"),
            "has_blobs": result["has_blobs"],
            # Store blob indices/scores only (compact); full features live in the store.
            "blobs": [{"index": b["index"], "score": b["score"]} for b in result["blobs"]],
        }
        conversation_log.log(record)

    return JSONResponse(result)


def main():
    global manager, conversation_log
    args = parse_args()
    conversation_log = ConversationLogger(None if args.no_log else args.log_file)
    manager = RuntimeManager(args)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
