"""
Satz demo v0.1: local web app.

Loads a trained Planck 1.3 checkpoint + blob store and serves a FastAPI
endpoint that generates text and shows which blobs are retrieved for a
given prompt.

Run:
    python -m satz.app --checkpoint checkpoints/planck/best.pt `
                       --tokenizer data/wikipedia/tokenizer.model `
                       --blobs-dir data/blobs/wikipedia

Then open http://localhost:8001 in a browser.
"""

import argparse
import json
import sys
from pathlib import Path

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


def parse_args():
    p = argparse.ArgumentParser(description="Satz demo server")
    p.add_argument("--checkpoint", required=True,
                   help="Path to Planck 1.3 best.pt checkpoint")
    p.add_argument("--tokenizer", default="data/wikipedia/tokenizer.model",
                   help="SentencePiece model file")
    p.add_argument("--blobs-dir", default="data/blobs/wikipedia",
                   help="Directory containing blobs.pt and meta.json")
    p.add_argument("--k", type=int, default=8,
                   help="Top-k blobs to retrieve (default 8)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-new", type=int, default=200,
                   help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    # Architecture (must match checkpoint)
    p.add_argument("--d-s", type=int, default=128)
    p.add_argument("--d-f", type=int, default=1000)
    p.add_argument("--n-passes", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--ffn-mult", type=int, default=4)
    return p.parse_args()


class SatzRuntime:
    """Loads Planck checkpoint + blob store, generates text + retrieves blobs."""

    def __init__(self, args):
        self.max_new = args.max_new
        self.temperature = args.temperature
        self.k = args.k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load tokenizer ──
        tokenizer_path = Path(args.tokenizer)
        if not tokenizer_path.exists():
            # Try relative to project root
            tokenizer_path = ROOT / args.tokenizer
        if not tokenizer_path.exists():
            print(f"[satz] ERROR: tokenizer not found at {args.tokenizer}")
            print(f"       Checked: {Path(args.tokenizer).resolve()}")
            print(f"       Checked: {tokenizer_path}")
            sys.exit(1)

        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))
        self.vocab_size = self.sp.get_piece_size()
        print(f"[satz] tokenizer: {tokenizer_path} ({self.vocab_size} tokens)")

        # ── Load blob store ──
        blobs_dir = Path(args.blobs_dir)
        if not blobs_dir.exists():
            blobs_dir = ROOT / args.blobs_dir
        if not blobs_dir.exists():
            print(f"[satz] ERROR: blob directory not found at {args.blobs_dir}")
            print(f"       Run scripts/build_blobs.py first to create the blob index.")
            sys.exit(1)

        blobs_pt = blobs_dir / "blobs.pt"
        meta_json = blobs_dir / "meta.json"
        if not blobs_pt.exists():
            print(f"[satz] ERROR: blobs.pt not found in {blobs_dir}")
            print(f"       Run scripts/build_blobs.py first.")
            sys.exit(1)

        print(f"[satz] loading blobs from {blobs_dir} ...")
        blob_data = torch.load(blobs_pt, map_location="cpu", weights_only=False)
        n_blobs = blob_data["mu"].shape[0]

        meta = {}
        if meta_json.exists():
            with open(meta_json) as f:
                meta = json.load(f)

        print(f"[satz] blob store: {n_blobs:,} blobs, "
              f"d_s={blob_data['mu'].shape[1]}, d_f={blob_data['features'].shape[1]}")

        self.blob_store = BlobStore(
            n_blobs=n_blobs,
            d_s=args.d_s,
            d_f=args.d_f,
            k=self.k,
        )
        self.blob_store.init_from_clusters(
            mu=blob_data["mu"],
            log_var=blob_data["log_var"],
            alpha=blob_data["raw_alpha"],
            features=blob_data["features"],
        )
        self.blob_store.to(self.device).eval()

        # ── Load Planck model ──
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            ckpt_path = ROOT / args.checkpoint
        if not ckpt_path.exists():
            print(f"[satz] ERROR: checkpoint not found at {args.checkpoint}")
            sys.exit(1)

        print(f"[satz] loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = migrate_state_dict(state)

        ckpt_vocab_size = state["tok_mu.weight"].shape[0]
        self.model = SGSLanguageModel(
            vocab_size=ckpt_vocab_size,
            d_s=args.d_s,
            d_f=args.d_f,
            n_passes=args.n_passes,
            n_heads=args.n_heads,
            max_len=args.context_len,
            ffn_mult=args.ffn_mult,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[satz] model: {self.model.count_parameters()/1e6:.1f}M params on {self.device}")
        print(f"[satz] ready.")

    def _tokenize(self, text: str) -> torch.Tensor:
        """Encode text to token IDs using SentencePiece."""
        ids = self.sp.encode(text)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)

    def _compute_query(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute blob query from input tokens (mean of token mu's)."""
        with torch.no_grad():
            mu = self.model.tok_mu(token_ids)  # [B, L, d_s]
            pos = torch.arange(token_ids.shape[1], device=token_ids.device)
            mu = mu + self.model.pos_mu(pos).unsqueeze(0)
            return mu.mean(dim=1)  # [B, d_s]

    @torch.no_grad()
    def generate(self, prompt: str, k: int | None = None, max_new: int | None = None,
                 temperature: float | None = None) -> dict:
        """Generate text and retrieve blobs for a prompt."""
        if k is not None:
            self.blob_store.k = k
        else:
            k = self.blob_store.k

        if max_new is None:
            max_new = self.max_new
        if temperature is None:
            temperature = self.temperature

        # Tokenize
        prompt_ids = self._tokenize(prompt)  # [1, L]
        prompt_len = prompt_ids.shape[1]

        # Retrieve blobs based on prompt
        query = self._compute_query(prompt_ids)  # [1, d_s]
        top_idx, top_scores = self.blob_store.retrieve(query)  # [1, k]

        # Get blob feature norms as a proxy for content richness
        blob_indices = top_idx[0].cpu().tolist()
        blob_scores = top_scores[0].cpu().tolist()
        blob_feature_norms = []
        for idx in blob_indices:
            feat = self.blob_store.features[idx]
            blob_feature_norms.append(float(feat.norm().item()))

        # Normalize scores to [0, 1] for display
        max_score = max(blob_scores) if blob_scores else 1.0
        blob_scores_normalized = [s / max_score if max_score > 0 else 0.0 for s in blob_scores]

        # Generate text autoregressively
        ids = prompt_ids.clone()
        generated_tokens = []

        for _ in range(max_new):
            ctx = ids[:, -self.model.max_len:]
            logits = self.model.forward(ctx)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            # Top-k sampling
            top_k_val = 50
            v, _ = logits.topk(min(top_k_val, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            generated_tokens.append(int(next_id[0, 0].item()))

        # Decode generated text
        generated_text = self._decode(generated_tokens)

        # Build blob info for response
        blobs_info = []
        for i, (idx, score, score_norm, feat_norm) in enumerate(
            zip(blob_indices, blob_scores, blob_scores_normalized, blob_feature_norms)
        ):
            blobs_info.append({
                "rank": i + 1,
                "index": idx,
                "score": round(score, 6),
                "score_normalized": round(score_norm, 4),
                "feature_norm": round(feat_norm, 4),
            })

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated_tokens),
            "k": k,
            "temperature": temperature,
            "blobs": blobs_info,
        }


# ── FastAPI app ──

app = FastAPI(title="Satz demo v0.1")
runtime: SatzRuntime | None = None

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    prompt: str
    k: int | None = None
    max_new: int | None = None
    temperature: float | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(runtime.device) if runtime else "uninitialised",
        "model_params": f"{runtime.model.count_parameters()/1e6:.1f}M" if runtime else None,
        "n_blobs": runtime.blob_store.n_blobs if runtime else None,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not initialised")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    try:
        result = runtime.generate(
            prompt,
            k=req.k,
            max_new=req.max_new,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(result)


def main():
    global runtime
    args = parse_args()
    runtime = SatzRuntime(args)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
