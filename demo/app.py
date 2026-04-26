"""
Raum demo v0: local web app.

Loads a trained RaumBridge checkpoint and serves a FastAPI endpoint that
turns a natural-language prompt into a cloud of 3D Gaussians, rendered
in the browser (Three.js) as point sprites.

Run on Windows:
    python -m demo.app --checkpoint checkpoints/raum_c_pos/best.pt ^
                       --glove data/glove.6B.300d.txt

Then open http://localhost:8000 in a browser.
"""

import argparse
import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Windows consoles default to cp1252. Keep prints ASCII-safe instead.

# Add project root so `from src.*` imports work when run as `python -m demo.app`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_glove
from src.gaussian import SemanticGaussianVocab
from src.raum.bridge import RaumBridge


def parse_args():
    p = argparse.ArgumentParser(description="Raum demo server")
    p.add_argument("--checkpoint", required=True, help="Path to RaumBridge best.pt")
    p.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Cap on prompt token count (padding/overflow guard)")
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────

class RaumRuntime:
    """Holds the frozen SGS vocab + trained bridge and exposes one call."""

    def __init__(self, checkpoint: str, glove_path: str, d_s: int, K: int,
                 vocab_size: int, max_tokens: int):
        self.max_tokens = max_tokens

        print("[raum] loading GloVe ...")
        word2idx, vectors, freqs, words = load_glove(glove_path, vocab_size=vocab_size)
        self.word2idx = word2idx
        d_f = vectors.shape[1]

        print("[raum] building SGS vocab ...")
        self.vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
        self.vocab.init_from_glove(vectors, freqs)
        self.vocab.eval()

        print(f"[raum] loading bridge checkpoint: {checkpoint}")
        self.model = RaumBridge(d_s=d_s, d_f=d_f, K=K)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab.to(self.device)
        self.model.to(self.device)
        print(f"[raum] ready on {self.device}")

    def tokenize(self, prompt: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Lowercase, split on whitespace, map through word2idx. Drops any
        word not in GloVe to keep the semantic vector clean."""
        words = [w.strip(".,!?;:").lower() for w in prompt.split()]
        words = [w for w in words if w]
        if not words:
            raise ValueError("empty prompt")
        words = words[: self.max_tokens]

        unk_idx = self.word2idx.get("<unk>", self.word2idx.get("unk", 0))
        ids = [self.word2idx.get(w, unk_idx) for w in words]

        token_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(token_ids, dtype=torch.float32)
        return words, token_ids, mask

    @torch.no_grad()
    def generate(self, prompt: str) -> dict:
        words, token_ids, mask = self.tokenize(prompt)

        mu_s, _, _, features = self.vocab.get_params(token_ids)
        scene = self.model(mu_s, features, mask)

        # Shape: [1, N*K, *]. Squeeze batch dim.
        means = scene["means"][0]
        scales = scene["scales"][0].exp()        # log-scale → scale
        opacities = torch.sigmoid(scene["opacities"][0])
        colors = scene["colors"][0].clamp(0.0, 1.0)
        coarse = scene["coarse_means"][0]

        # Drop near-zero opacity splats to keep the payload small.
        keep = opacities > 0.02
        means = means[keep]
        scales = scales[keep]
        opacities = opacities[keep]
        colors = colors[keep]

        return {
            "words": words,
            "coarse_means": coarse.cpu().tolist(),
            "splats": {
                "means": means.cpu().tolist(),
                "scales": scales.cpu().tolist(),
                "opacities": opacities.cpu().tolist(),
                "colors": colors.cpu().tolist(),
            },
            "n_splats": int(means.shape[0]),
        }


# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(title="Raum demo")
runtime: RaumRuntime | None = None

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "device": str(runtime.device) if runtime else "uninitialised"}


@app.post("/generate")
def generate(req: GenerateRequest):
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not initialised")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")
    try:
        result = runtime.generate(prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(result)


def main():
    global runtime
    args = parse_args()
    runtime = RaumRuntime(
        checkpoint=args.checkpoint,
        glove_path=args.glove,
        d_s=args.d_s,
        K=args.K,
        vocab_size=args.vocab_size,
        max_tokens=args.max_tokens,
    )
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
