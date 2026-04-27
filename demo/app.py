"""
Raum demo v0: local web app.

Loads a trained routing-bridge checkpoint and serves a FastAPI endpoint
that turns a natural-language prompt into a 3D scene. Each predicted
object token is routed to an object template (sphere/cube/cone/...)
and stamped at its predicted position, colour, and size. The resulting
Gaussian cloud is rendered in the browser by Three.js.

Run on Windows:
    python -m demo.app --checkpoint checkpoints\\raum_10\\best.pt ^
                       --glove data\\glove.6B.300d.txt

Then open http://localhost:8000 in a browser.
"""

import argparse
import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_glove
from src.gaussian import SemanticGaussianVocab
from src.raum.bridge import RaumBridge, assemble_scene
from src.raum.templates import build_template_library
from src.raum.vocab import OBJECTS, ROLE_OBJECT


def parse_args():
    p = argparse.ArgumentParser(description="Raum demo server")
    p.add_argument("--checkpoint", required=True, help="Path to routing-bridge best.pt")
    p.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--template-points", type=int, default=200,
                   help="Points per object template.")
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=32)
    return p.parse_args()


class RaumRuntime:
    """Loads vocab + bridge + template library once, generates scenes."""

    def __init__(
        self,
        checkpoint: str,
        glove_path: str,
        d_s: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        K: int,
        template_points: int,
        vocab_size: int,
        max_tokens: int,
    ):
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
        self.model = RaumBridge(
            d_s=d_s, d_f=d_f,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            K=K,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab.to(self.device)
        self.model.to(self.device)

        print("[raum] building template library ...")
        self.template_lib = build_template_library(n_gaussians=template_points)
        self.template_names = list(OBJECTS.keys())  # sphere, cube, cylinder, ...

        print(f"[raum] ready on {self.device}")

    def tokenize(self, prompt: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
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
        out = self.model(mu_s, features, mask)

        # Gate object selection on the role head so non-object tokens
        # ("a", "above", "red") don't get stamped.
        splats, objects = assemble_scene(
            out,
            self.template_lib,
            self.template_names,
            mask=mask,
            sample_index=0,
            object_role_id=ROLE_OBJECT,
        )

        # Convert log-scale → linear scale for the viewer.
        if splats["means"].numel() > 0:
            scales = splats["scales_log"].exp()
            opacities = torch.sigmoid(splats["opacities"])
            means = splats["means"]
            colors = splats["colors"].clamp(0.0, 1.0)
        else:
            scales = torch.zeros(0, 3)
            opacities = torch.zeros(0)
            means = torch.zeros(0, 3)
            colors = torch.zeros(0, 3)

        coarse = out["positions"][0].detach().cpu().tolist()

        return {
            "words": words,
            "coarse_means": coarse,
            "objects": [
                {
                    "word_index": o.word_index,
                    "word": words[o.word_index] if o.word_index < len(words) else "",
                    "template": o.template_name,
                    "template_id": o.template_id,
                    "confidence": o.template_confidence,
                    "position": o.position,
                    "color": o.color,
                    "scale": o.scale,
                }
                for o in objects
            ],
            "splats": {
                "means": means.cpu().tolist(),
                "scales": scales.cpu().tolist(),
                "opacities": opacities.cpu().tolist(),
                "colors": colors.cpu().tolist(),
            },
            "n_splats": int(means.shape[0]),
            "n_objects": len(objects),
        }


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
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        K=args.K,
        template_points=args.template_points,
        vocab_size=args.vocab_size,
        max_tokens=args.max_tokens,
    )
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
