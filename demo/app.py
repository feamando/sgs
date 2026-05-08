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
from src.raum.dsl import bridge_output_to_dsl, dsl_to_json, json_to_dsl, validate


def parse_args():
    p = argparse.ArgumentParser(description="Raum demo server")
    p.add_argument("--checkpoint", required=True, help="Path to routing-bridge best.pt")
    p.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")
    p.add_argument("--encoder-checkpoint", type=str, default=None,
                   help="Path to Planck checkpoint (enables 1.1 mode)")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="SentencePiece tokenizer model (auto-detected)")
    p.add_argument("--blobs-dir", type=str, default=None,
                   help="Path to blob library for template names")
    p.add_argument("--d-s", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--template-points", type=int, default=1000,
                   help="Points per object template.")
    p.add_argument("--template-confidence", type=float, default=0.35,
                   help="Min softmax confidence to stamp a template; below this "
                        "the object is flagged as unresolved.")
    p.add_argument("--vocab-size", type=int, default=50000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=32)
    return p.parse_args()


class RaumRuntime:
    """Loads vocab + bridge + template library once, generates scenes."""

    def __init__(self, args):
        self.max_tokens = args.max_tokens
        self.template_confidence = args.template_confidence
        self.use_encoder = args.encoder_checkpoint is not None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[raum] loading GloVe ...")
        word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=args.vocab_size)
        self.word2idx = word2idx

        # Load bridge checkpoint and infer architecture
        print(f"[raum] loading bridge checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        d_model = state["pos_emb"].shape[1]
        n_blobs = state["template_head.weight"].shape[0]
        d_in = state["input_proj.weight"].shape[1]
        n_layers = 0
        while f"encoder.layers.{n_layers}.self_attn.in_proj_weight" in state:
            n_layers += 1
        for nh in [8, 6, 4, 2, 1]:
            if d_model % nh == 0:
                n_heads = nh
                break
        has_relation = "relation_head.0.weight" in state
        print(f"[raum] inferred: d_model={d_model}, n_layers={n_layers}, "
              f"n_heads={n_heads}, n_blobs={n_blobs}, relation_head={has_relation}")

        if self.use_encoder:
            from src.raum.encoder import FrozenPlanckEncoder, build_sp_word2idx
            print(f"[raum] loading encoder: {args.encoder_checkpoint}")
            self.encoder = FrozenPlanckEncoder(args.encoder_checkpoint, device=self.device)
            self.encoder.to(self.device).eval()
            d_s = self.encoder.d_s
            d_f = self.encoder.d_f

            tokenizer_path = args.tokenizer
            if tokenizer_path is None:
                for c in [Path("data/wikipedia/tokenizer.model"),
                          Path("data/tinystories/tokenizer.model")]:
                    if c.exists():
                        tokenizer_path = str(c)
                        break
            if tokenizer_path and Path(tokenizer_path).exists():
                self.word2idx = build_sp_word2idx(tokenizer_path)
                print(f"[raum] SP tokenizer: {tokenizer_path}")

            self.vocab = None
        else:
            d_f = vectors.shape[1]
            d_s = args.d_s
            print("[raum] building SGS vocab ...")
            self.vocab = SemanticGaussianVocab(len(words), d_s=d_s, d_f=d_f)
            self.vocab.init_from_glove(vectors, freqs)
            self.vocab.to(self.device).eval()
            self.encoder = None

        self.model = RaumBridge(
            d_s=d_s, d_f=d_f,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_blobs=n_blobs,
            with_relation_head=has_relation,
            K=args.K,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        print("[raum] building template library ...")
        self.template_lib = build_template_library(n_gaussians=args.template_points)

        if args.blobs_dir and Path(args.blobs_dir).exists():
            import json
            with open(Path(args.blobs_dir) / "index.json") as f:
                self.template_names = json.load(f)
            print(f"[raum] blob library: {len(self.template_names)} classes")
        else:
            self.template_names = list(OBJECTS.keys())

        print(f"[raum] ready on {self.device} | {self.model.count_parameters():,} params")

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

        if self.use_encoder:
            mu_s, features = self.encoder.encode(token_ids)
        else:
            mu_s, _, _, features = self.vocab.get_params(token_ids)
        out = self.model(mu_s, features, mask)

        # Gate object selection on the role head so non-object tokens
        # ("a", "above", "red") don't get stamped.
        splats, objects, unresolved = assemble_scene(
            out,
            self.template_lib,
            self.template_names,
            mask=mask,
            sample_index=0,
            object_role_id=ROLE_OBJECT,
            template_confidence_threshold=self.template_confidence,
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

        warnings = []
        for u in unresolved:
            w = words[u.word_index] if u.word_index < len(words) else "?"
            warnings.append(
                f"could not resolve \u201c{w}\u201d to a known template "
                f"(top guess {u.top_template_name}, conf "
                f"{int(round(u.template_confidence * 100))}%)"
            )

        # Build DSL from bridge output
        dsl = bridge_output_to_dsl(
            out, mask,
            blob_names=self.template_names,
            sample_index=0,
            object_role_id=ROLE_OBJECT,
        )

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
            "unresolved": [
                {
                    "word_index": u.word_index,
                    "word": words[u.word_index] if u.word_index < len(words) else "",
                    "top_template": u.top_template_name,
                    "top_template_id": u.top_template_id,
                    "confidence": u.template_confidence,
                    "position": u.position,
                }
                for u in unresolved
            ],
            "warnings": warnings,
            "dsl": dsl,
            "splats": {
                "means": means.cpu().tolist(),
                "scales": scales.cpu().tolist(),
                "opacities": opacities.cpu().tolist(),
                "colors": colors.cpu().tolist(),
            },
            "n_splats": int(means.shape[0]),
            "n_objects": len(objects),
            "n_unresolved": len(unresolved),
        }


app = FastAPI(title="Raum demo")
runtime: RaumRuntime | None = None

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    prompt: str


class RenderDSLRequest(BaseModel):
    dsl: dict


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


@app.post("/render-dsl")
def render_dsl(req: RenderDSLRequest):
    """Re-render from an edited DSL without re-running the bridge."""
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime not initialised")

    valid, errors = validate(req.dsl)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Build splats from DSL objects using the template library
    import torch
    all_means = []
    all_scales_log = []
    all_opacities = []
    all_colors = []

    for obj in req.dsl.get("objects", []):
        blob_name = obj.get("blob", "")
        if blob_name not in runtime.template_lib:
            continue
        tpl = runtime.template_lib[blob_name]
        pos = torch.tensor(obj.get("position", [0, 0, 0]), dtype=torch.float32)
        col = torch.tensor(obj.get("color", [0.7, 0.7, 0.7]), dtype=torch.float32).clamp(0, 1)
        scl = float(obj.get("scale", 1.0))

        means = tpl.means * scl + pos.unsqueeze(0)
        sc_log = tpl.scales.clone() + torch.log(torch.tensor(max(scl, 1e-3)))
        all_means.append(means)
        all_scales_log.append(sc_log)
        all_opacities.append(tpl.opacities.clone())
        all_colors.append(col.unsqueeze(0).expand(means.shape[0], 3).clone())

    if all_means:
        means = torch.cat(all_means, dim=0)
        scales = torch.cat(all_scales_log, dim=0).exp()
        opacities = torch.sigmoid(torch.cat(all_opacities, dim=0))
        colors = torch.cat(all_colors, dim=0)
    else:
        means = torch.zeros(0, 3)
        scales = torch.zeros(0, 3)
        opacities = torch.zeros(0)
        colors = torch.zeros(0, 3)

    return JSONResponse({
        "dsl": req.dsl,
        "splats": {
            "means": means.tolist(),
            "scales": scales.tolist(),
            "opacities": opacities.tolist(),
            "colors": colors.tolist(),
        },
        "n_splats": int(means.shape[0]),
        "n_objects": len(req.dsl.get("objects", [])),
    })


@app.post("/validate-dsl")
def validate_dsl_endpoint(req: RenderDSLRequest):
    """Validate DSL without rendering."""
    valid, errors = validate(req.dsl)
    return JSONResponse({"valid": valid, "errors": errors})


def main():
    global runtime
    args = parse_args()
    runtime = RaumRuntime(args)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
