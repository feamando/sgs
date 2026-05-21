"""
Raum 1.3 Phase 3: Inference pipeline (prompt -> tree -> render).

Takes a text prompt, feeds it to the fine-tuned decomposer, parses
the output as a composition tree JSON, and serves it via the scene
viewer.

Usage:
    python scripts/infer_decomposer.py `
      --checkpoint checkpoints/planck_decomposer/best.pt `
      --tokenizer data/wikipedia/tokenizer.model `
      --prompt "a castle on a hill"

    python scripts/infer_decomposer.py `
      --checkpoint checkpoints/planck_decomposer/best.pt `
      --tokenizer data/wikipedia/tokenizer.model `
      --serve --port 8003
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import CompositionNode, tree_to_tensors, print_tree


def parse_args():
    p = argparse.ArgumentParser(description="Raum 1.3 inference: prompt -> tree -> render")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned decomposer checkpoint")
    p.add_argument("--tokenizer", required=True, help="SentencePiece model")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt to decompose")
    p.add_argument("--serve", action="store_true", help="Launch web UI for interactive prompts")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-new", type=int, default=4096, help="Max tokens to generate for tree")
    p.add_argument("--temperature", type=float, default=0.3, help="Low temp for structured output")
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


class Decomposer:
    """Wraps the fine-tuned Planck model for tree generation."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: torch.device):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.device = device

        from src.sgs_lm import SGSLanguageModel, migrate_state_dict
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = migrate_state_dict(state)
        vocab_size = state["tok_mu.weight"].shape[0]

        self.model = SGSLanguageModel(vocab_size=vocab_size)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def generate_tree(self, prompt: str, max_new: int = 4096,
                      temperature: float = 0.3, top_k: int = 30) -> dict | None:
        """
        Generate a composition tree from a text prompt.

        The model generates a STRUCTURE-ONLY tree (names, positions, scales,
        children) without gaussians. Gaussians are filled in procedurally
        at render time based on node names. This keeps output within the
        512-token context window.

        Returns the parsed tree dict, or None if parsing fails.
        """
        input_text = f"DECOMPOSE: {prompt}\nTREE: "
        input_ids = self.sp.encode(input_text, out_type=int)

        ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = []

        for _ in range(max_new):
            if ids_t.shape[1] > 512:
                ids_t = ids_t[:, -512:]

            logits = self.model(ids_t)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                topk_vals, topk_idx = next_logits.topk(top_k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(0, topk_idx, topk_vals)
                next_logits = mask

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == self.sp.eos_id():
                break

            generated.append(next_id)
            ids_t = torch.cat([ids_t, torch.tensor([[next_id]], device=self.device)], dim=1)

        # Decode and parse JSON
        output_text = self.sp.decode(generated)

        # Try to extract JSON from the output
        tree = self._parse_tree_json(output_text)

        # If we got a tree, fill in gaussians procedurally for leaf nodes
        if tree:
            self._fill_gaussians(tree)

        return tree

    def _fill_gaussians(self, node: dict):
        """Recursively fill leaf nodes with procedural Gaussians based on name."""
        import math, random

        if "children" in node and node["children"]:
            for child in node["children"]:
                self._fill_gaussians(child)
        elif "gaussians" not in node or not node.get("gaussians"):
            # Leaf node without gaussians: generate procedurally
            name = node.get("name", "").lower()
            color = node.get("color", [0.6, 0.6, 0.6])

            # Pick primitive type from name
            n = 30
            gaussians = []
            if any(w in name for w in ["ground", "floor", "plane", "field", "road", "water", "sand", "snow"]):
                # Flat plane
                side = int(math.sqrt(n))
                for i in range(side):
                    for j in range(side):
                        x = (i/side - 0.5) * 2.0
                        z = (j/side - 0.5) * 2.0
                        gaussians.append({"position": [x, random.uniform(-0.02, 0.02), z],
                                          "scale": [-2.5, -2.5, -2.5], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["tower", "trunk", "pole", "post", "column", "pillar", "mast", "chimney"]):
                # Cylinder
                for i in range(n):
                    theta = 2*math.pi*i/n
                    y = (i/n) * 1.0 - 0.5
                    gaussians.append({"position": [0.25*math.cos(theta), y, 0.25*math.sin(theta)],
                                      "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["roof", "cone", "top", "canopy", "cap"]):
                # Cone
                for i in range(n):
                    t = i/n
                    theta = 2*math.pi*(i*7)/n
                    r = 0.3*(1-t)
                    gaussians.append({"position": [r*math.cos(theta), t*0.5, r*math.sin(theta)],
                                      "scale": [-3.2, -3.2, -3.2], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["wall", "fence", "gate", "door", "box", "body", "hull", "block"]):
                # Box
                for _ in range(n):
                    axis = random.randint(0, 2)
                    sign = random.choice([-1, 1])
                    pos = [random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)]
                    pos[axis] = sign * 0.4
                    gaussians.append({"position": pos, "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})
            else:
                # Default: sphere
                golden = (1+math.sqrt(5))/2
                for i in range(n):
                    theta = 2*math.pi*i/golden
                    phi = math.acos(1-2*(i+0.5)/n)
                    r = 0.3
                    gaussians.append({"position": [r*math.sin(phi)*math.cos(theta), r*math.sin(phi)*math.sin(theta), r*math.cos(phi)],
                                      "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})

            node["gaussians"] = gaussians

    def _parse_tree_json(self, text: str) -> dict | None:
        """Attempt to parse a composition tree from generated text."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        start = text.find("{")
        if start == -1:
            return None

        # Find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None

        # JSON was truncated (model hit max tokens). Try to repair by
        # closing open brackets/braces.
        truncated = text[start:]
        # Count unclosed brackets
        open_braces = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")

        # Truncate at last complete object/array entry
        # Find the last comma or complete value
        repair = truncated.rstrip()
        if repair.endswith(","):
            repair = repair[:-1]

        # Close all open brackets
        repair += "]" * open_brackets + "}" * open_braces

        try:
            return json.loads(repair)
        except json.JSONDecodeError:
            # More aggressive: truncate to last complete gaussians entry
            # Find last complete "}" before the truncation
            last_close = repair.rfind("}")
            if last_close > 0:
                # Try progressively shorter substrings
                for cut in range(last_close, max(last_close - 200, 0), -1):
                    if repair[cut] == "}":
                        attempt = repair[:cut+1]
                        ob = attempt.count("{") - attempt.count("}")
                        obrk = attempt.count("[") - attempt.count("]")
                        attempt += "]" * obrk + "}" * ob
                        try:
                            return json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
            return None


VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"/><title>Raum 1.3: Recursive Decomposition</title>
<style>
* { box-sizing: border-box; margin: 0; }
body { background: #0a0a0f; color: #f5f1e8; font-family: Inter, sans-serif;
  display: grid; grid-template-rows: auto 1fr; height: 100vh; overflow: hidden; }
header { padding: 12px 20px; border-bottom: 1px solid #1f1f2a;
  display: flex; align-items: center; justify-content: space-between; }
header h1 { font-size: 15px; font-weight: 700; }
header .status { font-size: 11px; color: #8a8598; font-family: monospace; }
main { display: grid; grid-template-columns: 320px 1fr; overflow: hidden; }
.sidebar { padding: 16px; border-right: 1px solid #1f1f2a; overflow-y: auto; }
.sidebar label { font-size: 11px; color: #8a8598; display: block; margin-bottom: 6px; }
.sidebar textarea { width: 100%; height: 60px; background: #12121a; border: 1px solid #1f1f2a;
  color: #f5f1e8; border-radius: 6px; padding: 8px; font-size: 13px; resize: vertical; font-family: inherit; }
.sidebar textarea:focus { border-color: #ffb347; outline: none; }
.sidebar button { width: 100%; margin-top: 10px; padding: 10px; font-size: 13px; font-weight: 600;
  background: #ffb347; color: #0a0a0f; border: none; border-radius: 6px; cursor: pointer; }
.sidebar button:hover { filter: brightness(1.1); }
.sidebar button:disabled { background: #3a3a48; color: #8a8598; cursor: wait; }
.tree-output { margin-top: 16px; padding-top: 12px; border-top: 1px solid #1f1f2a; }
.tree-output pre { font-size: 10px; color: #8a8598; white-space: pre-wrap; max-height: 300px; overflow-y: auto;
  background: #12121a; padding: 8px; border-radius: 6px; }
.tree-output .stats { font-size: 11px; color: #ffb347; margin-bottom: 8px; }
#viewer { position: relative; }
#info { position: absolute; top: 10px; right: 10px; font-size: 11px; color: #8a8598;
  background: rgba(18,18,26,0.9); padding: 8px 12px; border-radius: 6px; }
</style></head><body>
<header>
  <h1>Raum 1.3 / Recursive Decomposition</h1>
  <span class="status" id="status">ready</span>
</header>
<main>
  <div class="sidebar">
    <label>Scene prompt</label>
    <textarea id="prompt" placeholder="a castle on a hill">a castle on a hill</textarea>
    <button id="generate">Decompose + Render</button>
    <div class="tree-output" id="tree-panel">
      <div class="stats" id="stats"></div>
      <pre id="tree-text"></pre>
    </div>
  </div>
  <div id="viewer">
    <div id="info"></div>
  </div>
</main>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const viewer = document.getElementById('viewer');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth/viewer.clientHeight, 0.1, 100);
camera.position.set(3, 3, 5);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
renderer.setClearColor(0x0a0a0f);
viewer.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.GridHelper(10, 20, 0x1f1f2a, 0x1f1f2a));
scene.add(new THREE.AxesHelper(2));

let points = null;

function renderSplats(data) {
  if (points) scene.remove(points);
  const { means, colors, scales, n_splats } = data;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n_splats * 3);
  const col = new Float32Array(n_splats * 3);
  for (let i = 0; i < n_splats; i++) {
    pos[i*3] = means[i][0]; pos[i*3+1] = means[i][1]; pos[i*3+2] = means[i][2];
    col[i*3] = colors[i][0]; col[i*3+1] = colors[i][1]; col[i*3+2] = colors[i][2];
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  points = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.03, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.85
  }));
  scene.add(points);
  document.getElementById('info').textContent = `${n_splats} gaussians`;
}

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();

const btn = document.getElementById('generate');
const promptEl = document.getElementById('prompt');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const treeEl = document.getElementById('tree-text');

btn.addEventListener('click', async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  btn.disabled = true;
  statusEl.textContent = 'decomposing...';
  try {
    const r = await fetch('/decompose', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({prompt}),
    });
    const data = await r.json();
    if (data.error) {
      statusEl.textContent = 'error: ' + data.error;
      treeEl.textContent = data.raw_output || '';
      return;
    }
    renderSplats(data.splats);
    statsEl.textContent = `${data.n_gaussians} gaussians | depth ${data.depth} | ${data.n_children} top-level parts`;
    treeEl.textContent = JSON.stringify(data.tree, null, 2);
    statusEl.textContent = 'ready';
  } catch(e) {
    statusEl.textContent = 'error: ' + e.message;
  } finally { btn.disabled = false; }
});

window.addEventListener('resize', () => {
  camera.aspect = viewer.clientWidth/viewer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
});
</script></body></html>"""


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading decomposer: {args.checkpoint}")
    decomposer = Decomposer(args.checkpoint, args.tokenizer, device)
    print(f"  Vocab: {decomposer.vocab_size}, ready.")

    if args.prompt and not args.serve:
        # Single inference
        print(f"\nPrompt: {args.prompt}")
        print("Generating tree...")
        tree_dict = decomposer.generate_tree(
            args.prompt, max_new=args.max_new,
            temperature=args.temperature, top_k=args.top_k,
        )
        if tree_dict is None:
            # Debug: show what was generated
            input_text = f"DECOMPOSE: {args.prompt}\nTREE: "
            input_ids = decomposer.sp.encode(input_text, out_type=int)
            ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            generated = []
            with torch.no_grad():
                for _ in range(500):
                    if ids_t.shape[1] > 512:
                        ids_t = ids_t[:, -512:]
                    logits = decomposer.model(ids_t)
                    next_logits = logits[0, -1, :] / args.temperature
                    topk_vals, topk_idx = next_logits.topk(args.top_k)
                    mask_t = torch.full_like(next_logits, float("-inf"))
                    mask_t.scatter_(0, topk_idx, topk_vals)
                    probs = torch.softmax(mask_t, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == decomposer.sp.eos_id():
                        break
                    generated.append(next_id)
                    ids_t = torch.cat([ids_t, torch.tensor([[next_id]], device=device)], dim=1)
            raw = decomposer.sp.decode(generated)
            print(f"ERROR: failed to generate valid JSON tree")
            print(f"\nRaw output (first 500 chars):")
            print(raw[:500])
            sys.exit(1)

        tree = CompositionNode.from_dict(tree_dict)
        print(f"\n=== Composition Tree ===")
        print_tree(tree)

        tensors = tree_to_tensors(tree)
        print(f"\n  Gaussians: {tensors['means'].shape[0]}")

        # Save
        out_path = Path("data/scenes/generated_scene.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(tree_dict, f, indent=2)
        print(f"  Saved: {out_path}")
        print(f"\nView: python scripts/render_scene.py --scene {out_path}")

    elif args.serve:
        # Web UI mode
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel

        app = FastAPI()

        class DecomposeRequest(BaseModel):
            prompt: str

        @app.get("/", response_class=HTMLResponse)
        def index():
            return VIEWER_HTML

        @app.post("/decompose")
        def decompose(req: DecomposeRequest):
            prompt = req.prompt.strip()
            if not prompt:
                return JSONResponse({"error": "empty prompt"})

            tree_dict = decomposer.generate_tree(
                prompt, max_new=args.max_new,
                temperature=args.temperature, top_k=args.top_k,
            )

            if tree_dict is None:
                return JSONResponse({"error": "failed to parse tree JSON", "raw_output": ""})

            try:
                tree = CompositionNode.from_dict(tree_dict)
                tensors = tree_to_tensors(tree)
            except Exception as e:
                return JSONResponse({"error": f"tree parse error: {e}"})

            n = tensors["means"].shape[0]
            return JSONResponse({
                "tree": tree_dict,
                "splats": {
                    "means": tensors["means"].tolist(),
                    "scales": tensors["scales_log"].tolist(),
                    "opacities": torch.sigmoid(tensors["opacities"]).tolist(),
                    "colors": tensors["colors"].tolist(),
                    "n_splats": n,
                },
                "n_gaussians": n,
                "depth": tree.depth,
                "n_children": len(tree.children),
            })

        import uvicorn
        print(f"\nServing at http://{args.host}:{args.port}")
        print("Enter a prompt in the UI to decompose + render.")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

    else:
        print("Specify --prompt or --serve")


if __name__ == "__main__":
    main()
