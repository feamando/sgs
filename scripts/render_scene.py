"""
Render a composition tree as a Gaussian splat scene.

Loads a tree JSON, flattens to tensors, serves via a minimal web viewer
(reuses the Raum demo's Three.js viewer).

Usage:
    python scripts/render_scene.py --scene data/scenes/castle_on_hill.json
    python scripts/render_scene.py --scene data/scenes/castle_on_hill.json --port 8002
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors, print_tree


def parse_args():
    p = argparse.ArgumentParser(description="Render a composition tree")
    p.add_argument("--scene", required=True, help="Path to scene JSON")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    return p.parse_args()


VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"/><title>Scene Viewer</title>
<style>
body { margin: 0; background: #0a0a0f; overflow: hidden; }
#info { position: absolute; top: 10px; left: 10px; color: #f5f1e8;
  font: 12px monospace; background: rgba(18,18,26,0.9); padding: 10px;
  border-radius: 6px; }
</style>
</head><body>
<div id="info">Loading...</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 100);
camera.position.set(3, 3, 5);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0a0a0f);
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);

// Grid
const grid = new THREE.GridHelper(10, 20, 0x1f1f2a, 0x1f1f2a);
scene.add(grid);

// Axes
scene.add(new THREE.AxesHelper(2));

// Load splats
fetch('/splats').then(r => r.json()).then(data => {
  const { means, colors, scales, opacities, n_splats, tree_info } = data;
  document.getElementById('info').innerHTML =
    `${tree_info}<br>${n_splats} gaussians | drag to orbit, scroll to zoom`;

  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(n_splats * 3);
  const colorArr = new Float32Array(n_splats * 3);
  const sizes = new Float32Array(n_splats);

  for (let i = 0; i < n_splats; i++) {
    positions[i*3] = means[i][0];
    positions[i*3+1] = means[i][1];
    positions[i*3+2] = means[i][2];
    colorArr[i*3] = colors[i][0];
    colorArr[i*3+1] = colors[i][1];
    colorArr[i*3+2] = colors[i][2];
    const s = Math.exp((scales[i][0] + scales[i][1] + scales[i][2]) / 3);
    sizes[i] = Math.max(s * 30, 1.5);
  }

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const mat = new THREE.PointsMaterial({
    size: 0.03, vertexColors: true, sizeAttenuation: true,
    transparent: true, opacity: 0.85
  });
  scene.add(new THREE.Points(geo, mat));
});

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();
window.addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script></body></html>"""


def main():
    args = parse_args()

    print(f"Loading scene: {args.scene}")
    tree = load_tree(args.scene)
    print_tree(tree)

    tensors = tree_to_tensors(tree)
    n = tensors["means"].shape[0]
    print(f"\nFlattened: {n} Gaussians")

    # Prepare data for the viewer
    splat_data = {
        "means": tensors["means"].tolist(),
        "scales": tensors["scales_log"].tolist(),
        "opacities": torch.sigmoid(tensors["opacities"]).tolist(),
        "colors": tensors["colors"].tolist(),
        "n_splats": n,
        "tree_info": f"{tree.name} | depth={tree.depth} | parts={len(tree.children)}",
    }

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return VIEWER_HTML

    @app.get("/splats")
    def splats():
        return JSONResponse(splat_data)

    import uvicorn
    print(f"\nServing at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
