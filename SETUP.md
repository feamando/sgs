# SGS — Full Setup Guide (Windows + RTX 4090)

Fresh install from scratch. Covers all tracks: experiments, Planck LM, and Raum text-to-3D.

---

## 1. Clone the Repo

```powershell
cd C:\Users\YourName\Documents\GitHub
git clone https://github.com/feamando/sgs.git
cd sgs
```

---

## 2. Python Environment

Requires **Python 3.11 or 3.12** (NOT 3.13 — PyTorch doesn't support 3.13 on Windows yet).

### Install Python 3.12 (if you have 3.13)

1. Download Python 3.12 from https://www.python.org/downloads/release/python-3120/
   (Windows installer 64-bit)
2. During install, check **"Add to PATH"** — install alongside 3.13, don't replace it.
3. Use `py -3.12` to target the right version:

```powershell
# Create venv with Python 3.12 specifically
py -3.12 -m venv .venv
.venv\Scripts\activate
python --version   # should say 3.12.x
```

### Install dependencies

```powershell
# PyTorch with CUDA (match your nvidia-smi CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# If cu124 fails, try cu121:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# All other dependencies
pip install -r requirements.txt
```

### Optional: gsplat (for Raum PoC-C GPU rendering)

```powershell
pip install gsplat
```

If gsplat fails to install (CUDA build issues on Windows), skip it — the Raum bridge trainer auto-falls back to a CPU-compatible renderer. You can always install it later.

---

## 3. Download GloVe Embeddings

```powershell
mkdir data
cd data
```

Download `glove.6B.zip` (~862 MB) from:
**https://nlp.stanford.edu/data/glove.6B.zip**

Extract `glove.6B.300d.txt` into `data/`. Delete the zip and other files.

```powershell
cd ..
```

After this you should have: `data\glove.6B.300d.txt` (~1 GB)

---

## 4. Verify Setup

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"
```

Expected:
```
PyTorch 2.x.x
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

Quick import check:
```powershell
python -c "from src.kernel import gaussian_kernel_diag; from src.rendering import render; from src.raum.vocab import OBJECTS; print('All imports OK')"
```

---

## 5. Project Structure

```
sgs/
├── SETUP.md                    ← you are here
├── requirements.txt
├── data/                       ← GloVe + generated data (gitignored)
│   └── glove.6B.300d.txt
│
├── src/
│   ├── gaussian.py             # Semantic Gaussian primitive (Atom A1)
│   ├── kernel.py               # Gaussian kernel (Atom A2)
│   ├── rendering.py            # Alpha-compositing rendering (Atom A3)
│   ├── model.py                # SGS encoder + all ablation models
│   ├── seq2seq.py              # SCAN seq2seq models
│   ├── scan.py                 # SCAN dataset
│   ├── data.py                 # GloVe + STS-B + AllNLI loading
│   ├── sgs_lm.py               # Radiance Planck — causal SGS language model
│   ├── tinystories.py          # TinyStories data pipeline
│   └── raum/                   # Radiance Raum — text-to-3D
│       ├── vocab.py            #   scene vocabulary (objects, colors, relations)
│       ├── templates.py        #   3D shape generators as Gaussian clouds
│       ├── cameras.py          #   orbit camera utilities
│       ├── data.py             #   synthetic scene data generation
│       ├── compositional.py    #   PoC-D model (~50K params)
│       ├── assemble.py         #   scene assembly from predictions
│       ├── eval.py             #   evaluation metrics
│       ├── bridge.py           #   PoC-C model (~335K params)
│       ├── render_3d.py        #   differentiable 3DGS rendering
│       └── analyze.py          #   space transform analysis tools
│
├── scripts/
│   ├── phase0_feasibility.py   # Numerical feasibility check
│   ├── train_stsb.py           # STS-B sentence similarity training
│   ├── run_all_ablations.py    # All ablation experiments
│   ├── run_phase2.py           # Phase 2: NLI + scaling
│   ├── run_phase3.py           # Phase 3: SCAN + gap closing
│   ├── run_scan_multiseed.py   # SCAN multi-seed validation
│   ├── run_paper_fixes.py      # M2/M6 kernel isolation experiments
│   ├── train_lm.py             # Radiance Planck training
│   ├── generate.py             # Planck text generation
│   ├── export_onnx.py          # Planck → ONNX export
│   ├── train_raum_compositional.py   # Raum PoC-D training
│   ├── train_raum_bridge.py          # Raum PoC-C training
│   └── analyze_raum_bridge.py        # Raum PoC-C analysis
│
├── paper/                      # Research papers (markdown)
├── docs/                       # Plans, analysis, proofs, whitepaper
│   ├── plans/
│   ├── analysis/
│   ├── proofs/lean/            # 13 Lean 4 formal proofs
│   └── ...
└── checkpoints/                ← model checkpoints (gitignored)
```

---

## 6. What to Run

### Track A: Prior Experiments (already done, re-run to verify)

```powershell
# Phase 0: feasibility check (~10 min)
python scripts/phase0_feasibility.py --glove data/glove.6B.300d.txt

# Phase 1: STS-B all ablations (~8 hours)
python scripts/run_all_ablations.py --glove data/glove.6B.300d.txt

# Phase 2: NLI + scaling
python scripts/run_phase2.py --glove data/glove.6B.300d.txt

# Phase 3: SCAN + gap closing
python scripts/run_phase3.py --glove data/glove.6B.300d.txt

# SCAN multi-seed (5 seeds)
python scripts/run_scan_multiseed.py --glove data/glove.6B.300d.txt

# Kernel isolation (M2/M6)
python scripts/run_paper_fixes.py --glove data/glove.6B.300d.txt
```

**Results:** Each script saves a JSON file in the repo root (e.g., `ablation_results.json`, `phase1_5_results.json`, `scan_length_full_ablation.json`, `paper_fixes_results.json`).

```powershell
# Commit Track A results
git add *.json
git commit -m "Track A: experiment results from Windows machine"
git push
```

### Track B1: Radiance Planck (100M language model)

```powershell
# Step 1: Download + tokenize TinyStories (~15 min, needs internet)
python src/tinystories.py --data-dir data/tinystories

# Step 2: Train from scratch (~2-3 hours on RTX 4090)
# Default: d_s=128, d_f=1000, 3 passes, 4 heads → 100.9M params
# 3 epochs × 27.5K steps = 82K steps total
python scripts/train_lm.py --data-dir data/tinystories

# Step 3: Generate text
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --prompt "Once upon a time"

# Interactive mode
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --interactive
```

**Results:** Training logs to wandb (if enabled) and saves checkpoints to `checkpoints/planck/`. Generation samples are printed to stdout.

```powershell
# Save 50 generated samples to file for review
mkdir results 2>nul
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --prompt "Once upon a time" --n-samples 50 | Out-File results/planck_samples.txt

# Commit results (not checkpoints — too large for git)
git add results/planck_samples.txt
git commit -m "Track B1: Planck 100M generation samples"
git push
```

### Track B1-1: Radiance Hertz (1B language model — internal benchmark)

```powershell
# Step 1: Download + tokenize FineWeb-Edu (~50GB download, takes a while)
python src/tinystories.py --dataset fineweb-edu --max-tokens 10B

# Step 2: Train (~3-5 days on RTX 4090, ~10GB VRAM)
# d_s=256, d_f=3700, 5 passes, 8 heads → 1.03B params
python scripts/train_lm.py \
  --data-dir data/fineweb \
  --d-s 256 --d-f 3700 --n-passes 5 --n-heads 8 \
  --context-len 1024 \
  --batch-size 8 \
  --lr 1e-4 \
  --checkpoint-dir checkpoints/hertz

# Step 3: Evaluate against TinyLlama/Pythia baselines
python scripts/evaluate_lm.py --checkpoint checkpoints/hertz/best.pt

# Step 4: Generate text
python scripts/generate.py --checkpoint checkpoints/hertz/best.pt --prompt "The future of artificial intelligence"
```

**Results:** Evaluation script saves benchmark scores to `results/hertz_eval.json`. Checkpoints in `checkpoints/hertz/` (gitignored — too large).

```powershell
# Save evaluation + samples
mkdir results 2>nul
python scripts/generate.py --checkpoint checkpoints/hertz/best.pt --prompt "The" --n-samples 50 > results/hertz_samples.txt

# Commit results
git add results/hertz_eval.json results/hertz_samples.txt
git commit -m "Track B1-1: Hertz 1B evaluation + samples"
git push
```

### Track C: Radiance Klang — Audio Gaussian Splatting

```powershell
# Install audio dependencies
pip install librosa soundfile

# Variant A: STFT point-blob Gaussians (baseline, slow)
python klang/run_stft_experiment.py --device cuda --n_gaussians 1500 3000

# Variant B: Layer-based Gaussians (RECOMMENDED — each Gaussian = a sound layer)
python klang/variant_b_experiment.py --device cuda --n_layers 10 20 40

# Compare output with upper bound:
#   klang/variant_b_*/audio.wav vs klang/diag_2_griffinlim_stft.wav
#   Check variant_b_*/trajectories.png for layer frequency paths
```

Variant B represents each sound source as ONE Gaussian layer with continuous
frequency trajectory and opacity over time. 10-40 layers, not 1000+ blobs.

**Results:** Audio WAVs + plots saved to `klang/variant_b_*/` (reconstruction.png, trajectories.png, opacity.png, audio.wav).

```powershell
# Commit Klang results (audio + plots)
git add klang/variant_b_*/ klang/stft_*.wav
git commit -m "Track C: Klang audio reconstruction results"
git push
```

### Track D1: Radiance Raum — Text-to-3D PoCs

#### PoC-D: Compositional Scene Graph (run first — CPU, ~10 min)

```powershell
python scripts/train_raum_compositional.py --glove data/glove.6B.300d.txt
```

This will:
1. Generate 20K synthetic scenes (sentences + ground truth)
2. Train the compositional model (role classifier + attribute heads)
3. Evaluate on held-out compositional generalization test
4. Print sample scene assemblies

Results → `checkpoints/raum_d/`

#### PoC-C: Shared-Equation Bridge (GPU, ~2-4 hours)

```powershell
python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt
```

This will:
1. Generate synthetic scenes + render GT from multiple viewpoints
2. Train the bridge (semantic Gaussians → spatial Gaussians)
3. Evaluate render quality (PSNR)

Results → `checkpoints/raum_c/`

#### PoC-C Analysis (after training completes)

```powershell
python scripts/analyze_raum_bridge.py ^
  --checkpoint checkpoints/raum_c/best.pt ^
  --glove data/glove.6B.300d.txt
```

This prints:
1. Word → xyz mapping (do spatial words map to spatial axes?)
2. Sentence composition (where do words land in 3D?)
3. Interpolation paths between word pairs
4. Weight matrix analysis of the space transform

Results → `results/raum_c_analysis/`

**Commit all Raum results:**

```powershell
git add results/raum_c_analysis/ checkpoints/raum_d/*.json
git commit -m "Track D: Raum text-to-3D PoC results"
git push
```

---

## Committing Results — Summary

After running any track, commit the results so they're available across machines:

| Track | What to commit | Command |
|---|---|---|
| A (experiments) | `*.json` in repo root | `git add *.json` |
| B1 (Planck) | `results/planck_samples.txt` | `git add results/` |
| B1-1 (Hertz) | `results/hertz_*.json`, `results/hertz_samples.txt` | `git add results/` |
| C (Klang) | `klang/variant_b_*/` (wav + png) | `git add klang/variant_b_*/` |
| D (Raum) | `results/raum_c_analysis/`, checkpoint JSONs | `git add results/` |

**Do NOT commit:** model checkpoints (`.pt` files >50MB), GloVe data, TinyStories data, FineWeb data. These are in `.gitignore`.

---

## 7. Troubleshooting

### CUDA out of memory
Reduce batch size:
```powershell
# Planck
python scripts/train_lm.py --data-dir data/tinystories --batch-size 16

# Raum bridge
python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt --batch-size 2 --img-size 32
```

### gsplat won't install
Skip it. Raum PoC-C auto-falls back to a simple CPU renderer:
```
Rendering backend: simple alpha-composite (CPU)
```
Training is slower but works. Install gsplat later if needed.

### sentencepiece won't install
Only needed for Planck (B1). Raum doesn't need it:
```powershell
pip install sentencepiece
```
If it fails, install via conda: `conda install -c conda-forge sentencepiece`

### GloVe download issues
Alternative mirrors:
- https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
- Or via Python:
```python
import urllib.request
urllib.request.urlretrieve("https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip", "data/glove.6B.zip")
```

### SSL errors on Windows
Some scripts handle this automatically. If you still get SSL errors:
```powershell
pip install certifi
set SSL_CERT_FILE=%VIRTUAL_ENV%\Lib\site-packages\certifi\cacert.pem
```

---

## 8. Quick Smoke Test (2 minutes)

After setup, run this to confirm everything works end-to-end:

```powershell
python -c "
import torch
from src.gaussian import SemanticGaussianVocab
from src.kernel import gaussian_kernel_diag
from src.rendering import render
from src.raum.templates import build_template_library
from src.raum.data import generate_dataset
from src.raum.compositional import RaumCompositional

# Test SGS core
vocab = SemanticGaussianVocab(100, d_s=64, d_f=300)
ids = torch.randint(0, 100, (2, 10))
mu, lv, alpha, feat = vocab.get_params(ids)
q = mu.mean(dim=1)
K = gaussian_kernel_diag(q, mu, lv, torch.tensor(64.0))
meaning, _ = render(feat, alpha, K)
print(f'SGS render: meaning shape = {meaning.shape}')

# Test Raum
templates = build_template_library(n_gaussians=50)
print(f'Templates: {list(templates.keys())}')
scenes = generate_dataset(10)
print(f'Generated {len(scenes)} scenes, first: \"{scenes[0].sentence}\"')
model = RaumCompositional(d_f=300)
print(f'RaumCompositional: {model.count_parameters():,} params')

print('All OK')
"
```

Expected:
```
SGS render: meaning shape = torch.Size([2, 300])
Templates: ['sphere', 'cube', 'cylinder', 'cone', 'plane', 'torus']
Generated 10 scenes, first: "a yellow cube behind a green sphere"
RaumCompositional: 50,033 params
All OK
```
