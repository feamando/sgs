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

Requires Python 3.10+ (3.11 recommended).

```powershell
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA 12.x (RTX 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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

### Track B1: Radiance Planck (100M language model)

```powershell
# Step 1: Download + tokenize TinyStories (~15 min, needs internet)
python src/tinystories.py --data-dir data/tinystories

# Step 2: Train (~6-12 hours on RTX 4090)
python scripts/train_lm.py --data-dir data/tinystories

# Step 3: Generate text
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --prompt "Once upon a time"

# Interactive mode
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --interactive
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
