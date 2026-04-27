# SGS, Full Setup Guide (Windows + RTX 4090)

Fresh install from scratch. Covers all tracks: experiments, Planck LM, Raum text-to-3D, Klang audio, and Hertz 1B.

> **Run order as of 2026-04-20** (Strategic pivot, see `docs/papers/sgs_training_acceleration.md`):
>
> **1. Klang → 2. Planck 1.1 → 3. Raum → 4. Planck 1.2 → 4.5. Klang 1.2 → 5. Hertz 1.2**
>
> Hertz 1.0 is paused. The new large-LM run is Hertz 1.2, with the acceleration recipe from Planck 1.2. This order validates cheaper tracks first, then the accel recipe on Planck 1.2, then commits GPU-weeks to Hertz 1.2.

---

## Table of contents

- §1-5: Install and smoke test (one-time setup)
- §6: Active tracks in run order (Klang → Planck 1.1 → Raum → Planck 1.2 → Klang 1.2 → Hertz 1.2)
- §7: Reference tracks (Track A experiments, Planck 1.0, Hertz 1.0)
- §8: Hertz disk cleanup, before Hertz 1.2 restart
- §9: Troubleshooting
- §10: Quick smoke test (2 minutes)

---

## 1. Clone the Repo

```powershell
cd C:\Users\YourName\Documents\GitHub
git clone https://github.com/feamando/sgs.git
cd sgs
```

---

## 2. Python Environment

Requires **Python 3.11 or 3.12** (NOT 3.13, PyTorch doesn't support 3.13 on Windows yet).

### Install Python 3.12 (if you have 3.13)

1. Download Python 3.12 from https://www.python.org/downloads/release/python-3120/ (Windows installer 64-bit)
2. During install, check **"Add to PATH"**, install alongside 3.13, don't replace it.
3. Use `py -3.12` to target the right version:

```powershell
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

If gsplat fails to install (CUDA build issues on Windows), skip it, the Raum bridge trainer auto-falls back to a CPU-compatible renderer. You can always install it later.

---

## 3. Download GloVe Embeddings

```powershell
mkdir data
cd data
```

Download `glove.6B.zip` (~862 MB) from https://nlp.stanford.edu/data/glove.6B.zip

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
│   ├── sgs_lm.py               # Radiance Planck, causal SGS language model
│   ├── tinystories.py          # TinyStories data pipeline
│   └── raum/                   # Radiance Raum, text-to-3D
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
│   ├── phase0_feasibility.py
│   ├── train_stsb.py
│   ├── run_all_ablations.py
│   ├── run_phase2.py
│   ├── run_phase3.py
│   ├── run_scan_multiseed.py
│   ├── run_paper_fixes.py
│   ├── train_lm.py                     # Planck 1.0
│   ├── train_planck11.py               # Planck 1.1 (H-SGS blobs)
│   ├── build_blobs.py                  # Blob builder for Planck 1.1
│   ├── train_hertz.py                  # Hertz 1B
│   ├── evaluate_lm.py                  # Perplexity + HellaSwag + ARC-Easy
│   ├── generate.py                     # Planck/Hertz text generation
│   ├── export_onnx.py
│   ├── train_raum_compositional.py     # Raum PoC-D
│   ├── train_raum_bridge.py            # Raum PoC-C
│   └── analyze_raum_bridge.py
│
├── klang/                      # Radiance Klang (audio)
├── paper/                      # Research papers (markdown)
├── docs/                       # Plans, analysis, proofs, whitepaper
│   ├── papers/
│   │   └── sgs_training_acceleration.md    # Planck/Hertz 1.2 recipe
│   ├── plans/
│   ├── analysis/
│   └── proofs/lean/            # 13 Lean 4 formal proofs
└── checkpoints/                ← model checkpoints (gitignored)
```

---

## 6. Active tracks, in run order

Follow these in order. Each validates the layer below before committing GPU-weeks.

### 6.1  Track 1: Klang, Audio Gaussian Splatting

**Status:** finalize before Planck 1.1.

```powershell
pip install librosa soundfile

# Variant A: STFT point-blob Gaussians (baseline, slow)
python klang/run_stft_experiment.py --device cuda --n_gaussians 1500 3000

# Variant B: Layer-based Gaussians (RECOMMENDED, each Gaussian = a sound layer)
python klang/variant_b_experiment.py --device cuda --n_layers 10 20 40
```

Variant B represents each sound source as ONE Gaussian layer with continuous frequency trajectory and opacity over time. 10-40 layers, not 1000+ blobs.

**Results:** audio WAVs + plots in `klang/variant_b_*/` (reconstruction.png, trajectories.png, opacity.png, audio.wav). Compare `variant_b_*/audio.wav` to the upper bound `klang/diag_2_griffinlim_stft.wav`.

```powershell
git add klang/variant_b_*/ klang/stft_*.wav
git commit -m "Track 1 Klang: audio reconstruction results"
git push
```

### 6.2  Track 2: Planck 1.1, Hierarchical SGS (Knowledge Splatting / blobs)

**Status:** validates whether the blob concept belongs in Hertz 1.2.
**Prerequisite:** `checkpoints/planck/best.pt` from Planck 1.0 (see §7.2 if missing).

Planck 1.1 extends SGS with pre-computed Gaussian "knowledge blobs" from TinyStories. Two rendering passes: (1) blob pass consumes transmittance with retrieved archetypes, (2) word pass renders token-level in the remaining transmittance. Math formally verified in Lean 4 (H1, H2, H4). Whitepaper: `docs/whitepaper/hierarchical_sgs.md`.

#### Step 1, Build blobs (~15 min)

```powershell
python scripts/build_blobs.py --checkpoint checkpoints/planck/best.pt --n-blobs 50000
```

Output: `data/blobs/tinystories/blobs.pt` (~60 MB).

Inspect:
```powershell
python scripts/inspect_blobs.py
```

#### Step 2, Train Planck 1.1 (two-stage, ~3 hours)

```powershell
# Stage A: blob warmup (base frozen, ~30 min)
python scripts/train_planck11.py --freeze-base --epochs 1

# Stage B: joint training (~2-3 hours)
python scripts/train_planck11.py --resume checkpoints/planck11/epoch_1.pt --epochs 3
```

Or all-in-one:
```powershell
python scripts/train_planck11.py
```

#### Step 3, Ablations (isolate blob contribution)

```powershell
# Blobs disabled (should match Planck 1.0 baseline)
python scripts/train_planck11.py --t-max 0.0 --save-dir checkpoints/planck11_noablob --epochs 3

# Fewer blobs (5K)
python scripts/build_blobs.py --checkpoint checkpoints/planck/best.pt --n-blobs 5000 --output data/blobs/tinystories_5k
python scripts/train_planck11.py --blob-dir data/blobs/tinystories_5k --n-blobs 5000 --save-dir checkpoints/planck11_5k

# Higher blob budget
python scripts/train_planck11.py --t-max 0.5 --save-dir checkpoints/planck11_t50

# More retrieval (top-16 vs default top-8)
python scripts/train_planck11.py --blob-k 16 --save-dir checkpoints/planck11_k16
```

#### Step 4, Hard gates (pass/fail before declaring blobs validated)

Automated runner: `scripts/validate_planck11.py` evaluates all four gates
and writes `results/planck11_validation.json`. Exits non-zero if any
gate fails.

```powershell
# Full run (requires checkpoints/planck11/best.pt and checkpoints/planck11_noablob/best.pt)
python scripts/validate_planck11.py

# Skip gate 1 if you haven't trained the --t-max 0.0 ablation yet
python scripts/validate_planck11.py --skip-gate-1
```

| # | Gate | How the script checks it | Pass condition |
|---|---|---|---|
| 1 | Base generation intact | Eval `checkpoints/planck11_noablob/best.pt` against Planck 1.0 val loss | `|Δ val_loss| ≤ 0.05` |
| 2 | Blobs being used | Mean of `(1 - t_residual) * gate` over eval batches | `> 0.05` |
| 3 | Perplexity improves | Planck 1.1 val loss vs Planck 1.0 | Planck 1.1 strictly lower |
| 4a | Intra-sample repetition | Mean count of repeated 4-grams *within* each generation | Planck 1.1 mean `≤` Planck 1.0 mean |
| 4b | Cross-sample diversity | Unique-4-gram ratio + mean pairwise Jaccard across the 50 samples per model | Informational only, no pass/fail |

Gates 1–4a must pass. If yes, blobs go into Hertz 1.2. If any fail, drop blobs from the 1.2 scope.

Gate 4 note: the original single-number "aggregate 4-gram repeats" conflates two different behaviours. Gate 4a isolates the actual failure mode (looping and copy-paste *inside* a sample). Gate 4b reports the other behaviour (consistency *across* samples of the same prompt) but does not fail the run on it. High cross-sample agreement is desirable for factual, search and code-style outputs and is expected to increase as blobs are added, so penalising it would discard the main upside of H-SGS.

Gate 1 note: the current gate trains an ablation with blobs off (`t_max=0`) and compares it to Planck 1.0. If the fine-tune updates base weights, that baseline will drift even when the model behaves correctly. For a clean Gate 1, pair this with `--freeze-base` at training time (planned for Planck 1.3). Until then, treat a small 4a pass + a strong Gate 3 improvement as the operative signal for "blobs help, base is not broken."

Thresholds live at the top of `scripts/validate_planck11.py` if you need to tune them.

#### Step 5, Commit results

```powershell
mkdir results 2>nul
python scripts/generate.py --checkpoint checkpoints/planck11/best.pt --prompt "Once upon a time" --n-samples 50 | Out-File results/planck11_samples.txt
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --prompt "Once upon a time" --n-samples 50 | Out-File results/planck10_baseline_samples.txt
git add results/planck11_samples.txt results/planck10_baseline_samples.txt
git commit -m "Track 2 Planck 1.1: H-SGS blobs validation"
git push
```

#### Step 6, Free disk before starting Track 3

Planck 1.1 writes per-step and per-epoch checkpoints (~400MB each). Before moving to Raum, sweep the intermediates:

```powershell
# Preview what would be deleted
python scripts/cleanup_planck11.py

# Actually delete
python scripts/cleanup_planck11.py --apply

# Also wipe __pycache__ and compile caches
python scripts/cleanup_planck11.py --apply --include-pycache

# Keep the 2 most recent intermediates per dir (for resume flexibility)
python scripts/cleanup_planck11.py --apply --keep-last 2
```

Defaults keep `best.pt` and `final.pt`; only touches `step_*.pt` / `epoch_*.pt` under `checkpoints/planck11*`. Dry-run unless `--apply` is passed.

### 6.3  Track 3: Raum, Text-to-3D PoCs

**Status:** finalize after Planck 1.1.

#### PoC-D: Compositional Scene Graph (CPU, ~10 min)

```powershell
python scripts/train_raum_compositional.py --glove data/glove.6B.300d.txt
```

Generates 20K synthetic scenes, trains the compositional model (role classifier + attribute heads), evaluates on held-out compositional generalization, prints sample scene assemblies.

Results: `checkpoints/raum_d/`.

> **Note:** `--feature-mode sgs` currently raises NotImplementedError. The SGS encoder integration is pending. Use the default `--feature-mode glove`.

#### PoC-C: Raum 1.0, template-routing bridge (GPU, ~30-60 min)

Raum 1.0 replaces the PoC-C render-loss bridge with a context-aware
transformer that routes each object-role token to an object template
from `src/raum/templates.py` (sphere, cube, cylinder, cone, plane,
torus) and stamps it at a predicted position, colour, and scale.
Training is analytic over the labels already produced by
`src/raum/data.py`, no rendering in the loop.

```powershell
python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt `
  --save-dir checkpoints/raum_11
```

Default architecture: `d_model=128`, 2 transformer layers, 4 heads,
~478K params. Default loss weights: `--lambda-pos 1.0 --lambda-dir 0.5
--lambda-tpl 1.0 --lambda-col 1.0 --lambda-scl 0.5 --lambda-rol 0.5
--pair-margin 0.3`. Default `--batch-size 32`, `--lr 3e-4`,
`--epochs 30`.

Watch these metrics on the step / eval lines:
- `pos` (position MSE) should fall below 0.1
- `tpl` (template argmax accuracy) should climb above 95%
- `dir` (pairwise direction accuracy) should climb above 95%
- `col` (colour MSE) should fall below 0.05

The val line reports a composite `score = tpl_acc + dir_acc - pos_mse`
and saves `best.pt` on each new high. `last.pt` carries
model + optimizer + scheduler + counters for resume.

Results: `checkpoints/raum_11/`.

> **2026-04-27 retrain**: the data generator in `src/raum/data.py` used
> to anchor object 1 at the origin and offset object 2 by the relation,
> which inverted English semantics ("cone above cylinder" put the
> cylinder on top). Fixed to "X rel Y" → Y at origin, X offset.
> Existing checkpoints under `checkpoints/raum_10/` have the old
> convention; retrain into `checkpoints/raum_11/` before running the
> demo.

Background on why this replaces the old bridge: the earlier PoC-C
used a render loss on images from a CPU alpha-compositing renderer.
PSNR pegged at 80 dB (the `log10(1e-8) * -10` clamp floor, not a real
score) because both predicted and GT images came back near-black. On
top of that, the original bridge was per-token and could not
distinguish "first sphere" from "second sphere" in a two-object
scene, so position MSE floored at the variance of the ground-truth
positions. Raum 1.0 fixes both: drop the dead render loss and make
the bridge context-aware via a transformer encoder. See commit
`bff5b75` for the full rewrite.

#### PoC-C analysis

```powershell
python scripts/analyze_raum_bridge.py `
  --checkpoint checkpoints/raum_11/best.pt `
  --glove data/glove.6B.300d.txt
```

Reports:
- Comp-gen test-set metrics (loss, `pos_mse`, `tpl_acc`, `dir_acc`,
  `col_mse`, `scl_mse`, `role_ce`) on held-out object pairs
- Per-sentence probes for a short fixture list of prompts, each
  showing predicted position, template (with confidence), colour,
  and scale per token

Results: `results/raum_11_analysis/report.txt`.

#### Raum demo (local web app)

After analysis passes, run the demo to eyeball the model live. The
backend loads the bridge + template library once; the frontend is
static HTML + Three.js with no build step.

```powershell
pip install -r demo/requirements.txt
python -m demo.app `
  --checkpoint checkpoints/raum_11/best.pt `
  --glove data/glove.6B.300d.txt
```

Then open <http://localhost:8000>. The default prompt auto-renders;
type any scene from the training vocabulary (objects / colours / sizes /
relations in `src/raum/vocab.py`) to regenerate. The side panel lists
each parsed object with its colour swatch, predicted template, and
template confidence; amber outline rings mark predicted coarse means
so misplaced objects are immediately visible.

More in `demo/README.md`. The demo is local-only; nothing is hosted.

#### Pause and resume

Training runs support pause and resume. Every eval step and end of
epoch writes `last.pt` (model + optimizer + scheduler + counters).
`best.pt` stays model-only so `analyze_raum_bridge.py` keeps working
unchanged.

```powershell
# Start
python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt `
  --save-dir checkpoints/raum_11

# Ctrl-C whenever. A final snapshot is written before exit.
# Resume:
python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt `
  --save-dir checkpoints/raum_11 --resume
```

Caveats:
- Use `--ckpt-interval N` for a snapshot every N steps (default is
  eval-step + end-of-epoch only).
- Double Ctrl-C skips the snapshot and kills immediately.
- Resume works only for runs started after commit `6af6842`.

#### Commit

```powershell
git add results/raum_11_analysis/ checkpoints/raum_d/*.json
git commit -m "Track 3 Raum: PoC-C Raum 1.0 + PoC-D results"
git push
```

#### Raum 0.0 polish (2026-04-27)

Round of fixes shipped alongside the retrain. Run the demo against a
fresh `checkpoints/raum_11/best.pt` to see them.

1. **"X rel Y" anchor convention** (`src/raum/data.py`). The data
   generator used to anchor object 1 at the origin and offset object 2
   by the relation, which inverts English ("cone above cylinder" put
   the cylinder on top). Fix: Y is the anchor, X gets the offset.
   Requires retrain into `checkpoints/raum_11/`.
2. **Cylinder seam artifact** (`src/raum/templates.py`). Cylinder caps
   previously sat flush against the side ring, stacking splats into a
   visible line at y = ±height/2. Fix: inset cap radius to 0.92r and
   pull the side's y-range in by 3%.
3. **Unresolved-object path** (`src/raum/bridge.py`, `demo/app.py`,
   `demo/static/`). `assemble_scene` now returns a third list,
   `unresolved`. Below a confidence threshold (default 0.35) the
   argmax template is not stamped; instead the backend emits a
   `warnings` array and the UI shows a coral-bordered row like
   "could not resolve 'xylophone' (top guess torus, conf 22%)".
   Expose via `--template-confidence 0.35` on `demo.app`.
4. **Floating text labels** (`demo/static/viewer.js`,
   `demo/static/style.css`). Ring-sprite markers replaced with HTML
   pills ("cone · 100%") anchored via `Vector3.project(camera)` each
   frame. Readable at any zoom, no z-fighting with splats.
5. **Animated Gaussian-splat logo** (`demo/static/logo.js`). Flat
   amber "R" disc replaced with a 180-point Fibonacci sphere of
   amber-gradient Gaussian blobs rotating on two axes. 2D canvas,
   zero-dep, ~36 px square. Inter is now loaded from rsms.me.
6. **5x template density**. Default `--template-points` raised from
   200 to 1000. Denser templates make individual objects read as
   volumetric clouds instead of sparse scatter.

```powershell
# Retrain with the fixed data generator + polish
python scripts\train_raum_bridge.py --glove data\glove.6B.300d.txt `
  --save-dir checkpoints\raum_11

# Run the updated demo
pip install -r demo\requirements.txt
python -m demo.app `
  --checkpoint checkpoints\raum_11\best.pt `
  --glove data\glove.6B.300d.txt `
  --template-points 1000 `
  --template-confidence 0.35
```

The Raum 0.1 plan (complex scenes, common-object vocabulary, OOV
handling) is in `docs/plans/d1c_raum_01_plan.md` and condensed in the
appendix at the bottom of this file.

### 6.4  Track 4: Planck 1.2, Acceleration recipe validation

**Status:** validates the SGS-native training accelerations from `docs/papers/sgs_training_acceleration.md` before Hertz 1.2.

The paper proposes five accelerations, compound expected ~2-3x:

| # | Proposal | Expected | Effort |
|---|---|---|---|
| §2.1 | Transmittance-weighted loss | ~1.4x sample efficiency | 1 day |
| §2.2 | Adaptive pass count (early exit) | ~1.3x | 2 days |
| §2.3 | Kernel top-k sparsity | ~1.4x forward | 4 days |
| §2.4 | Shared kernel across passes | ~1.05x | 1 day |
| §2.5 | Gaussian-native curriculum | ~1.15x (Hertz-2 material) | 2 days |

Recommended validation order on Planck 1.2:

1. **Validate prerequisite:** check correlation between `T[t,t]` and prediction correctness on current Hertz checkpoint. If correlation is strong, §2.1 is well-founded.
2. **§2.1 first** (cheap, doesn't change compute shape). A/B: `--transmittance-loss` vs plain CE on Planck 1.2, 500M tokens. Pass if perplexity parity at ≥20% fewer tokens.
3. **§2.3 second** (biggest compute win, orthogonal to §2.1). Measure kernel + render wall-clock delta.
4. **§2.2 and §2.4** bolt on once the pipeline is stable.
5. **§2.5** deferred to Hertz 2.x.

Gate before Hertz 1.2: Planck 1.2 must reach Planck 1.0's validation loss on ≤70% of the tokens (1.43x sample efficiency). If not, revisit the paper or drop individual proposals.

### 6.4.5  Track 4.5: Klang 1.2 revisit (after Planck 1.2, before Hertz 1.2)

**Status:** optional quality pass. Klang 1.1 shipped two decoded variants that are "comprehensible but artefacted" (Variant A: phase warble at 3000g; Variant B: sub-200 Hz dropout + near-Nyquist whine, identical across 10/20/40L). Klang 1.2 addresses both directly and adds quantitative metrics, so we can make a real decision about Klang's status.

Why it runs here and not earlier: the architectural fixes (complex-valued Gaussians, transmittance compositing) benefit from the same SGS theorem work that drives Planck 1.2, and the validator is modeled on `validate_planck11.py`. Running after 1.2 keeps the mental model consistent across tracks.

**Changes in Klang 1.2 (see `docs/papers/sgs_training_acceleration.md` for the underlying theorem and `src/klang/scene.py` for the model):**

| # | Fix | Target artefact |
|---|---|---|
| 1 | Complex-valued Gaussians (A·e^(iφ)) | Phase warble (Variant A) |
| 2 | Mel-scaled init + widened σ/f₀ bounds (~40 Hz to Nyquist + σ cap) | Bass dropout + near-Nyquist whine (Variant B) |
| 3 | Multi-resolution STFT loss (512/1024/2048) | Single-window over-fit |
| 4 | Transmittance-budgeted alpha compositing (SGS theorem) | Connects Klang to Planck/Hertz math |
| 5 | Mel + HiFi-GAN decode bridge | A/B decoder quality vs splat quality |
| 6 | Optional perceptual loss (VGGish/CLAP stub) | Feature-space quality |
| 7 | `scripts/validate_klang.py` with spectral MSE / log-MAE / MCD-13 / optional PESQ/STOI | Quantitative gates |

```powershell
# Train Klang 1.2 on the same test clip used by Variant B
python klang/klang_1_2_experiment.py --audio klang/test_clip.wav --n-layers 20 --n-steps 3000 --device cuda

# Ablations (optional, each ~10 min on 4090)
python klang/klang_1_2_experiment.py --no-complex --out-dir klang/klang_1_2_nocomplex
python klang/klang_1_2_experiment.py --compositing sum --out-dir klang/klang_1_2_sum
python klang/klang_1_2_experiment.py --no-mrstft --out-dir klang/klang_1_2_nomrstft

# Validate against original + Variant B baseline
python scripts/validate_klang.py ^
  --ref klang/original.wav ^
  --reference-for-gates klang/variant_b_20L/audio.wav
```

Gates (thresholds at the top of `scripts/validate_klang.py`):
- **Gate A:** Klang 1.2 spectral MSE vs `original.wav` below the absolute ceiling (`GATE_A_MSE_CEIL`).
- **Gate B:** Klang 1.2 log-mag MAE strictly lower than Variant B's.

Outputs in `klang/klang_1_2/`: `scene.pt`, `decode_istft.wav`, `decode_griffinlim.wav`, optional `decode_hifigan.wav`, plus `reconstruction.png`, `trajectories.png`, `loss.png`. Validator writes `results/klang_validation.json`.

```powershell
git add klang/klang_1_2/ results/klang_validation.json
git commit -m "Track 4.5 Klang 1.2: complex + widened + MRSTFT + metrics"
git push
```

If both gates pass, Klang is done and we ship it alongside Hertz 1.2. If not, roll back to Klang 1.1 results and document what 1.2 did vs didn't fix. Either way, cost is ~1-2 hours of GPU time — small enough to be worth running before Hertz burns GPU-weeks.

### 6.5  Track 5: Hertz 1.2, 1B SGS LM with accel recipe

**Status:** runs only after Planck 1.2 validates §2.1 and §2.3 (minimum).
**Prerequisite:** Hertz folder cleaned up (see §8).

Command shape (subject to flags added during Planck 1.2 work):

```powershell
python scripts/train_hertz.py ^
  --max-tokens 10B ^
  --no-compile ^
  --keep-last 3 ^
  --transmittance-loss ^
  --kernel-topk 64 ^
  --wandb
```

At a Planck-1.2-validated compound ~2x throughput on top of 11.8k tok/s baseline, 10B tokens should complete in ~5 days on the 4090, vs ~10 days without accel.

Evaluation:
```powershell
python scripts/evaluate_lm.py --checkpoint checkpoints/hertz/best.pt
python scripts/evaluate_lm.py --checkpoint checkpoints/hertz/best.pt --limit 5   # smoke test
python scripts/generate.py --checkpoint checkpoints/hertz/best.pt --prompt "The future of AI"
```

Commit:
```powershell
mkdir results 2>nul
python scripts/generate.py --checkpoint checkpoints/hertz/best.pt --prompt "The" --n-samples 50 > results/hertz_samples.txt
git add results/hertz_eval.json results/hertz_samples.txt
git commit -m "Track 5 Hertz 1.2: 1B SGS LM evaluation + samples"
git push
```

---

## 7. Reference tracks (already run / on hold)

### 7.1  Track A: Prior experiments (Phase 0-3, STS-B, SCAN, NLI)

Results live in repo root as `*.json`. Re-run for verification:

```powershell
python scripts/phase0_feasibility.py --glove data/glove.6B.300d.txt
python scripts/run_all_ablations.py --glove data/glove.6B.300d.txt
python scripts/run_phase2.py --glove data/glove.6B.300d.txt
python scripts/run_phase3.py --glove data/glove.6B.300d.txt
python scripts/run_scan_multiseed.py --glove data/glove.6B.300d.txt
python scripts/run_paper_fixes.py --glove data/glove.6B.300d.txt
```

### 7.2  Planck 1.0, 100M baseline LM

Needed as a prerequisite for Planck 1.1 blobs. Skip if `checkpoints/planck/best.pt` already exists.

```powershell
python src/tinystories.py --data-dir data/tinystories          # ~15 min
python scripts/train_lm.py --data-dir data/tinystories          # ~2-3 hours

python scripts/generate.py --checkpoint checkpoints/planck/best.pt --prompt "Once upon a time"
python scripts/generate.py --checkpoint checkpoints/planck/best.pt --interactive
```

### 7.3  Hertz 1.0, **PAUSED**

Hertz 1.0 training was paused 2026-04-20 due to wall-clock infeasibility without the SGS-native accel recipe. Do not restart Hertz 1.0. The next 1B-scale run is Hertz 1.2 (§6.5) which requires Planck 1.2 validation first.

If you want to *reproduce* Hertz 1.0 for comparison purposes only, see the pre-2026-04-20 instructions via `git log` on this file.

---

## 8. Hertz disk cleanup, before Hertz 1.2 restart

Before starting Hertz 1.2 you want a clean `checkpoints/hertz/` and enough free disk for the new run. What the old run leaves behind and what to keep:

### 8.1  What's on disk

| Path | Typical size | Safe to delete? |
|---|---|---|
| `checkpoints/hertz/best.pt` | ~8 GB | **Yes**, Hertz 1.2 is a fresh architecture |
| `checkpoints/hertz/step_*.pt` | ~8 GB each, up to hundreds of GB over a full run | **Yes** |
| `checkpoints/hertz/epoch_*.pt` | ~8 GB each | **Yes** |
| `checkpoints/hertz/final.pt` | ~8 GB | **Yes** |
| `checkpoints/hertz/trace.json` | varies | **Yes** (profiler artifact) |
| `data/fineweb/train.bin` | ~18 GB (9B tokens × 2 bytes) | **No, keep**, reused by Hertz 1.2 |
| `data/fineweb/val.bin` | ~180 MB | **No, keep** |
| `data/fineweb/tokenizer.model` | ~2 MB | **No, keep** |

Checkpoints from Hertz 1.0 are not architecturally portable to Hertz 1.2, delete them. The FineWeb-Edu tokenized data is reusable and costly to re-download/tokenize (~hours), keep it.

### 8.2  Cleanup commands (PowerShell)

Show what will be deleted, then delete:

```powershell
# Inventory first (no deletion)
Get-ChildItem checkpoints\hertz\*.pt | Select-Object Name, @{N='GB';E={[math]::Round($_.Length/1GB, 2)}}
Get-ChildItem checkpoints\hertz\trace*.json | Select-Object Name, Length

# Delete all checkpoints (KEEP data/fineweb)
Remove-Item checkpoints\hertz\*.pt
Remove-Item checkpoints\hertz\trace*.json -ErrorAction SilentlyContinue

# Verify data is intact
Get-ChildItem data\fineweb\ | Select-Object Name, @{N='GB';E={[math]::Round($_.Length/1GB, 2)}}
```

Expected disk savings: ~10-400 GB depending on how many `step_*.pt` accumulated.

### 8.3  Retention during Hertz 1.2

`train_hertz.py` now supports `--keep-last N` which rotates `step_*.pt` files, keeping only the N most recent. `best.pt`, `epoch_*.pt`, `final.pt` are never rotated.

**Default is `--keep-last 3`** (~24 GB of rotating step checkpoints, down from unbounded). Pass `--keep-last 0` to disable rotation (not recommended for long runs).

```powershell
# Default retention (3 most recent step_*.pt)
python scripts/train_hertz.py --max-tokens 10B

# Aggressive retention (only 1 step_*.pt, ~8 GB)
python scripts/train_hertz.py --max-tokens 10B --keep-last 1

# Legacy behavior (keep everything, dangerous)
python scripts/train_hertz.py --max-tokens 10B --keep-last 0
```

### 8.4  Also worth clearing

```powershell
# PyTorch compile cache (regenerates on next run)
Remove-Item -Recurse "$env:LOCALAPPDATA\torch\inductor" -ErrorAction SilentlyContinue
Remove-Item -Recurse "$env:USERPROFILE\.cache\torch_inductor" -ErrorAction SilentlyContinue

# Wandb local runs folder
Remove-Item -Recurse wandb\ -ErrorAction SilentlyContinue

# HuggingFace datasets cache (re-downloads if you need them again)
# Remove-Item -Recurse "$env:USERPROFILE\.cache\huggingface\datasets"
```

HuggingFace cache can be several tens of GB. Keep it unless you're desperate for space, redownloading FineWeb-Edu is slow.

---

## 9. Troubleshooting

### CUDA out of memory
```powershell
python scripts/train_lm.py --data-dir data/tinystories --batch-size 16
python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt --batch-size 2 --img-size 32
```

### Hertz throughput collapses on resume
`--resume` can trigger a throughput regression from fused-Adam fallback after loading optimizer state on Windows. Two workarounds:

```powershell
# Option A: weights-only resume (drops optimizer state, LR re-warms)
python scripts/train_hertz.py --warm-start checkpoints/hertz/best.pt --no-compile

# Option B: start fresh (best throughput, loses prior tokens)
python scripts/train_hertz.py --no-compile
```

### Hertz compile is slow on Windows
`torch.compile` via `triton-windows` can be *slower* than eager mode on a 4090 under Windows. Use `--no-compile`.

### gsplat won't install
Skip it. Raum PoC-C auto-falls back to a simple CPU renderer (output will show `Rendering backend: simple alpha-composite (CPU)`). Training is slower but works.

### sentencepiece won't install
Only needed for Planck + Hertz. If pip fails: `conda install -c conda-forge sentencepiece`

### GloVe download issues
Alternative mirror: https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip

Or via Python:
```python
import urllib.request
urllib.request.urlretrieve("https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip", "data/glove.6B.zip")
```

### SSL errors on Windows
```powershell
pip install certifi
set SSL_CERT_FILE=%VIRTUAL_ENV%\Lib\site-packages\certifi\cacert.pem
```

---

## 10. Quick smoke test (2 minutes)

After setup, confirm everything works end-to-end:

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

---

## Appendix: Committing results, summary

| Track | What to commit | Command |
|---|---|---|
| A (experiments) | `*.json` in repo root | `git add *.json` |
| 1 (Klang) | `klang/variant_b_*/` (wav + png) | `git add klang/variant_b_*/` |
| 2 (Planck 1.1) | `results/planck11_*.txt`, `results/planck10_baseline*.txt` | `git add results/` |
| 3 (Raum) | `results/raum_c_analysis/`, checkpoint JSONs | `git add results/` |
| 4 (Planck 1.2) | `results/planck12_*.txt`, ablation JSONs | `git add results/` |
| 4.5 (Klang 1.2) | `klang/klang_1_2/`, `results/klang_validation.json` | `git add klang/klang_1_2/ results/` |
| 5 (Hertz 1.2) | `results/hertz_eval.json`, `results/hertz_samples.txt` | `git add results/` |

**Do NOT commit:** model checkpoints (`.pt` files >50 MB), GloVe data, TinyStories data, FineWeb data, blob indices (~60 MB). These are in `.gitignore`.

---

## Appendix: Raum 0.1 plan (condensed)

Full plan: `docs/plans/d1c_raum_01_plan.md`.

Raum 0.0 proved text → template-routing works on 2-object scenes with the 6-shape library. Raum 0.1 fixes three limits.

### 1. Complex scenes from prompt context
- **Goal**: N-object scenes (N in {1,2,3}) with longer sentences and multi-anchor relations.
- **Changes**:
  - Data generator: emit 1/2/3-object scenes; anchor pointer per relation so "C right of B" uses B as anchor (not slot-0).
  - Training: extend pairwise-direction loss to all consecutive object pairs; DETR-style Hungarian matching deferred to 0.2.
  - Add per-pair relation-graph head to classify relation between any two object tokens; supervise from generator labels.
- **Success metric**: >90% direction accuracy on 3-object val set.

### 2. Train with common objects
- **Goal**: expand template library from 6 primitives to ~30 everyday objects (`car`, `tree`, `chair`, `cup`, ...).
- **Approach** (sharp version): procedural low-poly templates authored from primitive composites. Each new object is a `GaussianTemplate` built from cuboids + cylinders + spheres. ~30 entries in 1-2 days.
- **Approach** (harder): ShapeNet canonical mesh → Gaussian cloud per category. Better visuals, slower setup.
- **Bridge change**: just bump `n_templates`. Architecture is agnostic.
- Tradeoff: procedural looks crude up close; ShapeNet is demo-ready but needs mesh pipeline.

### 3. OOV object handling
- **Current (0.0)**: low-confidence template → unresolved warning in UI, nothing rendered.
- **0.1 (ship)**: nearest-neighbour over template embeddings. If GloVe(`xylophone`) is close enough to `GloVe(cube)` by cosine, stamp cube with a "closest guess" label; else keep the warning.
- **0.2 (research)**: conditional Gaussian-blob decoder driven by the word's feature vector. Unknown objects render as amorphous blobs whose shape is learned — this plays directly into the blob-plugin thesis in the v5 pitch.
- **Alt**: local LM router for OOV → known template. Fastest to ship, not SGS-native.

### Rollout order
1. **0.1.0**: 3-object data generator + analytic labels; analyzer verifies 3-object direction accuracy.
2. **0.1.1**: procedural 30-object library; retrain bridge; demo shows new names.
3. **0.1.2**: NN-cosine OOV policy with "closest guess" copy.
4. **0.2**: Hungarian matching, nested relations, conditional-blob OOV decoder (own plan doc).

### Open questions
- Shared embeddings between Raum and Planck (blob-plugin thesis says yes; Raum-only regime says no).
- Global vs per-template confidence threshold (plane/torus were 0.0 argmax attractors).
- "template" vs "blob class" naming once library > 30.

