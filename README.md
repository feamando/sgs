# SGS Phase 1 Experiment

Semantic Gaussian Splatting — Kill Gate Experiment

Tests whether alpha-compositing of Gaussian word representations produces better sentence embeddings than simple mean-pooling.

## Setup (Windows + RTX 4090)

### 1. Clone / copy this directory to your Windows machine

### 2. Install Python environment

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch with CUDA 12.x (for RTX 4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install -r requirements.txt
```

### 3. Download GloVe embeddings

```bash
mkdir data
cd data
# Download glove.6B.zip (~862MB) from:
# https://nlp.stanford.edu/data/glove.6B.zip
# Extract glove.6B.300d.txt into data/
cd ..
```

After extraction, you should have: `data/glove.6B.300d.txt` (~1GB)

### 4. Verify setup

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"
```

Expected: `CUDA: True, GPU: NVIDIA GeForce RTX 4090`

## Running the Experiment

### Phase 0: Numerical Feasibility (10 minutes)

```bash
python scripts/phase0_feasibility.py --glove data/glove.6B.300d.txt
```

This checks:
- Do Gaussian kernels produce non-degenerate values at d=64?
- What temperature τ works best?
- Is the kernel sparse (most evaluations negligible)?
- Do nearest-neighbor results make semantic sense?

**If PASS → continue. If FAIL → reduce d_s to 32.**

### Phase 1b: Train SGS (1 hour)

```bash
# Full SGS model (4 passes)
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 4
```

### Phase 1c: Run All Ablations (8-10 hours)

```bash
python scripts/run_all_ablations.py --glove data/glove.6B.300d.txt
```

This runs 9 model variants and produces a comparison table.

### Individual ablations (if you prefer)

```bash
# Baselines
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model mean_pool
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model mean_pool_mu
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model softmax_attn

# SGS variants
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 1
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 2
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model sgs --n_passes 8
python scripts/train_stsb.py --glove data/glove.6B.300d.txt --model no_transmittance
```

## Kill Gate

| Result | Action |
|---|---|
| Test Spearman ≥ 0.78 | **PASS** → Proceed to Phase 2 |
| 0.58 ≤ Spearman < 0.78 | **INVESTIGATE** → Which ablation helps? |
| Spearman < 0.58 | **KILL** → Pivot to Gaussian Transformer hybrid |

The critical comparison: **SGS-full vs softmax_attn**
- If SGS wins → rendering equation is validated
- If softmax wins → Gaussians help but rendering doesn't → hybrid

## Project Structure

```
sgs-experiment/
├── README.md
├── requirements.txt
├── config/default.yaml
├── data/                      # GloVe embeddings (gitignored)
│   └── glove.6B.300d.txt
├── src/
│   ├── __init__.py
│   ├── gaussian.py            # Semantic Gaussian primitive (A1)
│   ├── kernel.py              # Gaussian kernel (A2)
│   ├── rendering.py           # Rendering equation (A3) + baselines
│   ├── model.py               # Full SGS + ablation models
│   └── data.py                # GloVe + STS-B loading
└── scripts/
    ├── phase0_feasibility.py  # Numerical check
    ├── train_stsb.py          # Main training
    └── run_all_ablations.py   # All ablations
```
