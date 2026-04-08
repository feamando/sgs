# Phase 1 Experiment Plan: Gaussian Composition Kill Gate

**Objective:** Determine if alpha-compositing of Gaussian word representations produces better sentence embeddings than simple mean-pooling.
**Hardware:** AMD Ryzen 9, 64GB RAM, NVIDIA RTX 4090 (24GB VRAM)
**Timeline:** 2-4 weeks
**Kill condition:** If rendering ≤ mean-pooling on STS-B, SGS is dead. Pivot to hybrid.

---

## Experiment Overview

```
Phase 0 (Week 1):     Numerical feasibility — does the kernel work at d=64?
Phase 1a (Week 1-2):  Implement SGS rendering in PyTorch
Phase 1b (Week 2-3):  Train on STS-B, compare to baselines
Phase 1c (Week 3-4):  Ablations — what contributes? Multi-pass? Transmittance? Covariance?
```

---

## Prerequisites & Setup

### Software Stack

```
Python 3.11+
PyTorch 2.x (CUDA 12.x for 4090)
sentence-transformers (for STS-B data loading + baselines)
datasets (HuggingFace — for STS-B)
numpy, scipy
wandb (experiment tracking)
```

### Data

**STS-B (Semantic Textual Similarity Benchmark)**
- ~8,628 sentence pairs with human similarity scores (0-5)
- Train: 5,749 / Dev: 1,500 / Test: 1,379
- Metric: Spearman correlation between predicted and human scores
- Available via HuggingFace: `datasets.load_dataset("glue", "stsb")`

**GloVe Embeddings (initialization)**
- GloVe 6B 300d (400K vocab, ~1GB download)
- Source: https://nlp.stanford.edu/data/glove.6B.zip
- Used to initialize Gaussian means and features

### Baseline Targets

| Method | STS-B Spearman | Source |
|---|---|---|
| GloVe mean-pooling | ~0.58 | Standard baseline |
| GloVe + IDF weighting | ~0.72 | Standard baseline |
| SIF (Smooth Inverse Frequency) | ~0.78 | Arora et al., 2017 |
| InferSent | ~0.84 | Conneau et al., 2017 |
| BERT-base (fine-tuned) | ~0.87 | Devlin et al., 2019 |

**SGS must beat 0.78 (SIF) to pass. Must beat 0.58 (mean-pooling) to show the rendering equation adds value.**

---

## Phase 0: Numerical Feasibility (Days 1-3)

### What to Test

Before building the full system, verify that the Gaussian kernel produces usable values at d_s=64.

### Experiment

```python
# pseudocode
1. Load GloVe 300d embeddings for top 10K words
2. PCA → d_s=64 (splatting space)
3. Initialize covariances: Σ = σ² · I where σ² = 1/freq(word)
4. Pick 1000 random query points
5. For each query, evaluate K(q, μ_i, Σ_i) for all 10K Gaussians
6. Measure:
   a. Distribution of kernel values (should NOT all be ~0 or ~1)
   b. Effective sparsity: what % of evaluations < 1e-3?
   c. Gradient magnitudes: are ∂K/∂μ meaningful?
7. Sweep temperature τ ∈ {16, 32, 64, 128, 256}
```

### Success Criteria

- At τ_optimal: kernel values span [0.01, 0.9] for nearby Gaussians
- Sparsity > 80% (most evaluations negligible)
- Gradients non-zero and not exploding

### Failure Actions

- If all values ≈ 0: increase τ or reduce d_s to 32
- If all values ≈ 1: decrease τ or increase d_s
- If no τ works: try inverse-quadratic kernel K = 1/(1 + D_M/τ) instead of Gaussian

---

## Phase 1a: Implementation (Days 4-10)

### Code Structure

```
sgs-experiment/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # Hyperparameters
├── data/
│   └── glove/                 # GloVe embeddings (gitignored)
├── src/
│   ├── __init__.py
│   ├── gaussian.py            # Semantic Gaussian primitive
│   ├── kernel.py              # Kernel evaluation (A2)
│   ├── rendering.py           # Rendering equation (A3)
│   ├── multipass.py           # Multi-pass rendering (A5)
│   ├── model.py               # Full SGS encoder
│   ├── baselines.py           # Mean-pool, IDF, SIF baselines
│   └── data.py                # STS-B data loading
├── scripts/
│   ├── phase0_feasibility.py  # Numerical feasibility check
│   ├── train_stsb.py          # Main training script
│   └── evaluate.py            # Evaluation + ablations
└── notebooks/
    └── analysis.ipynb         # Visualization + results
```

### Core Components to Implement

**1. SemanticGaussian (gaussian.py)**

```python
class SemanticGaussian(nn.Module):
    """Vocabulary of Gaussian primitives."""
    def __init__(self, vocab_size, d_s=64, d_f=300):
        self.mu = nn.Parameter(...)        # [vocab_size, d_s] — means
        self.log_diag_L = nn.Parameter(...)  # [vocab_size, d_s] — log diagonal of Cholesky
        self.raw_alpha = nn.Parameter(...)  # [vocab_size] — pre-sigmoid opacity
        self.features = nn.Parameter(...)   # [vocab_size, d_f] — feature vectors

    def get_covariance(self, idx):
        """Returns diagonal covariance (start simple, upgrade to full later)."""
        diag = torch.exp(self.log_diag_L[idx]) ** 2  # ensure positive
        return diag  # [batch, d_s] — diagonal only for v1

    def get_alpha(self, idx):
        return torch.sigmoid(self.raw_alpha[idx])
```

**Design decision: Start with DIAGONAL covariance, not full Cholesky.** Full Cholesky has 2,080 params per Gaussian — overkill for Phase 1. Diagonal has 64 params. If diagonal works, upgrade to full later.

**2. GaussianKernel (kernel.py)**

```python
def gaussian_kernel(q, mu, diag_cov, tau):
    """
    q: [batch, d_s] — query points
    mu: [batch, n_tokens, d_s] — Gaussian means
    diag_cov: [batch, n_tokens, d_s] — diagonal covariance
    tau: scalar — temperature

    Returns: [batch, n_tokens] — kernel values
    """
    diff = q.unsqueeze(1) - mu  # [batch, n_tokens, d_s]
    inv_cov = 1.0 / (diag_cov + 1e-6)  # [batch, n_tokens, d_s]
    mahal = (diff * inv_cov * diff).sum(-1)  # [batch, n_tokens]
    return torch.exp(-0.5 * mahal / tau)
```

**3. RenderingEquation (rendering.py)**

```python
def render(features, alpha, kernel_vals):
    """
    features: [batch, n_tokens, d_f]
    alpha: [batch, n_tokens]
    kernel_vals: [batch, n_tokens]

    Returns: [batch, d_f] — rendered meaning vector
    """
    # Effective opacity
    eff_opacity = alpha * kernel_vals  # [batch, n_tokens]

    # Transmittance (cumulative product, front-to-back)
    one_minus_a = 1.0 - eff_opacity  # [batch, n_tokens]
    # T_i = prod_{j<i} (1 - a_j)
    # Use cumprod with exclusive=True
    log_transmittance = torch.cumsum(
        torch.log(one_minus_a + 1e-8), dim=1
    )
    # Shift right: T_1 = 1, T_i = exp(sum of log(1-a_j) for j<i)
    transmittance = torch.exp(
        torch.cat([
            torch.zeros_like(log_transmittance[:, :1]),
            log_transmittance[:, :-1]
        ], dim=1)
    )

    # Blending weights
    weights = eff_opacity * transmittance  # [batch, n_tokens]

    # Weighted sum of features
    meaning = (weights.unsqueeze(-1) * features).sum(dim=1)  # [batch, d_f]

    return meaning, weights
```

**4. SGSEncoder (model.py)**

```python
class SGSEncoder(nn.Module):
    def __init__(self, vocab_size, d_s=64, d_f=300, n_passes=4, tau_init=64.0):
        self.gaussians = SemanticGaussian(vocab_size, d_s, d_f)
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.n_passes = n_passes

        # Per-pass update MLPs
        self.mu_update = nn.ModuleList([
            nn.Sequential(nn.Linear(d_f*2, d_s), nn.Tanh())
            for _ in range(n_passes - 1)
        ])
        self.alpha_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(d_f*2, 1), nn.Sigmoid())
            for _ in range(n_passes - 1)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_f*2, d_f*2), nn.GELU(),
                nn.Linear(d_f*2, d_f)
            )
            for _ in range(n_passes - 1)
        ])

    def forward(self, token_ids):
        # Activate Gaussians
        mu = self.gaussians.mu[token_ids]        # [batch, seq, d_s]
        cov = self.gaussians.get_covariance(token_ids)
        alpha = self.gaussians.get_alpha(token_ids)
        features = self.gaussians.features[token_ids]

        for p in range(self.n_passes):
            # Query = mean of all means (simple; upgrade to learned query later)
            query = mu.mean(dim=1)  # [batch, d_s]

            # Evaluate kernel
            K = gaussian_kernel(query, mu, cov, self.tau)

            # Render
            meaning, weights = render(features, alpha, K)

            # Update parameters (except first pass)
            if p < self.n_passes - 1:
                # Context for each token: render at its own position
                K_self = gaussian_kernel(
                    mu.view(-1, mu.size(-1)),  # each token's position as query
                    mu.repeat_interleave(mu.size(1), dim=0),
                    cov.repeat_interleave(mu.size(1), dim=0),
                    self.tau
                )  # ... simplified; actual impl uses batched per-token rendering

                context = meaning.unsqueeze(1).expand_as(features)
                combined = torch.cat([features, context], dim=-1)

                mu = mu + self.mu_update[p](combined)
                alpha = alpha * self.alpha_gate[p](combined).squeeze(-1)
                features = features + self.ffn[p](combined)

        return meaning  # [batch, d_f]
```

**5. Baselines (baselines.py)**

```python
class MeanPoolBaseline(nn.Module):
    """Mean of Gaussian means — no rendering equation."""
    def forward(self, gaussians, token_ids):
        return gaussians.mu[token_ids].mean(dim=1)

class MeanPoolFeatures(nn.Module):
    """Mean of Gaussian features — no rendering equation."""
    def forward(self, gaussians, token_ids):
        return gaussians.features[token_ids].mean(dim=1)

class WeightedMeanBaseline(nn.Module):
    """IDF-weighted mean of features."""
    # ...

class SIFBaseline(nn.Module):
    """Smooth Inverse Frequency (Arora et al., 2017)."""
    # ...
```

---

## Phase 1b: Training (Days 10-18)

### Training Setup

**Task:** Predict similarity score (0-5) for sentence pairs.

**Architecture:**
```
Sentence A → SGSEncoder → meaning_A (d_f-dim vector)
Sentence B → SGSEncoder → meaning_B (d_f-dim vector)
similarity = cosine(meaning_A, meaning_B) × 5.0
loss = MSE(similarity, human_score)
```

**Hyperparameters:**

| Parameter | Value | Notes |
|---|---|---|
| d_s (splatting) | 64 | From PCA of GloVe 300d |
| d_f (features) | 300 | Original GloVe dimensions |
| Vocab size | 50,000 | Top-50K GloVe words |
| n_passes | 4 | Start with 4; ablate 1, 2, 4, 8 |
| τ (temperature) | 64.0 (learned) | Initialized at d_s |
| Covariance | Diagonal | Full Cholesky in Phase 2 if Phase 1 passes |
| Batch size | 64 | Fits comfortably in 4090 24GB |
| Learning rate | 1e-3 (AdamW) | With cosine schedule |
| Epochs | 50 | STS-B is small; will converge fast |
| Weight decay | 1e-4 | |

**Initialization:**
1. `mu`: GloVe 300d → PCA to 64d
2. `features`: Original GloVe 300d vectors
3. `log_diag_L`: Initialize from word frequency — rare words get larger variance
4. `raw_alpha`: Initialize to sigmoid⁻¹(0.5) = 0 for all words (equal salience)

**Training time estimate:** STS-B is small (~5.7K train pairs). With 4090, expect ~1 min/epoch. Full training: ~1 hour. Hyperparameter sweeps: ~12 hours total.

### What to Track (wandb)

```
- train_loss, val_loss
- val_spearman (THE metric)
- learned_tau (temperature evolution)
- mean/std of kernel values per epoch
- sparsity (% of kernel vals < 1e-3)
- mean/std of alpha values
- gradient norms per parameter group (mu, L, alpha, features)
- weight distribution (are transmittance effects visible?)
```

---

## Phase 1c: Ablations (Days 18-25)

### Critical Ablations

Each tests whether a specific component of SGS contributes:

| Ablation | What's Removed | Tests |
|---|---|---|
| **SGS-full** | Nothing (full model) | Baseline SGS |
| **no-transmittance** | Set T_i = 1 for all i (remove occlusion) | Does transmittance help? |
| **no-kernel** | Set K_i = 1 for all i (uniform weighting) | Does the Gaussian kernel help? |
| **no-multipass** | P = 1 (single rendering pass) | Does iterative refinement help? |
| **mean-pool-mu** | Replace rendering with mean of μ vectors | Does rendering beat averaging means? |
| **mean-pool-features** | Replace rendering with mean of feature vectors | Does rendering beat averaging features? |
| **softmax-attention** | Replace rendering weights with softmax(μ·q) | Does rendering beat attention? |
| **random-order** | Randomize token order each forward pass | Does sequence ordering matter? |
| **no-covariance** | Set all Σ = I (isotropic) | Does learned covariance help? |
| **SIF-baseline** | SIF (no learning) | External baseline |

### Expected Results (Hypotheses)

| Comparison | Expected Winner | Why |
|---|---|---|
| SGS-full vs mean-pool | SGS-full | Rendering adds compositional structure |
| SGS-full vs no-transmittance | SGS-full | Transmittance captures salience/occlusion |
| SGS-full vs no-kernel | SGS-full | Kernel provides locality |
| SGS-full vs no-multipass | SGS-full (modest) | Multi-pass helps disambiguation |
| SGS-full vs softmax-attention | Unclear | This is the key comparison! |
| SGS-full vs random-order | SGS-full | Order matters for composition |

**The softmax-attention ablation is critical.** If softmax-attention beats SGS-full, then Gaussians help but the rendering equation doesn't — pivot to Gaussian Transformer (hybrid A1).

---

## Phase 0+1 Combined: Minimal Viable Experiment

If you want the fastest possible validation before building the full system:

### Day 1-2: Ultrafast Feasibility

```python
# Literally just this:
1. Load GloVe, PCA to 64d
2. For 100 STS-B sentence pairs:
   a. Get word embeddings for both sentences
   b. Compute meaning_A via rendering equation (forward only, no learning)
   c. Compute meaning_B the same way
   d. cosine(meaning_A, meaning_B) vs human score
3. Compare Spearman for:
   - Rendering (with random init alpha, tau=64)
   - Mean pooling
   - IDF-weighted mean
```

This takes ~30 minutes to code and run. If rendering with RANDOM parameters already correlates with human similarity, the approach has legs. If it's pure noise, we know early.

---

## Decision Gates

| After | Metric | Decision |
|---|---|---|
| Phase 0 | Kernel values non-degenerate | CONTINUE or reduce d_s |
| Phase 1b (epoch 10) | val_spearman trending upward | CONTINUE or debug |
| Phase 1b (final) | val_spearman ≥ 0.78 | **PASS** → Phase 2 |
| Phase 1b (final) | val_spearman < 0.58 | **KILL** → Pivot to hybrid |
| Phase 1b (final) | 0.58 ≤ val_spearman < 0.78 | **INVESTIGATE** → which ablation helps? |
| Phase 1c | SGS-full > softmax-attention | Rendering equation is validated |
| Phase 1c | SGS-full ≤ softmax-attention | Rendering loses to attention → hybrid |

---

## Hardware Utilization

**RTX 4090 (24GB VRAM):**
- SGS model with 50K Gaussians, d_s=64, d_f=300, diagonal covariance: ~80MB model size
- Batch of 64 sentence pairs, max 50 tokens each: ~200MB activations
- Total VRAM: ~1-2GB — the 4090 is massively overprovisioned for this
- You could run 8 experiments in parallel if needed

**Ryzen 9 + 64GB RAM:**
- GloVe loading + PCA: ~4GB RAM, 30 seconds
- Data preprocessing: trivial
- Hyperparameter sweeps via multiprocessing: plenty of headroom

**Estimated total compute time:**
- Phase 0: 1 hour
- Phase 1b (single run): 1 hour
- Phase 1b (hyperparameter sweep, 20 configs): 20 hours
- Phase 1c (10 ablations × 1 hour): 10 hours
- **Total: ~32 hours of GPU time over 2-3 weeks**

---

## Repository Setup

I recommend a clean GitHub repo for reproducibility:

```bash
# On the Windows machine:
git init sgs-experiment
cd sgs-experiment
python -m venv .venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentence-transformers datasets wandb scipy numpy pyyaml
```

---

## Deliverables

At the end of Phase 1:

1. **A number:** STS-B Spearman correlation for SGS vs baselines
2. **An ablation table:** Which components contribute how much
3. **A go/no-go decision:** Continue to Phase 2, pivot to hybrid, or kill
4. **A trained model checkpoint** (if successful): foundation for Phase 2
5. **A short paper draft** (if successful): "Semantic Gaussian Splatting: Alpha-Compositing as Sentence Composition"
