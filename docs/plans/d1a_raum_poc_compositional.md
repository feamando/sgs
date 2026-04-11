# D1-PoC-D: Raum Compositional Scene Graph

**Goal:** Text → 3D Gaussian scene via compositional decomposition. Prove the concept works on synthetic simple scenes. 1 week. CPU only.

---

## Architecture

```
"a red sphere above a blue cube"
        ↓
  SGS encoder (frozen, pre-trained on STS-B)
        ↓
  per-word features: [B, n_words, d_f=300]
  per-word Gaussians: μ [B, n, d_s=64], Σ, α
        ↓
  ┌──────────────────────────────────┐
  │ Role classifier (per-word)       │
  │   features → {object, color,     │
  │     size, relation, other}       │
  └──────────────────────────────────┘
        ↓
  ┌──────────────────────────────────┐
  │ Attribute heads (per-word)       │
  │   object_head:   features → 6   │  (sphere, cube, cylinder, cone, plane, torus)
  │   color_head:    features → 3   │  (RGB)
  │   size_head:     features → 1   │  (scale factor)
  └──────────────────────────────────┘
        ↓
  ┌──────────────────────────────────┐
  │ Relation head (word-pair)        │
  │   [f_prep, f_ref_obj] → xyz     │  (relative offset)
  └──────────────────────────────────┘
        ↓
  Scene assembly: templates + colors + positions
        ↓
  3DGS render (gsplat or matplotlib 3D)
```

**Total trainable parameters:** ~50K (SGS encoder frozen).

---

## Scene Vocabulary

```python
OBJECTS = {
    "sphere": 0, "cube": 1, "cylinder": 2,
    "cone": 3, "plane": 4, "torus": 5,
}

COLORS = {
    "red": [1,0,0], "blue": [0,0,1], "green": [0,1,0],
    "yellow": [1,1,0], "white": [1,1,1], "black": [0.1,0.1,0.1],
    "orange": [1,0.5,0], "purple": [0.5,0,1],
}

SIZES = {
    "tiny": 0.3, "small": 0.6, "medium": 1.0, "large": 1.5, "huge": 2.0,
}

RELATIONS = {
    "above":       [0, 1.5, 0],
    "below":       [0, -1.5, 0],
    "left of":     [-1.5, 0, 0],
    "right of":    [1.5, 0, 0],
    "in front of": [0, 0, 1.5],
    "behind":      [0, 0, -1.5],
    "on":          [0, 0.8, 0],     # smaller offset than "above"
    "next to":     [1.2, 0, 0],
}
```

---

## Synthetic Data

### Sentence Templates

```
T1: "a {color} {object}"                                           → 1 object
T2: "a {size} {color} {object}"                                    → 1 object + size
T3: "a {color} {object} {relation} a {color} {object}"             → 2 objects
T4: "a {size} {color} {object} {relation} a {size} {color} {object}" → 2 objects + sizes
```

### Ground Truth Format

```python
@dataclass
class SceneGT:
    sentence: str
    objects: list[dict]   # [{type: int, color: [r,g,b], scale: float, pos: [x,y,z]}]
    # object[0] always at origin, object[1] offset by relation
```

### Splits

| Split | Templates | Held-out |
|---|---|---|
| Train | All T1-T4 combos | — |
| Val | 10% random | — |
| Comp-gen | T3 with **unseen object pairs** | Train: sphere+cube, cube+cylinder, ... Test: sphere+cylinder, ... |

The **compositional generalization** split is the key metric — can the model assemble objects it has seen individually but never seen together?

### Size

- T1: 6 × 8 = 48
- T2: 6 × 8 × 5 = 240
- T3: 6 × 8 × 8 × 6 × 8 = 18,432
- T4: 6 × 8 × 5 × 8 × 6 × 8 × 5 = 460,800
- Total: ~480K. Sample 50K for training, 5K val, 5K comp-gen test.

---

## Model

```python
# src/raum/compositional.py

class RaumCompositional(nn.Module):
    def __init__(self, sgs_encoder: SGSEncoder, d_f: int = 300):
        super().__init__()
        self.sgs = sgs_encoder
        for p in self.sgs.parameters():
            p.requires_grad = False  # frozen

        # Per-word role classifier: 5 roles
        self.role_head = nn.Linear(d_f, 5)  # object, color, size, relation, other

        # Attribute heads
        self.object_head = nn.Linear(d_f, len(OBJECTS))
        self.color_head = nn.Sequential(
            nn.Linear(d_f, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid()
        )
        self.size_head = nn.Sequential(
            nn.Linear(d_f, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()
        )

        # Relation head: takes concat of [relation_word_features, reference_object_features]
        self.relation_head = nn.Sequential(
            nn.Linear(d_f * 2, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, token_ids, mask):
        # 1. SGS encode (frozen) — get per-word features after multi-pass rendering
        #    We need per-word features, not just the final meaning vector.
        #    Use the vocab features directly (+ positional modulation from encoder).
        mu, log_var, alpha, features = self.sgs.vocab.get_params(token_ids)

        # 2. Classify roles
        role_logits = self.role_head(features)           # [B, n, 5]

        # 3. Predict attributes for each word
        obj_logits = self.object_head(features)          # [B, n, 6]
        color_pred = self.color_head(features)           # [B, n, 3]
        size_pred = self.size_head(features)             # [B, n, 1]

        return role_logits, obj_logits, color_pred, size_pred, features
```

### Scene Assembly (inference)

```python
def assemble_scene(role_preds, obj_preds, color_preds, size_preds, rel_preds):
    """
    1. Find words classified as 'object' → pick top object class
    2. Find nearest preceding 'color' word → use its color prediction
    3. Find nearest preceding 'size' word → use its scale prediction
    4. First object at origin. For subsequent objects:
       find the 'relation' word between them → predict xyz offset
    5. Return list of (object_type, color, scale, position)
    """
```

---

## Object Templates

Each object type is a pre-built set of 3D Gaussians. For the PoC, generate programmatically:

```python
# src/raum/templates.py

def make_sphere(n_gaussians=200, radius=0.5):
    """Uniformly sample points on sphere → Gaussian means. Small isotropic covariance."""
    # Fibonacci sphere sampling
    points = fibonacci_sphere(n_gaussians) * radius
    return GaussianTemplate(
        mu=points,                           # [N, 3]
        scales=torch.full((N, 3), 0.05),     # small uniform
        opacity=torch.full((N,), 0.9),
        sh=torch.zeros(N, 3),                # color set at assembly time
    )

def make_cube(n_gaussians=200, size=0.5): ...
def make_cylinder(n_gaussians=200): ...
def make_cone(n_gaussians=200): ...
def make_plane(n_gaussians=100): ...
def make_torus(n_gaussians=300): ...
```

At assembly time: `template.mu * scale + position`, `template.sh = color`.

---

## Training

```python
# scripts/train_raum_compositional.py

Loss = (
    CE(role_logits, role_labels)          # role classification
  + CE(obj_logits, obj_labels)            # object type (only for object-role words)
  + MSE(color_pred, color_labels)         # color (only for color-role words)
  + MSE(size_pred, size_labels)           # scale (only for size-role words)
  + MSE(relation_pred, relation_labels)   # xyz offset (only for relation-role words)
)
```

- Optimizer: Adam, lr=1e-3
- Batch size: 64
- Should converge in <1000 steps on CPU
- Labels come directly from the synthetic data generator (perfect supervision)

---

## Evaluation

| Metric | What | Target |
|---|---|---|
| Role accuracy | Correct word→role classification | >95% (synthetic, should be easy) |
| Object accuracy | Correct shape for object words | >95% |
| Color L2 | RGB distance | <0.05 |
| Position MAE | xyz offset error | <0.2 units |
| **Comp-gen object acc** | Object accuracy on unseen pairs | >90% (the real test) |
| **Comp-gen position MAE** | Position error on unseen pairs | <0.3 units |
| Render PSNR (optional) | Visual quality of assembled scene vs GT | >25 dB |

The comp-gen metrics are the headline — they test whether SGS's compositional encoding helps the model generalize to novel object combinations.

### Baseline

**CLIP baseline:** Replace SGS features with CLIP token embeddings. Same heads, same data, same training. If SGS comp-gen > CLIP comp-gen, the compositional inductive bias helps.

**GloVe baseline:** Use raw GloVe vectors (no SGS rendering). Same heads. Tests whether SGS's multi-pass rendering adds value over raw embeddings.

---

## Visualization

For each test sentence, produce:
1. **3D scatter plot** of assembled Gaussians (colored by object, matplotlib)
2. **Side-by-side** with ground truth
3. **Transmittance diagram** from SGS showing how words compose
4. **Failure cases** with analysis

---

## File Structure

```
src/raum/
├── __init__.py
├── vocab.py              — OBJECTS, COLORS, SIZES, RELATIONS dicts
├── templates.py          — make_sphere(), make_cube(), etc.
├── data.py               — synthetic scene generation + dataset class
├── compositional.py      — RaumCompositional model
├── assemble.py           — scene assembly from predictions
└── eval.py               — metrics

scripts/
├── train_raum_compositional.py
└── eval_raum_compositional.py
```

---

## Day-by-Day

| Day | Deliverable | Exit Criterion |
|---|---|---|
| 1 | `vocab.py`, `templates.py`, `data.py` — synthetic data pipeline | Generate 50K scenes, inspect 20 manually |
| 2 | `compositional.py` — model + training loop | Loss decreases, role accuracy >90% on train |
| 3 | `assemble.py` + evaluation | Object acc >95%, color L2 <0.05 on val |
| 4 | Comp-gen experiment + CLIP/GloVe baselines | Compare SGS vs CLIP vs GloVe on held-out pairs |
| 5 | Visualization + write-up | 10 rendered scenes, failure analysis, results table |

---

## Dependencies

```
torch          (already installed)
numpy          (already installed)
matplotlib     (for 3D visualization)
```

No GPU. No gsplat. No new heavy dependencies.
