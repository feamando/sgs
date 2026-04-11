# D1-PoC-C: Raum Shared-Equation Bridge

**Goal:** Learn a space transform from SGS semantic Gaussians to 3DGS spatial Gaussians. Test whether the structural isomorphism between SGS and 3DGS is exploitable. 2 weeks. GPU required.

---

## Core Idea

SGS encodes text as Gaussians in semantic space (d_s=64). 3DGS represents scenes as Gaussians in physical space (xyz). Both use alpha-compositing. The bridge is a learned transform between the spaces — if it learns interpretable structure, the isomorphism is real.

```
"a red sphere above a blue cube"
        ↓
  SGS encoder (frozen) → per-word Gaussians in semantic space
        ↓
  Space transform: μ_semantic → μ_xyz           (d_s → 3)
  Covariance map:  log_var_semantic → L_xyz      (d_s → 6, lower-triangular)
  Color map:       features → SH coefficients    (d_f → 3 or 48)
  Opacity map:     α_semantic → α_spatial        (1 → 1, or pass through)
        ↓
  Coarse scene: n_words Gaussians in 3D (one per word)
        ↓
  Gaussian upsampler: 1 coarse → K fine Gaussians (local cluster)
        ↓
  Dense 3DGS scene: n_words × K Gaussians
        ↓
  Differentiable render (gsplat) from M viewpoints
        ↓
  Loss vs ground truth renders
```

---

## Architecture

```python
# src/raum/bridge.py

class RaumBridge(nn.Module):
    def __init__(
        self,
        sgs_encoder: SGSEncoder,
        d_s: int = 64,
        d_f: int = 300,
        K: int = 64,         # Gaussians per word
        sh_degree: int = 0,  # 0 = RGB only, no view-dependence
    ):
        super().__init__()
        self.sgs = sgs_encoder
        for p in self.sgs.parameters():
            p.requires_grad = False

        self.K = K
        n_sh = 3 * (sh_degree + 1) ** 2  # 3 for degree 0

        # ── Space transform (the key component) ──
        # Position: semantic → xyz
        self.mu_proj = nn.Sequential(
            nn.Linear(d_s, 64), nn.ReLU(), nn.Linear(64, 3)
        )

        # ── Per-word attribute heads ──
        # Color: features → RGB (SH degree 0)
        self.color_head = nn.Sequential(
            nn.Linear(d_f, 64), nn.ReLU(), nn.Linear(64, n_sh)
        )

        # Scale: features → 3D log-scales
        self.scale_head = nn.Sequential(
            nn.Linear(d_f, 32), nn.ReLU(), nn.Linear(32, 3)
        )

        # Opacity: pass through SGS alpha (or learn a small correction)
        self.opacity_head = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

        # ── Gaussian upsampler ──
        # Input: per-word features + coarse xyz → K local offset + attributes
        up_out = K * (3 + 3 + 1 + n_sh)  # offsets, scales, opacity, color per splat
        self.upsampler = nn.Sequential(
            nn.Linear(d_f + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, up_out),
        )

    def forward(self, token_ids, mask, cameras):
        B, N = token_ids.shape

        # 1. SGS encode → per-word Gaussians
        mu_s, log_var_s, alpha_s, features = self.sgs.vocab.get_params(token_ids)
        # Optionally run SGS multi-pass to refine positions:
        # meaning = self.sgs(token_ids, mask)
        # But for PoC, raw vocab params are simpler

        # 2. Space transform
        mu_xyz = self.mu_proj(mu_s)                    # [B, N, 3]
        color_coarse = self.color_head(features)       # [B, N, 3]
        scale_coarse = self.scale_head(features)       # [B, N, 3]
        opacity_coarse = self.opacity_head(
            alpha_s.unsqueeze(-1)
        ).squeeze(-1)                                  # [B, N]

        # 3. Upsample: 1 word → K spatial Gaussians
        up_in = torch.cat([features, mu_xyz], dim=-1)  # [B, N, d_f+3]
        up_out = self.upsampler(up_in)                 # [B, N, K*(3+3+1+3)]
        up_out = up_out.view(B, N, self.K, -1)         # [B, N, K, 10]

        offsets = up_out[..., :3] * 0.5                # local offsets, bounded
        scales = up_out[..., 3:6]
        opacities = torch.sigmoid(up_out[..., 6:7]).squeeze(-1)
        colors = torch.sigmoid(up_out[..., 7:10])

        # 4. Assemble dense scene
        # Position: coarse center + local offset
        splat_mu = mu_xyz.unsqueeze(2) + offsets       # [B, N, K, 3]
        splat_mu = splat_mu.view(B, N * self.K, 3)

        splat_scales = scale_coarse.unsqueeze(2) + scales
        splat_scales = splat_scales.view(B, N * self.K, 3)

        splat_opacity = (opacity_coarse.unsqueeze(2) * opacities)
        splat_opacity = splat_opacity.view(B, N * self.K)

        splat_color = colors.view(B, N * self.K, 3)

        # 5. Mask: only real words produce Gaussians
        if mask is not None:
            word_mask = mask.unsqueeze(2).expand(B, N, self.K)
            word_mask = word_mask.reshape(B, N * self.K)
            splat_opacity = splat_opacity * word_mask.float()

        # 6. Render from each camera (done outside, in training loop)
        return {
            "means": splat_mu,         # [B, N*K, 3]
            "scales": splat_scales,    # [B, N*K, 3]
            "opacities": splat_opacity,# [B, N*K]
            "colors": splat_color,     # [B, N*K, 3]
        }
```

**Trainable parameters:**
- mu_proj: 64×64 + 64×3 ≈ 4K
- color_head: 300×64 + 64×3 ≈ 20K
- scale_head: 300×32 + 32×3 ≈ 10K
- opacity_head: ~300
- upsampler: (303×256 + 256×256 + 256×640) ≈ 300K
- **Total: ~335K** (SGS encoder frozen)

---

## Data

Same synthetic scene vocabulary as PoC-D (sphere, cube, cylinder, cone, plane, torus; 8 colors; 8 spatial relations).

### Ground Truth Generation

```python
# src/raum/data_bridge.py

def generate_scene_and_renders(sentence, scene_gt, n_views=8, img_size=128):
    """
    1. Place object templates at GT positions with GT colors/scales
    2. Render from n_views camera positions (orbit around scene center)
    3. Return: sentence, Gaussian params, rendered images [n_views, 3, H, W]
    """
    # Camera positions: orbit at fixed elevation, varying azimuth
    azimuths = torch.linspace(0, 2 * pi, n_views + 1)[:-1]
    cameras = [orbit_camera(az, elev=30, radius=4.0) for az in azimuths]

    # Render GT scene with gsplat
    gt_images = []
    for cam in cameras:
        img = gsplat_render(scene_gt.gaussians, cam, img_size)
        gt_images.append(img)

    return {
        "sentence": sentence,
        "token_ids": tokenize(sentence),
        "mask": make_mask(sentence),
        "gt_images": torch.stack(gt_images),     # [M, 3, H, W]
        "cameras": cameras,
    }
```

### Pre-render Dataset

Generate and cache 20K scenes (50K is overkill for 335K params):
- 15K train, 2.5K val, 2.5K comp-gen test
- 8 views per scene at 128x128
- Storage: 20K × 8 × 128 × 128 × 3 × 4 bytes ≈ 10 GB (float32)
- Or 20K × 8 views as PNG ≈ 1 GB (use PNG)

---

## Training

```python
# scripts/train_raum_bridge.py

for batch in dataloader:
    token_ids, mask, gt_images, cameras = batch

    # Forward: text → Gaussians
    scene = model(token_ids, mask, cameras)

    # Render from each viewpoint
    loss = 0.0
    for v in range(n_views):
        rendered = gsplat_render(
            scene["means"], scene["scales"],
            scene["opacities"], scene["colors"],
            cameras[v], img_size=128,
        )
        loss += F.mse_loss(rendered, gt_images[:, v])
        loss += 0.1 * (1 - ssim(rendered, gt_images[:, v]))

    # Regularization
    loss += 0.01 * scene["scales"].exp().mean()   # prevent Gaussian explosion
    loss += 0.001 * scene["means"].norm(dim=-1).mean()  # keep scene centered

    loss.backward()
    optimizer.step()
```

- Optimizer: Adam, lr=5e-4, weight_decay=1e-5
- Batch size: 8 (limited by rendering memory)
- Mixed precision: bf16
- Expected training: ~2-4 hours on RTX 4090 (20K scenes, 8 views, 128x128)

---

## The Key Analysis: What Does mu_proj Learn?

After training, extract and visualize the space transform:

### Test 1: Spatial Word Mapping

Encode isolated spatial words and project through mu_proj:

```python
words = ["left", "right", "above", "below", "front", "behind",
         "near", "far", "big", "small", "red", "blue"]
for w in words:
    ids = torch.tensor([[word2idx[w]]])
    mu_s = sgs.vocab.mu[ids]
    mu_xyz = model.mu_proj(mu_s)
    print(f"{w:>10s} → xyz = [{mu_xyz[0,0]:.2f}, {mu_xyz[0,1]:.2f}, {mu_xyz[0,2]:.2f}]")
```

**If mu_proj is interpretable:**
- "left"/"right" should map to opposite x values
- "above"/"below" should map to opposite y values
- "front"/"behind" should map to opposite z values
- Color words should cluster (they don't have spatial meaning)

**This analysis is publishable regardless of generation quality.**

### Test 2: Composition in Projected Space

Encode full sentences, plot the coarse word positions in 3D:

```
"a red sphere above a blue cube"
→ sphere at (0, 1.5, 0), cube at (0, 0, 0)?
→ Or: sphere and cube near each other, with correct relative offset?
```

### Test 3: Interpolation

Interpolate between two sentences in SGS semantic space, project each intermediate point through mu_proj. Does the 3D scene interpolate smoothly?

```
"a red sphere above a blue cube"  →→→  "a green cone below a white cylinder"
```

---

## Evaluation

| Metric | What | Target |
|---|---|---|
| Render PSNR | Image quality vs GT | >20 dB (coarse quality OK) |
| Render SSIM | Structural similarity | >0.7 |
| Object position MAE | Coarse Gaussian center vs GT center | <0.5 units |
| Color MSE | Predicted vs GT color per object | <0.1 |
| **mu_proj interpretability** | Do spatial words map to axes? | Qualitative |
| **Comp-gen PSNR** | Quality on unseen object pairs | >18 dB |

### Baselines

1. **Random init** (no SGS features — random projection): How much does SGS contribute?
2. **PoC-D assembly** (from compositional approach): template-based ceiling for this vocabulary
3. **CLIP features** (replace SGS with CLIP token embeddings): Does Gaussian-native matter?

---

## Dependencies

```
torch
gsplat            # pip install gsplat (differentiable 3DGS rendering)
                  # Requires CUDA. If unavailable, use pytorch3d gaussian rasterizer
                  # or simple differentiable splatting (flat alpha-blend, no sorting)
numpy
matplotlib
Pillow
```

### gsplat Fallback

If gsplat installation is problematic, implement a minimal differentiable splatting renderer:

```python
# src/raum/render_simple.py

def simple_splat_render(means_2d, colors, opacities, scales_2d, img_size):
    """
    Simplified 2D Gaussian splatting (no depth sorting, no SH).
    Each Gaussian is a 2D blob stamped onto the image.
    Uses alpha-compositing (our own rendering.py logic, extended to 2D image).
    """
    # Project means to 2D via camera
    # For each pixel, evaluate all Gaussians, alpha-composite front-to-back
    # This IS the SGS rendering equation applied to image space
```

This fallback is slower but avoids the gsplat CUDA dependency and directly reuses our rendering equation — which is thematically perfect.

---

## File Structure

```
src/raum/
├── __init__.py
├── bridge.py              — RaumBridge model
├── render_3d.py           — gsplat wrapper or simple_splat_render fallback
├── cameras.py             — orbit camera generation
├── data_bridge.py         — scene + multi-view render generation + dataset
└── analyze.py             — mu_proj visualization, interpolation, word mapping

scripts/
├── generate_raum_data.py  — pre-render GT dataset (run once, ~1 hour)
├── train_raum_bridge.py   — training loop
└── analyze_raum_bridge.py — post-training analysis (the key output)
```

---

## Day-by-Day

| Day | Deliverable | Exit Criterion |
|---|---|---|
| 1 | `cameras.py`, `render_3d.py` — rendering pipeline | Render a single GT scene from 8 views, save PNGs, visually correct |
| 2 | `data_bridge.py` + `generate_raum_data.py` | Pre-render 20K scenes. Inspect 20. Storage <2 GB |
| 3-4 | `bridge.py` — model implementation | Forward pass runs. Output shapes correct. Gradients flow (check with `torch.autograd.gradcheck` on tiny input) |
| 5-6 | `train_raum_bridge.py` — training loop | Loss decreases. After 1K steps, rendered images show vaguely correct blobs |
| 7-8 | Train to convergence | PSNR >18 dB on val. Rendered scenes recognizable |
| 9 | `analyze.py` — mu_proj analysis | Word→xyz mapping table. Scatter plots of spatial words |
| 10 | Comp-gen experiment + baselines | Results table: SGS vs CLIP vs random |
| 11-12 | Interpolation analysis + write-up | 5 interpolation sequences. Failure analysis |

---

## Decision Point After Day 6

If after 6 days the model produces no recognizable structure (PSNR <15 dB, mu_proj maps everything to the same point), **pivot:**

1. **Simplify:** Drop the upsampler. Use 1 Gaussian per word (n_words total Gaussians). If even coarse positioning fails, the space transform can't learn.
2. **Add supervision:** Add explicit position loss (not just rendering loss). This makes it partially supervised but tests whether mu_proj CAN learn the mapping.
3. **Abort C, focus on D results.** The negative result is still informative.
