# Physical Gaussians: Literature Review and Mathematical Foundations

## Part I: Literature Review

### 1. Physics-Aware Gaussian Splatting (2024-2026)

| Paper | Venue | Key Contribution | Relevance to SGS |
|-------|-------|-----------------|------------------|
| **PhysGaussian** (Xie et al.) | CVPR 2024, 442 citations | Kinematic deformation + MPM simulation on the Gaussian primitives themselves. "What you see is what you simulate." | Foundational proof that Gaussians can carry physics. Requires manual material assignment. |
| **DreamPhysics** (Huang et al.) | AAAI 2025 | Learns material properties from video diffusion via "motion distillation sampling." KAN-based material field. | Shows material can be *learned* not just assigned. |
| **PhysDreamer** (Zhang et al.) | ECCV 2024 | Distills dynamics from video generation models to endow static 3D objects with physics. | Uses video priors as proxy for material stiffness, no ground-truth needed. |
| **GASP** (Borycki et al.) | 2024 | Converts Gaussians to triangle meshes for standard physics engines (Genesis). | Pragmatic: no custom solver, works with existing engines. |
| **PhysSplat** (Zhao et al.) | ICCV 2025 | Uses GPT-4V to estimate material properties from static images, drives MPM on Gaussians. | Bridges perception and simulation without video. |
| **PG-3DGS** (Lee, Jacobson, Xue) | 2026 | Differentiable physics objectives in the 3DGS optimization loop. Joint inference of geometry and material. | Closest to our "backprop through physics to refine e_p" idea. |
| **GaussianProperty** (Xu et al.) | ICCV 2025 | Training-free: SAM segmentation + GPT-4V material reasoning in 2D, projected to 3D via multi-view voting. | Demonstrates LMM-driven property assignment without training. |

**Key takeaway:** PhysGaussian proved Gaussians can be both rendering and physics primitives. The field is moving toward *learned* material assignment (DreamPhysics, PhysSplat, PG-3DGS) rather than manual. Our approach differs: we predict e_p from the semantic embedding (which already encodes "what this is") rather than from external vision models.

### 2. Material Property Prediction

| Paper | Approach | Data Source |
|-------|----------|-------------|
| **PAC-NeRF** (Li et al., ICLR 2023) | Estimates elasticity, viscosity, friction from multi-view video via hybrid Eulerian-Lagrangian simulation | Video observation |
| **GaussianProperty** (above) | GPT-4V reasons about materials from rendered images | Static images + LMM |
| **LLM-Guided Material Inference** (Izadyar, Schneider, 2025) | LLMs + geometric deep learning predict material from point cloud geometry alone | Point cloud shape |
| **PhysSplat** (above) | Multimodal LLM estimates from single image | Static image |

**What we do differently:** All existing work infers material from *visual observation* (images, video) or *external LLMs*. Our hypothesis is that material is inferable from the *semantic embedding* directly, because the embedding already encodes "this is stone" or "this is cloth." No external vision model needed, no image rendering required for the prediction. The correlation is: meaning implies physics.

### 3. Unified Scene Representations

| System | Rendering | Physics | Semantics | Audio | Unified Primitive? |
|--------|-----------|---------|-----------|-------|-------------------|
| Traditional engine | Mesh | PhysX collider | Manual tags | Enums | No (4 separate) |
| PhysGaussian | 3DGS | MPM on Gaussians | None | None | Partial (render+phys) |
| LangSplat | 3DGS | None | CLIP per-Gaussian | None | Partial (render+sem) |
| Feature 3DGS | 3DGS | None | SAM+CLIP per-Gaussian | None | Partial (render+sem) |
| AV-GS | 3DGS | None | None | Material-aware acoustics | Partial (render+audio) |
| PAC-NeRF | NeRF | MPM | None | None | Partial (render+phys) |
| **Physical Gaussians (ours)** | 3DGS | From e_p | From e_s | From e_p + e_s | **Full (all four)** |

**The gap we fill:** No existing work unifies all four channels (rendering, physics, semantics, audio) in a single primitive. Each paper solves one bridge (render+physics, or render+semantics). Our contribution is the unified primitive where all four are projections of the same underlying representation.

### 4. Per-Gaussian Feature Vectors (Semantic Gaussians)

| Paper | Feature Type | How Trained | Open-Vocab? |
|-------|-------------|-------------|-------------|
| **Gaussian Grouping** (Ye et al., ECCV 2024, 487 citations) | Identity/instance | SAM masks projected to 3D | Segment-anything |
| **LangSplat** (Qin et al., CVPR 2024) | CLIP language features | Autoencoder from CLIP | Open-vocabulary queries |
| **Feature 3DGS** (Zhou et al., 2024) | SAM + CLIP-LSeg | Distillation | Point/box/language prompts |
| **GaussianVLM** (Jul 2025) | Linguistic features per-Gaussian | Aligned to LLM space | 5x improvement over prior |

**What this validates:** Per-Gaussian feature vectors work. They're efficient (small storage overhead per Gaussian), trainable (distillation or direct supervision), and enable downstream tasks. Our e_p is the same paradigm applied to physics rather than language.

### 5. Audio from Gaussians

| Paper | Connection |
|-------|-----------|
| **AV-GS** (Bhosale et al., NeurIPS 2024) | Material-aware acoustic synthesis from GS scene representation. Room geometry + material properties -> sound propagation. |
| **VibraVerse** (Pang et al., 2025) | Large-scale dataset pairing object geometry + material with impact sounds. Enables geometry-to-audio learning. |

**Connection to Klang:** Our Klang track already synthesizes audio from Gaussian parameters (position, scale, transmittance). Adding e_p gives Klang the missing piece: what material is being struck, scraped, or vibrated. Impact sound = f(e_p hardness, geometry, velocity).

---

## Part II: Mathematical Foundations Audit

### What exists in the SGS codebase

| Component | What's implemented | Where |
|-----------|-------------------|-------|
| Quaternion multiplication | Hamilton product `[w,x,y,z]` | `decomposition.py:_quat_mul` |
| Quaternion composition in tree | Parent * child rotation propagation | `decomposition.py:flatten_gaussians` |
| Scale compounding (log-space) | `g.scale[i] + log(world_scale)` | `decomposition.py:flatten_gaussians` |
| Tensor export with rotations | `[N,4]` quaternion tensors | `decomposition.py:tree_to_tensors` |
| Diagonal covariance (semantic space) | `log_var [vocab_size, d_s]` | `gaussian.py:SemanticGaussianVocab` |
| Cosine similarity | `F.cosine_similarity` | `model.py` (multiple models) |
| Nearest-neighbor (spatial) | `torch.cdist` on positions | `densify.py:densify_loop` |
| Variable-length PLY properties | `f_rest_*` pattern for SH bands | `export/ply.py:write_ply` |
| MLP with multiple heads | template_head + deform_head | `subdivider.py:SubdivisionMLP` |

### What's missing (5 specific gaps)

#### Gap 1: Quaternion to rotation matrix

**Needed for:** Building the full 3x3 covariance matrix `Sigma = R @ diag(s^2) @ R^T`. Required to compute covariance similarity between Gaussians (part of the material clustering hypothesis).

**Math:**
```
Given q = [w, x, y, z]:

R = | 1-2(y^2+z^2)   2(xy-wz)       2(xz+wy)     |
    | 2(xy+wz)       1-2(x^2+z^2)   2(yz-wx)     |
    | 2(xz-wy)       2(yz+wx)       1-2(x^2+y^2) |
```

**Where it goes:** `src/raum/decomposition.py` or `src/export/utils.py` as `quat_to_rotmat(q: Tensor) -> Tensor`.

#### Gap 2: Physical embedding field (e_p)

**Needed for:** Storing per-Gaussian material properties.

**Math:** `e_p in R^d_p` (d_p = 32-64). Initialized from a material lookup table, refined by the physics prediction network.

**Where it goes:**
- `GaussianParams.physics_embedding: list[float] | None = None`
- `tree_to_tensors` returns `"physics_embeddings": [N, d_p]`
- `SemanticGaussianVocab.physics_features: nn.Parameter [vocab_size, d_p]`

#### Gap 3: Combined similarity metric

**Needed for:** Material region clustering. Gaussians form material regions when they're similar across spatial, semantic, AND geometric dimensions.

**Math:**
```
material_sim(G_i, G_j) = w_s * cos(e_s_i, e_s_j)           # semantic
                        + w_g * exp(-||S_i - S_j||_2)        # geometric (scale)
                        + w_a * exp(-|alpha_i - alpha_j|)     # opacity
                        + w_d * exp(-||p_i - p_j||_2 / r)    # spatial proximity
                        + w_c * cov_similarity(Sigma_i, Sigma_j)  # covariance
```

Where covariance similarity can be computed via:
```
cov_similarity(A, B) = exp(-D_KL(N(0,A) || N(0,B)))
                     = exp(-0.5 * (tr(B^{-1}A) + tr(A^{-1}B) - 2d))
```

Or simplified via Frobenius norm of log-ratio for diagonal covariances:
```
cov_similarity(A, B) = exp(-||log(diag(A)) - log(diag(B))||_2)
```

**Where it goes:** `src/raum/densify.py` (extend NN computation) and a new `src/raum/physics.py`.

#### Gap 4: Physics prediction head on SubdivisionMLP

**Needed for:** When a Gaussian is subdivided into children, each child needs a physics embedding. The MLP should predict this from the semantic + geometric context.

**Math:** Third output head: `Linear(256, d_p)`. Input extended by d_p dimensions. Loss: L2 distance to ground-truth material vectors from the lookup table.

**Where it goes:** `src/raum/subdivider.py:SubdivisionMLP` (add `self.physics_head`).

#### Gap 5: PLY export with physics properties

**Needed for:** Exporting Physical Gaussians to external tools.

**Format:** Add `property float phys_0` through `property float phys_{d_p-1}` after the rotation block in the PLY header.

**Where it goes:** `src/export/ply.py:write_ply` (follow the `f_rest_*` pattern).

### Covariance construction (the key missing math)

The full pipeline for computing geometric similarity between two Gaussians:

```python
def quat_to_rotmat(q: Tensor) -> Tensor:
    """[4] quaternion -> [3,3] rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]),
    ])

def build_covariance(scale_log: Tensor, rotation: Tensor) -> Tensor:
    """Build 3x3 covariance from log-scale [3] and quaternion [4]."""
    S = torch.diag(torch.exp(scale_log))  # [3,3]
    R = quat_to_rotmat(rotation)           # [3,3]
    return R @ S @ S @ R.T                 # [3,3] = R @ diag(s^2) @ R^T

def covariance_similarity(cov_a: Tensor, cov_b: Tensor) -> float:
    """Simplified: Frobenius norm of log-eigenvalue difference."""
    # For computational efficiency with diagonal-dominant covariances:
    eig_a = torch.linalg.eigvalsh(cov_a)
    eig_b = torch.linalg.eigvalsh(cov_b)
    return torch.exp(-(torch.log(eig_a) - torch.log(eig_b)).norm()).item()
```

### Physics prediction network

```python
class PhysicsPredictionMLP(nn.Module):
    """Predict physical embedding from semantic + geometric features."""

    def __init__(self, d_s: int = 300, d_p: int = 32):
        super().__init__()
        # Input: semantic_embed(d_s) + scale(3) + opacity(1) + covariance_features(6)
        input_dim = d_s + 3 + 1 + 6
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, d_p),
        )

    def forward(self, semantic_embed, scale_log, opacity, covariance_features):
        x = torch.cat([semantic_embed, scale_log, opacity, covariance_features], dim=-1)
        return self.net(x)  # [B, d_p]
```

The `covariance_features` are the upper-triangle of the 3x3 covariance matrix (6 values), providing shape/orientation information to the physics predictor.

---

## Part III: Gap Between Us and Prior Art

### What PhysGaussian does that we don't (yet)

1. **MPM simulation integration.** PhysGaussian runs Material Point Method on the Gaussians. We have no physics simulator integrated.
2. **Constitutive models.** PhysGaussian supports specific material models (Neo-Hookean, von Mises plasticity). We have continuous embeddings, not explicit constitutive equations.
3. **Force computation.** PhysGaussian computes stress tensors from deformation gradients. We have no force/stress math.

### What we do that PhysGaussian doesn't

1. **Semantic embeddings.** PhysGaussian has no per-Gaussian semantic vector. Materials are manually assigned per-region.
2. **Predicted physics.** PhysGaussian requires manual material assignment. We predict from semantics.
3. **Unified with language.** Our primitive connects text (composition tree) -> semantics (e_s) -> physics (e_p). PhysGaussian starts from a reconstructed scene with no text connection.
4. **Audio connection.** PhysGaussian does not address sound. Our architecture connects to Klang.

### The unique contribution

The literature shows:
- You CAN do physics on Gaussians (PhysGaussian, 2024)
- You CAN embed language features per-Gaussian (LangSplat, 2024)
- You CAN predict materials from perception (GaussianProperty, PhysSplat, 2025)
- You CAN synthesize audio from GS scenes (AV-GS, 2024)

Nobody has yet: encoded all of these as projections of a single learned embedding space where the primitive's meaning, shape, physics, and sound are correlated dimensions of one vector that collapses into different modalities on query.

That is the Physical Gaussian thesis.
