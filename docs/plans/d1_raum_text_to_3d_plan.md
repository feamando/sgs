# D1: Radiance Raum — Text-to-3D Gaussian Splatting via SGS

## Concept

Use SGS's Gaussian-native text representation to bridge language and 3D scenes. If SGS renders meaning from word Gaussians using the same equation that 3DGS uses to render images from spatial Gaussians, then the mathematical machinery is already shared. Raum explores whether that shared structure enables:

1. **Text → 3D:** Generate 3D Gaussian splat scenes from text descriptions
2. **3D → Text:** Extract natural-language descriptions from existing 3DGS scenes
3. **Text × 3D:** Modify a 3DGS scene via text instructions

**The hook:** *"Both your words and your 3D scene are Gaussians rendered by the same equation. We just connect the spaces."*

---

## Why SGS Makes This Different

Every existing text-to-3D system (DreamGaussian, GSGEN, GaussianDreamer, GaussianEditor) uses the same pipeline:

```
Text → CLIP embedding → Score Distillation Sampling → optimize 3DGS
```

CLIP encodes text as a single dense vector. The 3D Gaussians are optimized from scratch via gradient descent against a diffusion prior. There is no structural relationship between the text representation and the 3D representation — CLIP and 3DGS share nothing architecturally.

SGS is structurally different:

```
              SGS                           3DGS
        ───────────────               ───────────────
Primitive:  Word Gaussian               Spatial Gaussian
            (μ, Σ, α, f)               (μ, Σ, α, SH)
Space:      Semantic (d_s)              Physical (xyz)
Compose:    Alpha-compositing           Alpha-compositing
Output:     Meaning vector              Rendered pixel
Ordering:   Sequence position           Depth sort
```

The primitive is the same (Gaussian). The composition is the same (alpha-compositing with transmittance). The rendering equation is identical. Only the spaces differ: semantic vs. physical. This structural isomorphism is unique to SGS — no other text representation shares the primitive and composition mechanism with 3DGS.

**The research question:** Can a learned projection between semantic space and physical space leverage this shared structure to generate, describe, or edit 3D scenes?

---

## Feasibility Analysis: Five Approaches

### Approach A: Direct Gaussian Projection

```
Text → SGS encoder → semantic Gaussians {(μ_s, Σ_s, α_s, f_s)}_i
                         ↓ learned projection
                    spatial Gaussians {(μ_xyz, Σ_xyz, α_xyz, SH)}_i
                         ↓ standard 3DGS rendering
                    rendered image
```

**How it works.** Train a projection network that maps each SGS Gaussian's parameters to 3DGS parameters. The word "sphere" in SGS has (μ, Σ, α, f) in semantic space; the projection maps these to (xyz position, 3D covariance, opacity, spherical harmonics) in physical space.

**Training data.** Paired (text caption, 3DGS scene). Objaverse-XL has 800K+ annotated 3D objects.

**Strengths.**
- Conceptually clean: one Gaussian in, one Gaussian out
- Leverages the structural alignment directly
- No diffusion model needed
- Fast inference (single forward pass)

**Weaknesses.**
- SGS produces one Gaussian per word (~5-20). 3DGS scenes need 10K-1M Gaussians. The cardinality mismatch is severe — each word-Gaussian must "expand" into many spatial Gaussians.
- The learned semantic space has no guaranteed geometric relationship to physical xyz coordinates. "Left" and "right" are close in semantic space (both directions) but opposite in physical space.
- Features in SGS carry semantic content (d_f=300). SH coefficients in 3DGS encode view-dependent color. These are fundamentally different representations.

**Feasibility: LOW.** The cardinality mismatch and space mismatch make direct projection unlikely to work without heavy additional machinery. If you add enough machinery (upsampling, refinement), you're essentially training a generative model with extra steps.

---

### Approach B: SGS-Conditioned Gaussian Generative Model

```
Text → SGS encoder → meaning vector (d_f)
                         ↓ conditioning
       Gaussian Diffusion/Flow Model → 3DGS scene
                         ↓
       rendered image (multi-view supervision)
```

**How it works.** Use SGS as the text encoder (replacing CLIP) in a generative model that outputs 3DGS parameters. The generative model — a diffusion model, flow-matching model, or autoregressive model — produces Gaussian parameters conditioned on SGS's meaning vector.

**Training data.** (Text caption, multi-view renders) or (text caption, 3DGS scene). Objaverse-XL + Cap3D captions.

**Strengths.**
- Proven paradigm: DreamGaussian, GSGEN already work with CLIP conditioning
- SGS provides a richer conditioning signal than CLIP (it preserves compositional structure via transmittance, not just a single vector)
- Can condition on per-word Gaussians (not just the final meaning), giving the generator compositional scaffolding
- Handles the cardinality mismatch naturally (generator decides how many Gaussians to produce)

**Weaknesses.**
- Requires training a generative model — significant compute and data
- SGS's 100M-param Planck model is a weaker text encoder than CLIP (trained on much less data). Results may be worse than CLIP-based systems initially
- The structural alignment between SGS and 3DGS is underutilized: the generator still learns to produce Gaussians from scratch

**Feasibility: MEDIUM-HIGH.** This will work because the paradigm is proven. The research question becomes whether SGS conditioning is better than CLIP conditioning — particularly for compositional scenes ("a red sphere to the left of a blue cube on a green plane") where SGS's compositional inductive bias should help.

---

### Approach C: Shared-Equation Bridging (SGS-native)

```
Text → SGS encoder → semantic Gaussians
                         ↓ space transform T: R^d_s → R^3
                    coarse spatial Gaussians (one per word)
                         ↓ Gaussian upsampler (1 → N splats per word)
                    dense spatial Gaussians
                         ↓ SH head (features → color)
                    full 3DGS scene
```

**How it works.** The key insight: SGS's multi-pass rendering already positions word Gaussians in semantic space via learned (μ, Σ, α). If we add a learned linear projection T: R^d_s → R^3 that maps semantic positions to spatial positions, then after SGS encodes the sentence, each word-Gaussian has an approximate 3D position. A Gaussian upsampler then expands each word-Gaussian into a cluster of spatial Gaussians (e.g., "sphere" → 500 Gaussians arranged in a sphere shape). A small SH head converts SGS features to view-dependent color.

**Training.** Multi-view supervision (render from multiple angles, compare to ground truth). The space transform T, upsampler, and SH head are trained end-to-end. SGS encoder can be frozen or fine-tuned.

**Why this is novel.** The rendering equation is literally shared: SGS composes word meanings via alpha-compositing, then the same alpha-compositing renders the 3D scene. The text-to-3D pipeline IS a rendering pipeline at both ends. No other approach has this — it's only possible because SGS represents text as Gaussians.

**Strengths.**
- Most architecturally novel — leverages the SGS/3DGS structural isomorphism fully
- The space transform T is a small, interpretable component — we can visualize where words map in 3D
- Compositional by construction: "red sphere left of blue cube" → SGS positions the word Gaussians relative to each other in semantic space → T projects these relative positions to 3D
- The upsampler is per-word, so it can learn shape priors (the "sphere" upsampler learns sphere topology)
- Potentially lightweight: SGS encoder is frozen, only train T + upsampler + SH head

**Weaknesses.**
- Assumes SGS's learned semantic space has structure that meaningfully maps to 3D. This is unproven. Semantic similarity ≠ spatial proximity.
- The upsampler must learn to produce diverse shapes from a single Gaussian — this is essentially a shape generator per word, which is hard
- Scenes with complex spatial relations (occlusion, containment, support) may require more than a linear space transform
- Novel architecture with no prior validation — high research risk

**Feasibility: LOW-MEDIUM.** The most interesting approach but the riskiest. Worth attempting as the PoC because even a partial success would be a significant finding. If the space transform T learns interpretable structure (spatial words map to spatial axes), that alone is publishable.

---

### Approach D: Compositional Scene Graph via SGS

```
Text → SGS encoder → per-word Gaussians + rendered meaning
                         ↓ scene graph extraction
      noun phrases → objects (3D primitive + Gaussian cluster)
      adjectives → properties (color, size, material)
      prepositions → spatial relations (relative xyz offsets)
                         ↓ scene assembly
      positioned 3D Gaussian clusters
                         ↓ optional refinement (SDS or reconstruction loss)
      final 3DGS scene
```

**How it works.** Rather than learning an end-to-end mapping, decompose the problem linguistically. SGS's rendering naturally assigns different weights to different words (transmittance + opacity), and its multi-pass refinement positions words relative to each other in semantic space. Extract this structure as a scene graph, then map to 3D using:

- **Object library:** Pre-built 3DGS templates for common nouns (sphere, cube, chair, table, tree). Each template is a cluster of Gaussians representing the object shape.
- **Property modifiers:** Adjective → Gaussian parameter changes (color adjectives → SH modification, size adjectives → covariance scaling).
- **Spatial relations:** Preposition → xyz offset relative to reference object ("above" → +y, "left of" → -x).

**Training.** The object library can be pre-built from Objaverse. Property modifiers and spatial relations can be learned from captioned 3D scenes or even hand-coded for a PoC.

**Strengths.**
- Most feasible for a quick PoC — can produce visible results with minimal training
- Transparent and interpretable: you can see exactly which word maps to which object
- Leverages SGS's compositional structure: the transmittance ordering and per-word weighting directly inform scene assembly
- Can start with a tiny vocabulary (10 objects, 5 colors, 5 relations) and scale up

**Weaknesses.**
- Requires explicit linguistic parsing (noun/adjective/preposition extraction) — brings in NLP machinery beyond SGS
- Limited to scenes describable as simple object + property + relation compositions
- Object library approach doesn't generalize to novel objects
- Doesn't leverage the mathematical isomorphism between SGS and 3DGS as deeply as Approach C

**Feasibility: HIGH for simple scenes, LOW for complex scenes.** Good PoC path. Demonstrates the concept with guaranteed visual results.

---

### Approach E: Bidirectional Gaussian Alignment

```
Training (contrastive):
  Text → SGS encoder → semantic Gaussians → meaning vector
  3D   → 3DGS render → multi-view images → visual encoder → visual embedding
  
  Loss: align meaning vector with visual embedding (contrastive)
  
Inference (text → 3D):
  Text → SGS → meaning vector → search nearest 3D scene in dataset
  
Inference (3D → text):
  3DGS → visual encoder → embedding → search nearest text description
```

**How it works.** Train a contrastive alignment between SGS text representations and 3DGS visual representations. Like CLIP, but both sides use Gaussians. The text side uses SGS's rendering equation; the visual side renders the 3DGS scene from multiple viewpoints and encodes the renders.

This doesn't generate new 3D scenes — it retrieves the closest match from a database. But it establishes the alignment between the two Gaussian spaces, which can then be used for generation (via optimization in the aligned space) or editing (via moving in the aligned space).

**Strengths.**
- Avoids the generation problem entirely for the first stage
- Establishes a measurable alignment between SGS and 3DGS
- Retrieval-based approach always produces valid 3D scenes
- The alignment can later be used as a loss function for generative approaches (B, C)

**Feasibility: MEDIUM-HIGH.** Contrastive alignment is well-understood. The question is whether SGS's semantic space aligns with visual features better than CLIP does.

---

## Comparative Assessment

| Approach | Novelty | Risk | PoC Effort | Generative? | Leverages SGS Isomorphism |
|---|---|---|---|---|---|
| A: Direct Projection | Medium | High | 5 days | Yes | Fully |
| B: SGS-Conditioned Gen | Low | Low | 3-4 weeks | Yes | Partially (conditioning only) |
| C: Shared-Equation Bridge | **Very High** | High | 2 weeks | Yes | **Fully** |
| D: Compositional Scene Graph | Medium | Low | **1 week** | Yes (constrained) | Partially |
| E: Bidirectional Alignment | Medium | Medium | 2 weeks | No (retrieval) | Partially |

**Recommendation for Objective 1 (feasibility):**

Run two parallel PoCs:

1. **Approach D** (compositional scene graph) — 1 week, guaranteed visual results, proves the concept is coherent. This is the "can we get anything working at all" experiment.

2. **Approach C** (shared-equation bridging) — 2 weeks, high risk but high novelty. This is the "does the SGS/3DGS structural isomorphism actually buy us something" experiment. Even if generation quality is poor, analyzing what the space transform T learns is publishable.

Approach B is the safe fallback if C fails — swap SGS for CLIP in a proven pipeline and compare.

---

## Proof-of-Concept Experiment: Approach D (Compositional Scene Graph)

### Setup

**Scene vocabulary:**

| Category | Items |
|---|---|
| Objects | sphere, cube, cylinder, cone, plane, torus |
| Colors | red, blue, green, yellow, white, black, orange, purple |
| Sizes | small, medium, large, tiny, huge |
| Relations | above, below, left of, right of, in front of, behind, on, next to |

**Sentence templates:**
```
"a {color} {size} {object}"
"a {color} {object} {relation} a {color} {object}"
"a {color} {object} {relation} a {color} {object} {relation} a {color} {object}"
```

This gives ~6 × 8 × 5 = 240 single-object descriptions, ~240 × 8 × 240 = 460K two-object scenes, enough for training.

### Architecture

```python
# poc/raum_poc.py

class RaumCompositional(nn.Module):
    """PoC: text → 3D Gaussian scene via compositional decomposition."""

    def __init__(self, sgs_encoder, object_library):
        super().__init__()
        self.sgs = sgs_encoder  # frozen SGS-2pass

        # Map SGS features → object class logits
        self.object_head = nn.Linear(d_f, n_objects)

        # Map SGS features → color (RGB)
        self.color_head = nn.Sequential(
            nn.Linear(d_f, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid(),
        )

        # Map SGS features → scale factor
        self.scale_head = nn.Sequential(
            nn.Linear(d_f, 32), nn.ReLU(),
            nn.Linear(32, 3), nn.Softplus(),
        )

        # Map SGS Gaussian pairs → relative xyz offset
        self.relation_head = nn.Sequential(
            nn.Linear(d_f * 2, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )

        # Pre-built 3DGS templates per object class
        self.templates = object_library  # dict: class → Gaussian params
```

### Data

**Ground truth generation.** Programmatically render scenes from the vocabulary:

1. Pick a sentence template + fill slots randomly
2. Place 3DGS object templates at computed positions (based on spatial relations)
3. Apply color/scale modifications
4. Render multi-view images (Blender or differentiable 3DGS renderer)
5. Store: (sentence, 3DGS scene params, rendered views)

This is fully synthetic — no real-world data needed for the PoC.

### Training

```
Loss = L_render + L_object + L_spatial

L_render:  MSE between rendered views and ground truth renders
L_object:  Cross-entropy on object classification
L_spatial: MSE on predicted xyz offsets vs ground truth positions
```

Train SGS encoder frozen. Only train the heads (object, color, scale, relation). ~50K parameters total. Trains in minutes on CPU.

### Evaluation

| Metric | What it measures |
|---|---|
| Object accuracy | Does the model pick the right shape? |
| Color L2 | Does the rendered color match? |
| Position error | How close is object placement to ground truth? |
| Render PSNR | Visual quality of final rendered image |
| Compositional generalization | Performance on unseen 2-object combos (train on A+B, B+C; test A+C) |

The **compositional generalization** metric is the most interesting — it directly parallels SCAN. If SGS's compositional rendering enables better generalization to novel object combinations than a CLIP-based baseline, that's the headline finding.

### Timeline

| Day | What |
|---|---|
| 1 | Build object template library (6 shapes as 3DGS clusters) |
| 2 | Synthetic data generation (sentences + scenes + renders) |
| 3 | Train compositional model (object/color/scale/relation heads) |
| 4 | Evaluate + compositional generalization experiment |
| 5 | CLIP baseline comparison + analysis |
| 6-7 | Write up, visualizations, failure analysis |

---

## Proof-of-Concept Experiment: Approach C (Shared-Equation Bridge)

### Setup

Same scene vocabulary as Approach D, but the architecture is fundamentally different.

### Architecture

```python
# poc/raum_bridge.py

class RaumBridge(nn.Module):
    """
    Shared-equation bridge: SGS Gaussians → 3DGS Gaussians.
    The rendering equation is used on both sides.
    """

    def __init__(self, sgs_encoder):
        super().__init__()
        self.sgs = sgs_encoder  # frozen

        # Space transform: semantic → physical
        # Separate transforms for μ and Σ
        self.mu_transform = nn.Linear(d_s, 3)    # semantic position → xyz
        self.cov_transform = nn.Linear(d_s, 6)   # semantic shape → 3D covariance (lower-tri)

        # Feature → SH color (degree 0 = RGB, higher = view-dependent)
        self.sh_head = nn.Linear(d_f, 3 * (sh_degree + 1)**2)

        # Gaussian upsampler: 1 word-Gaussian → N spatial Gaussians
        self.upsampler = nn.Sequential(
            nn.Linear(d_f + 3, 256),  # features + xyz position
            nn.ReLU(),
            nn.Linear(256, N_per_word * (3 + 6 + 1 + 3)),  # μ, L, α, RGB per splat
        )

    def forward(self, token_ids, mask):
        # 1. SGS encodes text → semantic Gaussians
        mu_s, log_var_s, alpha_s, features = self.sgs.vocab.get_params(token_ids)
        # ... (run SGS multi-pass to get refined positions)

        # 2. Project to 3D
        mu_xyz = self.mu_transform(mu_s)            # [B, n_words, 3]
        cov_xyz = self.cov_transform(log_var_s)     # [B, n_words, 6]
        sh = self.sh_head(features)                 # [B, n_words, n_sh]

        # 3. Upsample: each word-Gaussian → N spatial Gaussians
        up_input = torch.cat([features, mu_xyz], dim=-1)
        splats = self.upsampler(up_input)            # [B, n_words, N * params]
        # Reshape to [B, n_words * N, params]

        # 4. Offset upsampled splats relative to word position
        splat_mu = mu_xyz.unsqueeze(2) + splat_offsets  # local clusters

        # 5. Render via standard 3DGS differentiable renderer
        rendered = render_3dgs(splat_mu, splat_cov, splat_alpha, splat_sh, camera)

        return rendered
```

### Key Experiment: What Does T Learn?

The most informative analysis is visualizing the space transform `mu_transform`:

- Does it map spatial words (left, right, above, below) to corresponding xyz axes?
- Does it place semantically related words nearby in 3D?
- Does sequence ordering in SGS (via transmittance) translate to depth ordering in 3D?

Even if generation quality is poor, if T learns interpretable structure, that validates the research direction.

### Training

Use a differentiable 3DGS renderer (gsplat or diff-gaussian-rasterization). Multi-view supervision:

```
Loss = Σ_views MSE(rendered_view, gt_view) + λ_reg * L_reg
```

### Timeline

| Day | What |
|---|---|
| 1-2 | Set up differentiable 3DGS rendering pipeline (gsplat) |
| 3-4 | Implement RaumBridge model |
| 5 | Synthetic data generation (same as Approach D) |
| 6-8 | Train with multi-view supervision |
| 9-10 | Analyze space transform T, visualize, compare to Approach D |
| 11-12 | Compositional generalization experiment |
| 13-14 | Write up |

---

## Reverse Direction: 3D → Text

Once either approach works for text → 3D, the reverse is straightforward:

```
3DGS scene → render multi-view → visual encoder → embedding
                                                      ↓
                                                 nearest SGS meaning vector
                                                      ↓
                                                 decode to text (via Planck LM or retrieval)
```

For Approach C specifically, the reverse is architecturally natural:

```
3DGS scene → extract Gaussian params → inverse space transform T⁻¹ → semantic Gaussians
           → SGS rendering in semantic space → meaning vector → decode to text
```

This only works if T is invertible (or approximately so). A linear T is always pseudo-invertible.

---

## Text-Guided 3D Editing

```
Existing 3DGS scene + "make the sphere green"
    ↓
SGS encodes instruction → identifies target ("sphere") + modification ("green")
    ↓
Match target to scene Gaussians (via feature similarity in aligned space)
    ↓
Apply modification to matched Gaussians (update SH coefficients for color)
    ↓
Modified 3DGS scene
```

For Approach C, the alignment between SGS features and 3DGS features makes the matching step natural — both live in spaces connected by the learned transform.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SGS semantic space has no useful geometric relationship to 3D xyz | **High** | Fatal for C, irrelevant for D | Approach D doesn't need this. For C, the space transform T can learn non-linear mappings (swap Linear for MLP). |
| Cardinality mismatch (5-20 word Gaussians → 10K+ scene Gaussians) | High | Degrades quality | Upsampler in C; template library in D. Both handle this explicitly. |
| SGS features don't encode visual properties (color, shape) | Medium | Degrades color/shape accuracy | Feature space carries GloVe semantics. "Red" and "blue" ARE distinguishable in GloVe space. The color head can learn the mapping. For Planck (learned from scratch), features may encode whatever is useful. |
| Differentiable 3DGS rendering is slow / hard to set up | Medium | Delays PoC | Use gsplat (maintained, pip-installable). Fallback: non-differentiable rendering + CLIP loss. |
| Compositional generalization doesn't beat CLIP baseline | Medium | Weakens paper | This is an honest finding either way. If CLIP matches SGS, the structural isomorphism doesn't help — worth knowing. |
| The concept is too speculative for publication | Low | Reduces impact | The feasibility analysis itself is valuable. Negative results on C + positive results on D = "the structural isomorphism is elegant but insufficient; compositional decomposition is the practical path." |

---

## Related Work

| Paper | What | Raum's Difference |
|---|---|---|
| DreamGaussian (Tang et al., 2023) | Image/text → 3DGS via SDS | Uses CLIP, not Gaussian-native text |
| GSGEN (Chen et al., 2024) | Text → 3DGS via progressive SDS | Same |
| GaussianEditor (Chen et al., 2024) | Text-guided 3DGS editing | Uses InstructNeRF2NeRF approach |
| LangSplat (Qin et al., 2024) | Embed CLIP features INTO 3DGS | Language into splats; Raum is splats from language |
| Point-E (Nichol et al., 2022) | Text → point cloud → mesh | Point clouds, not Gaussians |
| 3D-LLM (Hong et al., 2023) | LLM grounded in 3D scenes | Uses LLM, not rendering equation |
| Word2Gauss (Vilnis & McCallum, 2015) | Words as Gaussians | No composition, no 3D |
| SGS (Gorshkov, 2026) | Alpha-compositing for language | Raum extends SGS to 3D generation |

**Raum's unique angle:** Every existing system treats text and 3D as separate modalities connected by a bridge (CLIP, SDS). Raum is the first to use a representation where both text and 3D scenes are natively the same primitive (Gaussian) composed by the same operation (alpha-compositing).

---

## What We Learn (Regardless of Success)

| Outcome | What it means |
|---|---|
| Approach D works, C fails | Compositional structure transfers but geometric structure doesn't. SGS is useful for parsing, not for spatial mapping. |
| Both D and C work | The SGS/3DGS structural isomorphism is real and exploitable. Major finding. Paper upgrades to top venue. |
| Both fail | The rendering equation works for composition but the semantic→physical space gap is too large. Negative result, still publishable as an analysis. |
| C's space transform T learns interpretable structure | Even if generation quality is poor, this validates the research direction and informs better architectures. |
| SGS conditioning beats CLIP for compositional scenes | SGS's compositional inductive bias transfers to 3D generation. Strong result for Approach B as a practical system. |

---

## Cost & Resources

| Item | Approach D | Approach C |
|---|---|---|
| Compute | CPU only (minutes) | GPU, ~2-4 hours |
| Data | Synthetic (generate on the fly) | Synthetic + multi-view renders |
| Dependencies | PyTorch, existing SGS code | + gsplat or diff-gaussian-rasterization |
| Human time | 1 week | 2 weeks |
| Total cost | ~$0 | ~$5-10 (electricity) |

---

## File Structure

```
docs/plans/d1_raum_text_to_3d_plan.md     — this document
src/raum/
├── __init__.py
├── scene_vocab.py          — object templates, colors, spatial relations
├── compositional.py        — Approach D: scene graph decomposition
├── bridge.py               — Approach C: shared-equation bridge
├── render_3d.py            — 3DGS rendering wrapper (gsplat)
├── data.py                 — synthetic scene generation
└── eval.py                 — metrics (PSNR, object acc, position error)
scripts/
├── train_raum_compositional.py
├── train_raum_bridge.py
└── eval_raum.py
```
