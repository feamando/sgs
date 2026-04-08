# Article 4: The Rendering Equation — How Splats Become Images

*The math of blending, and why it's more than just averaging*

---

## The Setup

You have a scene — thousands of overlapping Gaussian blobs floating in 3D space. You have a camera position. You need to produce a single color for each pixel. How?

This article walks through the rendering equation step by step, because this exact equation — transposed to a different domain — is the core of the SGS proposal.

---

## Step 1: Evaluate the Gaussians at the Pixel

For a given pixel, each Gaussian has a **contribution strength** at that pixel location. This is simply the Gaussian function evaluated at the pixel's position:

```
Gᵢ(x) = exp(-½ · (x - μᵢ)^T Σᵢ^{-1} (x - μᵢ))
```

- If the pixel x is near the Gaussian's center μᵢ → Gᵢ(x) ≈ 1 (strong contribution)
- If the pixel is far from the center → Gᵢ(x) ≈ 0 (negligible contribution)
- The covariance Σᵢ controls the shape: a large Σ means the Gaussian spreads widely; a small Σ means it's tight and concentrated

This evaluation is the **kernel** — it answers "how relevant is this Gaussian to this pixel?"

Think of it as a spotlight. Each Gaussian casts a soft, elliptical spotlight on the image. Where the spotlight is bright (near center), the Gaussian has strong influence. Where it's dim (far from center), it has little influence.

---

## Step 2: Why You Can't Just Average

The naive approach: add up all the Gaussian contributions, weighted by their kernel values.

```
Color(x) = Σᵢ Gᵢ(x) · colorᵢ / Σᵢ Gᵢ(x)    [weighted average — WRONG]
```

This is weighted averaging, and it produces **see-through** objects. A red ball in front of a blue wall would appear purple — an average of red and blue — instead of solid red blocking the blue wall.

Real scenes have **occlusion**: objects in front block objects behind. A proper rendering equation must model this. That's what transmittance does.

---

## Step 3: Transmittance — Modeling Occlusion

Imagine you're looking through a stack of colored glass panes. Each pane is partially transparent (has an opacity α between 0 and 1):

```
Pane 1: opacity 0.8, color red       ← closest to you
Pane 2: opacity 0.5, color green
Pane 3: opacity 0.9, color blue      ← farthest from you
```

How much of pane 3's blue color reaches your eye?

- Pane 1 blocks 80% of the light → 20% passes through (transmittance T₂ = 0.20)
- Pane 2 blocks 50% of what's left → 10% passes through (transmittance T₃ = 0.20 × 0.50 = 0.10)
- So pane 3 contributes its color scaled by 10%

The **transmittance** at position i is the fraction of light that has NOT been absorbed by all preceding panes:

```
T₁ = 1.0          (nothing in front — full transmittance)
T₂ = 1 - α₁       = 0.20     (80% blocked by pane 1)
T₃ = T₂ · (1-α₂)  = 0.10     (another 50% blocked by pane 2)
T₄ = T₃ · (1-α₃)  = 0.01     (90% blocked by pane 3)
```

General formula:
```
Tᵢ = ∏ⱼ₌₁ⁱ⁻¹ (1 − αⱼ)
```

T is always between 0 and 1. It can only decrease (or stay the same). Once all transmittance is absorbed (T → 0), nothing behind matters — total occlusion.

---

## Step 4: The Full Rendering Equation

Combining the kernel evaluation, opacity, and transmittance:

```
Color(x) = Σᵢ cᵢ · αᵢ · Gᵢ(x) · Tᵢ
```

For each Gaussian i, its contribution to the pixel is:

```
contribution = (its color) × (its opacity) × (how close it is to this pixel) × (how much light is left)
     cᵢ      ×     αᵢ     ×        Gᵢ(x)       ×           Tᵢ
```

**This is NOT a weighted average.** The weights don't sum to 1 (they sum to 1 - T_final). If some transmittance remains (T_final > 0), the pixel has "unaccounted" light — typically filled with a background color.

Let's trace through a concrete example:

```
Three Gaussians at a pixel, sorted front-to-back:
  G₁: color=red,   α=0.9, G₁(x)=0.8
  G₂: color=green, α=0.5, G₂(x)=0.6
  G₃: color=blue,  α=0.7, G₃(x)=0.3

Transmittances:
  T₁ = 1.0
  T₂ = 1 - 0.9×0.8 = 0.28
  T₃ = 0.28 × (1 - 0.5×0.6) = 0.28 × 0.70 = 0.196

Contributions:
  w₁ = 0.9 × 0.8 × 1.0   = 0.720 → 72.0% red
  w₂ = 0.5 × 0.6 × 0.28  = 0.084 → 8.4% green
  w₃ = 0.7 × 0.3 × 0.196 = 0.041 → 4.1% blue

Total: 0.720 + 0.084 + 0.041 = 0.845 (84.5% accounted for)
Remaining transmittance: 15.5% → background color
```

The red Gaussian dominates because it's in front, highly opaque, and close to the pixel. The green contributes a little. The blue barely registers — most light was absorbed before reaching it.

---

## Step 5: What Makes This Special

### Property 1: Order Matters

Swap the order — put blue in front and red in back — and you get a completely different color. The compositing is **not commutative**. This is physically correct: the order you see things depends on what's in front.

### Property 2: The Weights Are Bounded

The total contribution Σᵢ wᵢ is always ≤ 1. No single Gaussian can contribute more than its fair share. The output is naturally bounded — no need for explicit normalization (unlike softmax, which must divide by the sum of exponentials).

### Property 3: Locality

If a Gaussian is far from the pixel (Gᵢ(x) ≈ 0), its contribution is essentially zero — regardless of its opacity or transmittance. Only nearby Gaussians matter. This means you don't need to evaluate all N Gaussians for every pixel — just the ones whose kernels overlap with that pixel.

### Property 4: Every Parameter Has a Gradient

The rendering equation is differentiable. You can compute ∂Color/∂μᵢ (moving a Gaussian changes the output), ∂Color/∂αᵢ (changing opacity changes the output), ∂Color/∂Σᵢ (changing shape changes the output). This is what allows end-to-end training via gradient descent.

### Property 5: Transmittance Creates "Depth"

The transmittance mechanism means front Gaussians have disproportionate influence. But "front" is determined by the sorting order — and that order can be different for different pixels/queries. This creates a form of selective attention: different queries "see" the scene differently based on which Gaussians are "in front" from their perspective.

---

## Comparing to Attention

Let's put the rendering equation next to transformer attention:

```
Attention:  output = Σᵢ softmax(qᵀkᵢ/√d) · vᵢ
Rendering:  output = Σᵢ αᵢ · Gᵢ(x) · Tᵢ    · cᵢ
```

Both are **weighted sums of value vectors**. The difference is how the weights are computed:

| | Attention | Rendering |
|---|---|---|
| **Weight source** | Dot product between query and key, normalized by softmax | Gaussian kernel × opacity × transmittance |
| **Normalization** | Global — weights always sum to exactly 1 | Sequential — weights sum to at most 1 |
| **Scope** | Global — every token contributes to every other | Local — only nearby Gaussians contribute (kernel decay) |
| **Order sensitivity** | Via positional encoding (added externally) | Via transmittance (built into the composition) |
| **Interpretability** | Weight = "how much does token j attend to token i" | Weight = Gaussian proximity × remaining capacity |

The rendering equation is more structured than attention. It has built-in locality (Gaussians far away don't contribute), built-in ordering (transmittance creates depth), and built-in capacity management (transmittance depletes as you consume contributions). Attention is more flexible — any token can attend to any other with any weight — but this flexibility comes at the cost of O(n^2) computation and less interpretable weights.

---

## The Bridge to Language

Now imagine the same equation, but instead of:
- **Gaussians in 3D space** → **Gaussians in semantic space** (each representing a word/concept)
- **Pixel position** → **query position** (what meaning are we trying to extract?)
- **Color** → **semantic features** (rich meaning vectors)
- **Opacity** → **salience** (how important is this word?)
- **Depth sorting** → **sequence ordering** (word order in the sentence)
- **Transmittance** → **semantic capacity** (how much meaning is left to account for?)

The rendering equation becomes:

```
Meaning(q) = Σᵢ fᵢ · αᵢ · K(q, μᵢ, Σᵢ) · Tᵢ
```

This is the **Semantic Gaussian Splatting rendering equation** — and understanding exactly what it means and why it might work is the subject of the next article.

---

*Next: [Article 5 — The Leap: Radiance Fields for Meaning](05-radiance-fields-for-meaning.md)*
