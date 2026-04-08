# Article 3: What Is a Radiance Field?

*How computers learned to represent the 3D world as a continuous field of light*

---

## The Problem: Capturing Reality

Imagine you photograph a coffee mug from 50 different angles. You now have 50 flat images. But the mug is a 3D object — it exists in space, has shape, material, reflections that change depending on where you look from.

**The question:** Given those 50 photographs, can a computer build a representation of the mug so complete that it can generate a *new* photograph from any angle you choose — even one you never captured?

This is the **novel view synthesis** problem. Solving it requires the computer to understand not just what the mug looks like, but how light interacts with it in 3D space.

---

## The Naive Approach: 3D Meshes

Traditionally, 3D objects are represented as **meshes** — surfaces made of triangles. Think of a wireframe model. Each triangle has a color and material property. To render a new view, you project the triangles onto the camera's image plane and shade them.

This works, but:
- Meshes are hard to reconstruct from photos automatically
- Fine details (fur, hair, fog, glass) are extremely difficult to mesh
- The representation is **discrete** — a fixed number of triangles, no continuous interpolation

---

## The Breakthrough: Neural Radiance Fields (NeRF, 2020)

In 2020, Mildenhall et al. proposed a radical alternative: **don't build a surface at all.** Instead, represent the entire 3D space as a continuous **field**.

At every point in 3D space (x, y, z), the field stores two properties:
1. **Density (σ):** How opaque is the space here? High density = solid object. Low density = air.
2. **Color (c):** What color does this point emit *when viewed from a specific direction*?

The color depends on viewing direction because real surfaces look different from different angles. A shiny coffee mug has a highlight that moves as you walk around it. A matte surface looks the same from every angle.

Formally:

```
F(x, y, z, θ, φ) → (color, density)
```

Where (x, y, z) is the 3D position and (θ, φ) is the viewing direction.

This function F is represented as a neural network — a small MLP (multilayer perceptron) with ~5M parameters. You feed it a 3D coordinate and a viewing direction, and it outputs a color and density.

**This is the radiance field.** It's a continuous function defined over all of 3D space that encodes what the world looks like from any position in any direction.

---

## How NeRF Renders an Image

To produce an image from a specific camera position, NeRF uses **volume rendering** — a technique from physics for how light travels through semi-transparent media (like fog or clouds).

For each pixel in the output image:

1. **Cast a ray** from the camera through the pixel into the 3D scene
2. **Sample points** along the ray (say, 64-256 points)
3. **Query the neural network** at each sample point → get color and density
4. **Composite** the samples front-to-back using the volume rendering equation:

```
Color of pixel = Σᵢ Tᵢ · αᵢ · cᵢ
```

Where:
- cᵢ is the color at sample point i
- αᵢ is the opacity (derived from density — denser regions are more opaque)
- Tᵢ is the **transmittance** — how much light has NOT been absorbed by all the points in front of point i

```
Tᵢ = ∏ⱼ₌₁ⁱ⁻¹ (1 − αⱼ)
```

**Transmittance is the key concept.** It models occlusion: if something opaque is in front, it blocks what's behind. The first opaque surface along the ray absorbs most of the light; anything behind it contributes little.

Think of looking through a sequence of colored glass panes. The first pane tints the light. The second pane further tints what's left. By the time light passes through several panes, the last ones barely contribute — most of the light was already absorbed.

---

## NeRF's Training: Learning From Photos

The beautiful part: you train NeRF using only the photographs. For each training image:

1. For each pixel, cast a ray through the known camera position
2. Render the predicted color using the volume rendering equation
3. Compare to the actual pixel color in the photograph
4. Backpropagate the error to update the neural network

The neural network learns to assign the right density and color at every point in 3D space so that the rendered views match the photographs. After training, you can render from any new viewpoint — the network has internalized the 3D structure.

---

## NeRF's Problem: It's Slow

NeRF produces stunning results but is painfully slow:
- **Training:** Hours on a GPU
- **Rendering:** Each pixel requires querying the neural network hundreds of times (one per sample point along the ray). A 1080p image = 2 million pixels × ~200 samples = 400 million network evaluations. This takes seconds per frame — far from real-time.
- **The representation is implicit:** The scene is "locked inside" the neural network weights. You can't easily move, delete, or edit individual objects.

---

## The Revolution: 3D Gaussian Splatting (2023)

In 2023, Kerbl et al. asked: **what if we replace the implicit neural field with an explicit collection of primitives?**

Instead of a neural network that you query at arbitrary points, represent the scene as a collection of **3D Gaussian ellipsoids** — literally, blobs floating in space.

Each Gaussian is defined by:

| Parameter | What It Means | Intuition |
|---|---|---|
| **Position (μ)** | Where the blob is centered | "This part of the coffee mug is at (2.1, 0.5, 1.3)" |
| **Covariance (Σ)** | The shape and size of the blob | A flat disc for a surface, a round sphere for a corner, a stretched ellipsoid for an edge |
| **Opacity (α)** | How opaque the blob is | 1.0 = solid surface, 0.1 = slight haze |
| **Color (SH coefficients)** | What color the blob appears from different angles | Spherical harmonics encode view-dependent appearance (highlights, sheen) |

A typical scene contains 1-5 million of these Gaussians. Together, they approximate the continuous radiance field:

```
Scene = {G₁, G₂, G₃, ..., G₅₀₀₀₀₀₀}
```

Each Gᵢ is a small blob. Where blobs overlap, their contributions blend. Where there are no blobs, there's empty space (air).

---

## How Gaussian Splatting Renders an Image

Instead of casting rays and sampling a neural network (NeRF's approach), Gaussian Splatting works in reverse — it **projects** each 3D Gaussian onto the 2D image:

1. **Project:** Each 3D Gaussian is transformed into a 2D Gaussian (an ellipse) on the image plane, using the camera's projection matrix. This is called **splatting** — you "splat" the 3D blob onto the 2D screen.

2. **Sort:** The 2D Gaussians are sorted by depth (distance from camera), front to back.

3. **Composite:** For each pixel, blend the overlapping Gaussians front-to-back:

```
Pixel color = Σᵢ cᵢ · αᵢ · Gᵢ(pixel) · Tᵢ
```

Where Gᵢ(pixel) evaluates the 2D Gaussian at that pixel position (high value if the pixel is near the Gaussian's center, low if far away), and Tᵢ is the accumulated transmittance from all Gaussians in front.

This is the **same volume rendering equation** as NeRF — but instead of sampling a continuous field along rays, we're compositing a finite set of explicit primitives.

---

## Why 3DGS Is Better Than NeRF

| Property | NeRF | 3D Gaussian Splatting |
|---|---|---|
| **Speed** | Seconds per frame | 100-200+ FPS (1000x faster) |
| **Representation** | Implicit (inside neural network weights) | Explicit (each Gaussian is a named, inspectable object) |
| **Editability** | Very hard — change the scene, retrain the network | Easy — move, delete, add individual Gaussians |
| **Training** | Hours | 20-45 minutes |
| **Quality** | Excellent | Matches or beats NeRF |

The speed comes from the rendering approach: splatting (projecting blobs onto the screen) is embarrassingly parallel and maps perfectly to GPU hardware. NeRF's ray marching requires sequential sampling along each ray.

The interpretability comes from the explicit representation: you can point at a Gaussian and say "this blob represents the handle of the coffee mug." In NeRF, the handle is distributed across millions of network weights with no clear decomposition.

---

## How 3DGS Learns: Adaptive Density Control

During training, 3DGS starts from a sparse set of Gaussians (from initial structure-from-motion) and optimizes them to match training photographs. But it also **grows and shrinks** the set:

- **Splitting:** A large Gaussian in a region with high error → split into two smaller Gaussians (add detail where needed)
- **Cloning:** A small Gaussian in a region with high error → clone it and nudge the copy (add coverage)
- **Pruning:** A Gaussian with near-zero opacity → remove it (it contributes nothing)

This produces a self-organizing representation: more Gaussians in detailed regions (textures, edges, fine geometry), fewer in simple regions (flat walls, sky). The model **allocates its representational capacity** where it's most needed.

---

## The Key Concepts to Carry Forward

From this article, the concepts that matter for the language analogy:

1. **A radiance field is a continuous function over space** that encodes appearance and density at every point. It's a "meaning landscape" — but for light instead of words.

2. **Gaussians are explicit, interpretable primitives** that approximate this field. Each one has a position, shape, weight, and features. You can inspect them individually.

3. **The rendering equation composes overlapping Gaussians** into a final image via alpha-blending with transmittance. This is a specific mathematical operation for combining contributions from multiple sources, accounting for occlusion.

4. **Adaptive density control** lets the model grow and shrink its primitive set during training, self-organizing to match the data.

5. **View-dependent appearance** means the same scene looks different from different angles — the color of a Gaussian changes depending on where you're looking from.

In the next article, we'll look closely at the rendering equation itself — the mathematical heart of splatting — because this is the operation we propose to repurpose for language.

---

*Next: [Article 4 — The Rendering Equation: How Splats Become Images](04-rendering-equation.md)*
