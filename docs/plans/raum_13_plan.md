# Raum 1.3: Recursive Semantic-to-Geometric Decomposition

**Status:** open (architecture design phase)
**Swimlane:** 6 (Raum model)
**Depends on:** Raum 1.2 findings (subword tokenizer incompatible with class routing)
**Product counterpart:** 10-raum-0-3

## Architecture (text diagram)

```
prompt ("a castle on a hill")
    |
    v
[Planck Decomposer] ── frozen Planck 1.3 encoder + learned decomposition head
    |
    v
Composition Tree:
    castle_scene
    ├── castle
    │   ├── tower (x4, corners)
    │   │   ├── base (cylinder)
    │   │   ├── body (tapered)
    │   │   └── crenellations (repeating)
    │   ├── gate (arched, front)
    │   ├── keep (center, tall)
    │   └── curtain_wall (connecting)
    └── hill (below, sloped)
        |
        v
[Recursive Expansion] ── each non-terminal node re-enters decomposer
        |
        v
Terminal nodes (primitives)
        |
        v
[GloVe + Primitive Renderer] ── GloVe embedding → splat parameters
        |
        v
Positioned, colored, oriented Gaussian splats
```

## Three Components

### 1. Planck as Decomposer

The frozen Planck 1.3 encoder provides contextual token representations. A new
learned decomposition head (lightweight transformer or MLP) takes these
representations and outputs a composition tree. Each tree node contains:

- A concept label (natural language)
- Spatial relation to parent (relative position, scale, orientation)
- Terminal flag (is this a primitive or does it decompose further?)
- Variation parameters (for blob library lookup)

The decomposer runs recursively: non-terminal nodes re-enter the head with
their concept label as input, producing child nodes. The Planck encoder
provides contextual understanding (it knows "tower" in "castle" context is
different from "tower" in "cell tower" context). GloVe provides the terminal
discrimination at leaves.

### 2. Blob Library as Distribution over Compositions

The existing Objaverse 300-class blob library evolves from a flat lookup table
into a distribution over compositions. Each blob becomes a template composition
tree (a "tower" blob might encode: base + body + top, with default proportions).
Traversal of the tree produces variation: changing proportions, adding/removing
optional sub-parts, scaling.

This means the blob library is no longer just "what does a tower look like as
a point cloud" but "what is the typical decomposition of a tower and what are
its parameters." Blobs become generative programs, not static geometry.

### 3. GloVe + Primitive Renderer at Leaf Level

At the leaves of the composition tree, we reach terminal concepts simple enough
to render directly. Each terminal node's concept label is embedded via GloVe
(not Planck, to avoid the subword collision problem) and mapped to Gaussian
splat parameters: position offset from parent, covariance (shape/orientation),
color, opacity.

A small MLP (concept embedding + spatial context from parent chain) produces
the final splat parameters. This is the differentiable rendering endpoint.

## Open Questions

1. **What is a "primitive"?** How do we define when a concept is simple enough
   to stop recursing? Options: (a) fixed depth limit, (b) learned terminal
   classifier, (c) concept complexity score from the decomposer.

2. **How deep does recursion go?** Unbounded recursion risks explosion.
   Practical limit is likely 3-5 levels for most scenes. Need to validate
   empirically whether depth > 3 adds meaningful geometric detail.

3. **How to train the decomposer?** Ground-truth composition trees do not exist
   at scale. Options: (a) synthetic data from hand-authored trees, (b) LLM-
   generated supervision (use GPT-4 to produce decomposition trees, train
   Planck head to reproduce them), (c) end-to-end through the renderer (reward
   = reconstruction quality, but credit assignment through the tree is hard).

4. **Differentiability through the tree.** If tree structure is discrete
   (branching decisions), gradients do not flow through structure choices. May
   need to separate structure learning (reinforcement/supervision) from
   parameter learning (gradient descent on splat params).

5. **Scaling.** A 5-level tree with branching factor 4 produces ~1000 leaf
   nodes, each generating a few splats. Total: 3000-5000 splats per scene.
   Is this enough for visual quality? Too many for training efficiency?

## Scope

This is a multi-week research track, not a quick iteration. Expected phases:

- Phase A: Decomposer head design + synthetic tree data generation (1 week)
- Phase B: Single-level decomposition (prompt to parts, no recursion) (1 week)
- Phase C: Recursive expansion + terminal renderer (1-2 weeks)
- Phase D: End-to-end training pipeline + quality evaluation (1 week)

Total estimate: 4-6 weeks. Product demo (10-raum-0-3) follows after Phase D
gates pass.

## References

- Literature review: `docs/papers/raum_13_literature_review.md`
- Raum 1.2 failure analysis: `SETUP_202605.md` §4.5
- Prior art in this repo: Raum 1.0 (template routing), Raum 1.1 (frozen encoder
  bridge), Raum 1.2 (subword collision diagnosis)
