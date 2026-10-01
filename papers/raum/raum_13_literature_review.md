# Literature Review: Recursive Semantic-to-Geometric Decomposition for Text-to-3D

## 1. Compositional 3D Generation

**GALA3D** (Zhou et al., 2024) generates complex scenes by first using an LLM to plan a layout (object list + bounding boxes), then generating each object independently via Score Distillation Sampling (SDS), and compositing them into a unified Gaussian splat scene. The compositional planning is explicit but stops at the object level. Each object is generated as an opaque blob from a 2D diffusion prior. There is no internal part structure.

**SceneWiz3D** (Zhang et al., 2024) extends compositional generation to room-scale scenes with constraints (contact, support, occlusion). Objects are generated independently and placed via a physics-aware optimizer. Again, composition operates at the inter-object level. Intra-object structure is delegated entirely to the diffusion model's implicit knowledge.

**Set-the-Scene** (Cohen-Bar et al., 2023) focuses on arranging pre-generated objects in a scene by optimizing their 6-DoF poses to match a text description. It takes generated NeRF objects as inputs and finds placements. No generation of internal structure.

**CompoNeRF** (Lin et al., 2024) composes neural radiance fields from parts but uses a fixed, predefined part vocabulary (e.g., "seat", "leg", "back" for chairs). The decomposition is not learned from language; it is hard-coded per category.

**Gap:** All compositional methods treat composition as inter-object arrangement. None recursively decompose objects into semantically meaningful sub-parts, and none produce the sub-parts themselves as structured Gaussian splat clusters.

## 2. Gaussian-Native Text-to-3D

**GaussianDreamer** (Yi et al., 2024) initializes a 3D Gaussian field from a text prompt using a combination of 3D diffusion priors (Point-E or Shap-E for coarse initialization) and 2D SDS for refinement. The output is a set of Gaussian splats, but they are optimized holistically with no part structure.

**DreamGaussian** (Tang et al., 2023) generates textured 3D meshes via Gaussian splatting in under 2 minutes. Uses SDS with a progressive densification strategy. Extremely fast but produces monolithic objects. No compositional hierarchy.

**LucidDreamer** (Liang et al., 2024) improves multi-view consistency in SDS-based Gaussian generation using an Interval Score Matching loss. Addresses the Janus problem (multi-face artifacts) but remains at the whole-object level.

**GSGEN** (Chen et al., 2024) generates 3D Gaussians in two stages: coarse geometry from Point-E, then refinement via SDS against a 2D diffusion model. Produces high-quality single objects but has no mechanism for compositional decomposition.

**Gap:** All Gaussian-native methods produce flat (unstructured) sets of splats for whole objects. The splats have no semantic labels, no hierarchy, and no correspondence to named parts. Our approach would produce a tree where every internal node carries a semantic label and every leaf is a small, semantically coherent cluster of splats.

## 3. Part-Based and Hierarchical Shape Generation

**PartNeRF** (Tertikas et al., 2023) learns to generate shapes as compositions of part-level NeRFs. Each part is a small implicit function, and the whole is their union. Parts are discovered unsupervised via a reconstruction objective. However, parts are geometric primitives (roughly convex regions), not semantic concepts. "Leg" and "seat" emerge sometimes but are not guaranteed.

**Neural Parts** (Paschalidou et al., 2021) represents shapes as unions of learned implicit part primitives. Similar to PartNeRF but earlier. Parts are volumetric primitives fit to minimize reconstruction error. No language grounding.

**BSP-Net** (Chen et al., 2020) decomposes shapes into convex parts via Binary Space Partitioning. The decomposition is purely geometric (halfplane intersections). It produces interpretable parts but they have no semantic correspondence to language concepts.

**StructureNet** (Mo et al., 2019) learns hierarchical graph structures for shapes (e.g., chair = {back, seat, legs}). Closest to our vision in that it maintains a part hierarchy. However, the hierarchy is category-specific, learned from annotated PartNet data, and generation produces point clouds not Gaussian splats. It cannot generalize to open-vocabulary text prompts.

**Gap:** Part-based methods either discover geometric (not semantic) parts, or require category-specific supervised data. None connect a language model's compositional understanding to the part hierarchy. None use Gaussian splats as the rendering primitive at the part level.

## 4. Hierarchical Discrete Representations

**VQ-VAE** (van den Oord et al., 2017) and its descendants (VQ-VAE-2, dVAE) learn discrete codebooks at multiple spatial scales. Top-level codes capture global structure; bottom-level codes capture local detail. This is analogous to our multi-scale decomposition but operates in latent space, not in interpretable semantic space.

**Multi-scale codebooks** (Razavi et al., 2019, VQ-VAE-2) demonstrate that hierarchical discrete bottlenecks improve generation quality. The top level captures category and pose; the bottom level captures texture and fine geometry. A parallel to our composition tree where upper nodes are abstract concepts and leaves are concrete splat parameters.

**Residual VQ** (Zeghidour et al., 2021, SoundStream) applies hierarchical quantization to audio. Each codebook level refines the previous. The recursion is fixed-depth (8 or 16 levels), not content-adaptive. Our recursive decomposition would be variable-depth based on semantic complexity ("sphere" terminates at depth 1; "gothic cathedral" may recurse to depth 5+).

**Gap:** Hierarchical codebook methods demonstrate that multi-scale discrete decomposition works, but their "codes" are opaque (no semantic interpretation) and the depth is fixed. Our approach makes every level semantically interpretable (each node has a natural-language label) and the depth is content-adaptive.

## 5. Language-Guided Planning and Decomposition

**SayCan** (Ahn et al., 2022) uses an LLM to decompose high-level instructions into sequences of robot-executable primitives. The LLM plans; a value function scores feasibility. Direct analogue to our decomposer: language model outputs structured actions, a downstream system executes them. The difference: SayCan's primitives are motor actions, ours are geometric splat clusters.

**Code-as-Policies** (Liang et al., 2023) prompts an LLM to write Python code that composes pre-defined spatial primitives (move, grasp, place). The LLM does compositional reasoning over a fixed function library. Similar to our vision of the LM emitting a composition tree over a fixed primitive vocabulary.

**ProgPrompt** (Singh et al., 2023) generates executable programs from natural language for embodied agents. Programs have hierarchical structure (subroutines, loops). Shows that LLMs produce reliable hierarchical decompositions when the output format is well-defined.

**3D-GPT** (Sun et al., 2024) uses an LLM pipeline to generate procedural 3D content via Blender Python scripts. The LLM decomposes a scene description into procedural modeling operations. Closest existing work to our concept, but the "primitives" are Blender operations (add_cube, extrude, bevel), not Gaussian splats, and there is no learned renderer.

**Gap:** Language-guided decomposition is well-established in robotics and procedural 3D. Nobody has applied it where the terminal primitives are differentiable Gaussian splats that can be trained end-to-end. The key novelty is closing the loop: the decomposer can be trained via gradients flowing back from a differentiable Gaussian renderer through the composition tree.

## 6. Recursive and Fractal Neural Representations

**Fractal representations** in neural networks are rare. Recursive neural networks (TreeRNN, Socher et al., 2011) process tree-structured inputs but do not generate tree-structured 3D outputs. Fractal Network (Larsson et al., 2017) uses fractal-like layer connectivity for image classification but is not related to 3D generation.

**DeepSDF + recursive detail** (Park et al., 2019 and follow-ups) implicitly represents shapes at multiple levels of detail via a single MLP conditioned on different resolution queries. Not explicitly recursive or tree-structured.

**Octree-based neural representations** (OctField, Martel et al., 2021) use octree spatial subdivisions to allocate neural capacity adaptively. Spatial recursion (subdivide where detail is needed) parallels our semantic recursion (decompose where concept complexity is high). However, octrees are axis-aligned spatial splits with no semantic meaning at each level.

**Gap:** No existing work combines recursive (tree-structured, variable-depth) decomposition with semantic labeling at every level and Gaussian splat rendering at the leaves. The closest structural analogues (octrees) lack semantic grounding; the closest semantic analogues (SayCan, 3D-GPT) lack differentiable rendering.

## Gap Analysis: What Is Novel

Our proposed approach, recursive semantic-to-geometric decomposition for text-to-3D Gaussian splatting, occupies a unique intersection:

1. **Semantic hierarchy, not geometric.** Unlike PartNeRF, BSP-Net, or octrees, every node in our tree has a natural-language label. The decomposition is driven by conceptual complexity, not spatial extent.

2. **Variable-depth recursion, not fixed.** Unlike VQ-VAE hierarchies (fixed 2-3 levels) or SceneWiz3D (flat object lists), recursion depth adapts to concept complexity. "Ball" terminates immediately. "Medieval castle with a moat" recurses 4-5 levels.

3. **Language model as decomposer, not as prompt encoder.** Unlike GaussianDreamer or DreamGaussian (which encode text into a fixed conditioning vector), our LM actively reasons about structure, producing a tree of sub-concepts with spatial relations.

4. **Gaussian splats as universal primitive.** Unlike 3D-GPT (Blender operations) or SayCan (motor primitives), our terminal primitives are differentiable Gaussian splats. This enables end-to-end training through the composition tree via the differentiable Gaussian renderer.

5. **Separation of compositional reasoning and class discrimination.** Unlike our own Raum 1.2 (which tried to use the LM for both), we factor the problem: LM for decomposition structure, word vectors (GloVe) for terminal-level class discrimination. This avoids the subword tokenization collision problem entirely.

The result, if successful, would be the first system where a language model's compositional understanding directly generates hierarchical 3D structure, with Gaussian splats as the universal primitive at every scale of the hierarchy.
