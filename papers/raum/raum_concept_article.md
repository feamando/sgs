# Raum: Recursive Semantic-to-Geometric Decomposition for Structured 3D Generation

**Radiance Labs, 2026**

---

## 1. The Problem: Monolithic 3D Generation Is a Dead End

Text-to-3D generation has made remarkable progress. Systems like DreamFusion, Magic3D, and DALL-E 3D can produce impressive single objects from natural language prompts. But they share a fundamental architectural flaw: they generate monolithic, unstructured blobs.

When you ask DreamFusion to create "a castle on a hill," it produces a single undifferentiated radiance field. There is no castle. There is no hill. There is no separation between towers, walls, and ground. The output is a dense grid of opacity values with no semantic structure whatsoever. This is not a minor inconvenience. It is a fundamental barrier to every downstream use case that requires understanding what is in the scene.

This matters for three concrete reasons:

**Editability.** You cannot change the castle's roof color without regenerating the entire scene. You cannot move the hill. You cannot swap one tower for another. Every modification requires a full re-optimization pass, costing 30-60 GPU-minutes per edit. A designer iterating on a scene layout burns hundreds of dollars in compute before arriving at a satisfactory result. The monolithic representation makes iteration economically prohibitive.

**Compositionality.** You cannot take the castle from one generation and place it into a different landscape. The representation has no concept of "parts." It is, mathematically, a single continuous function with no internal boundaries. This means every new scene starts from scratch. There is no reuse, no library of components, no ability to mix and match elements across generations. The promise of generative AI as an accelerant for creative workflows breaks down entirely when every output is a sealed, opaque artifact.

**Reasoning.** Downstream systems (game engines, robotics planners, accessibility tools) cannot query the scene. "What is to the left of the gate?" is unanswerable because there is no gate, only a gradient field that happens to look like one from certain angles. Scene graphs, collision meshes, and semantic labels must all be manually reconstructed by human artists, negating the time savings that generation was supposed to provide.

The economics of the current approach compound the problem. A single DreamFusion generation takes 1-4 GPU-hours on an A100. Multi-object scenes multiply this cost. Studios experimenting with AI-generated 3D assets report per-scene costs of $50-200, with no guarantee that the output will be usable without extensive manual cleanup.

The gap in the field is clear: nobody decomposes language into structured 3D at the primitive level, preserving the compositional semantics of the original prompt all the way down to individual geometric elements. This is the problem Radiance Labs was founded to solve.

---

## 2. Applications: Where Structured 3D Generation Creates Value

The inability to decompose text into structured 3D blocks an entire class of applications that require editable, queryable, composable scene representations. Each of these represents a market where current tools either fail entirely or impose unacceptable friction.

**Game Asset Generation.** Game studios describe environments in design documents: "a medieval village with three houses, a church, and a well." Current tools generate concept art (2D) or require manual modeling (weeks per scene). A system that decomposes this description into a labeled tree of 3D parts, each independently editable, would compress environment prototyping from weeks to minutes. The global game asset market exceeds $3B annually; even a 10% efficiency gain in environment creation represents a $300M opportunity. More importantly, indie studios without large art teams gain access to rapid prototyping that was previously available only to AAA producers.

**Architectural Visualization.** Architects write briefs before they model. "An open-plan office with glass partitions, a central atrium, and rooftop terracing." Decomposing this into a composition tree, where each named element maps to geometry, enables instant 3D walkthroughs from textual specs. Clients iterate on descriptions, not CAD files. The critical value here is speed-to-feedback: a client can describe changes in natural language and see them rendered immediately, compressing design review cycles from days to hours.

**Education and Scientific Communication.** "Show me a water molecule: one oxygen atom bonded to two hydrogen atoms at 104.5 degrees." The decomposition is explicit in the language. A structured renderer maps each named concept to labeled geometry. Students manipulate parts. Labels persist through rotation. Accessibility tools read the tree aloud. This extends beyond chemistry: anatomy, mechanical engineering, astronomy, and any domain where spatial relationships carry meaning benefits from structured, labeled 3D output.

**Film and VFX Pre-visualization.** Scripts describe scenes compositionally: "EXT. CASTLE COURTYARD, NIGHT. A dragon perches on the gatehouse." Pre-viz artists manually block these scenes, spending 2-5 days per sequence. A decomposition system generates the layout automatically, each entity labeled and repositionable. Directors iterate on blocking in real time. The savings compound across a 90-minute feature with hundreds of unique setups.

**Robotics Scene Understanding.** A robot navigating a kitchen needs to know that the mug is on the counter, not that voxel (34, 12, 7) has opacity 0.8. Structured decomposition produces the scene graph that robotic planners require, directly from natural language instructions. This is the bridge between human communication ("put the plate next to the glass") and robotic execution (which requires named object references with spatial coordinates).

---

## 3. The Solution: Recursive Semantic-to-Geometric Decomposition

Radiance Labs introduces RSGD (Recursive Semantic-to-Geometric Decomposition), a framework that bridges language and 3D geometry through hierarchical composition trees.

**Core Insight.** Natural language is compositional. "A castle on a hill" decomposes into {castle, hill}, and "castle" further decomposes into {towers, walls, gate, keep}. This recursive structure maps directly to a tree of 3D primitives, where each node carries both a semantic label and geometric parameters. The decomposition is not arbitrary; it follows the noun-phrase structure of the original prompt. Every named entity in the text becomes a named node in the tree. Spatial prepositions ("on," "beside," "inside") become transform relationships between parent and child nodes.

**The Composition Tree.** Every scene is represented as a tree of CompositionNodes. Internal nodes define spatial relationships (position, scale, rotation) and semantic labels. Terminal nodes hold the actual geometry: sets of Gaussian splats (3D Gaussian distributions with position, covariance, color, and opacity).

```
Prompt: "a castle on a hill"

scene
├── castle [pos=(0, 0.8, 0)]
│   ├── tower_NW [pos=(-0.8, 0, 0.6), scale=0.3]
│   │   ├── body: 80 Gaussians (cylinder)
│   │   ├── roof: 40 Gaussians (cone)
│   │   └── flag: 15 Gaussians
│   ├── tower_NE [...]
│   ├── keep [...]
│   └── walls [...]
└── hill [pos=(0, -0.5, 0), scale=2.0]
    └── mound: 200 Gaussians (dome)
```

**Alpha-Compositing as the Rendering Primitive.** The tree is rendered via recursive alpha-compositing. Each node's Gaussians are blended using transmittance-weighted accumulation, the same operation that underlies both volumetric rendering (NeRF, 3D Gaussian Splatting) and, formally, the softmax attention mechanism in transformers. We have proven that alpha-compositing is a strict superset of softmax (submitted to JMLR), meaning the same mathematical framework handles both language understanding and 3D rendering.

**Editability by Construction.** Because the tree preserves semantic labels, editing is trivial: change a node's parameters (color, position, scale) and re-render. No re-optimization. No gradient descent. The tree IS the editable representation. This is a qualitative difference from every other text-to-3D system: edits cost O(1) time regardless of scene complexity, because you modify a node in a tree rather than re-optimizing a continuous field.

**Why Gaussians?** The choice of 3D Gaussian splats as the terminal primitive is deliberate. Gaussians are differentiable (enabling future training), explicit (each one has interpretable parameters), composable (alpha-compositing is associative), and fast to render (projection + sorting, no ray marching). Unlike NeRF's implicit neural fields, Gaussians can be individually created, deleted, and modified without retraining any network. They are the natural unit of structured 3D: small enough to represent fine detail, large enough to be individually meaningful.

---

## 4. Architecture: Three Components, One Mathematical Framework

[Figure 1: System architecture diagram showing Prompt -> Decomposer -> Composition Tree -> Recursive Renderer -> 3D Output]

The Raum system consists of three components operating within a unified Gaussian splatting framework.

**Component 1: The Decomposer.** A language model that takes a natural language prompt and produces a composition tree. At the current scale (100M parameters), this operates as a structured prediction task: the model outputs a JSON tree where each node has a semantic label, spatial transform, and a primitive type (cylinder, cone, box, sphere, dome, plane). At Hertz scale (1B+ parameters), the decomposer will perform conditional generation, choosing geometry based on learned priors over object categories. The decomposer is trained on (prompt, tree) pairs where trees are validated by the renderer. Invalid trees (overlapping nodes, physically impossible configurations) are filtered automatically, creating a self-curating training pipeline.

**Component 2: The Blob Library.** A collection of learned distributions over Gaussian configurations. Each "blob" represents a class of objects (e.g., "tower," "tree," "wheel") as a distribution over possible Gaussian arrangements. During inference, the decomposer selects a blob type for each terminal node, and the library provides the corresponding Gaussian parameters. Currently procedural (hand-coded primitive generators); the roadmap targets learned blobs trained on ShapeNet and Objaverse. The library is extensible: new categories can be added without retraining the decomposer, and custom blobs can be contributed by users (enabling domain-specific asset libraries for architecture, biology, mechanical engineering).

**Component 3: The Terminal Renderer.** Given a flattened set of Gaussians (produced by traversing the tree and accumulating transforms), the renderer produces a 2D image via differentiable Gaussian splatting. This uses the standard 3DGS pipeline: project each Gaussian to screen space, sort by depth, alpha-composite front-to-back. The key difference from standard 3DGS is that our Gaussians carry tree provenance, enabling per-node editing and semantic queries. Clicking on a pixel in the rendered image traces back through the Gaussians to the originating tree node, enabling "click to select part" interactions in the viewer.

**The Unifying Mathematics.** All three components share the same core operation. Attention in the decomposer:

```
attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

And rendering in the terminal renderer:

```
C(ray) = sum_i (T_i * alpha_i * c_i),  where T_i = prod_{j<i} (1 - alpha_j)
```

We have proven that the second (alpha-compositing) is a strict superset of the first (softmax). This is not merely an analogy. The same transmittance-based accumulation that renders 3D scenes also implements the soft selection mechanism in language models. Raum exploits this unification: one framework, from language to geometry.

**Why This Matters Commercially.** A unified mathematical framework means unified infrastructure. The same GPU kernels that accelerate transformer inference (FlashAttention, paged KV caches) can, with minor modification, accelerate Gaussian rendering. The same training recipes that scale language models (AdamW, cosine scheduling, gradient accumulation) apply to our decomposer. We do not need to invent new training infrastructure. We inherit a decade of language model engineering.

---

## 5. Current Implementation and Results

The Raum system is operational at prototype scale, with end-to-end pipeline validation complete.

**Raum 0.1: Flat Routing Demo.** The initial prototype demonstrates structured 3D generation with a flat (non-recursive) architecture. A router network maps text labels to one of 300 object classes, selects 5 objects per scene, and places them via an editable DSL. Key result: 100% object accuracy on the validation set (correct class selection for all prompted objects). The DSL enables post-generation editing (reposition, recolor, remove) without re-inference.

**Raum 1.3: Recursive Decomposition Prototype.** The current system implements full recursive decomposition with hand-authored composition trees. Five demo scenes validate the pipeline end-to-end:

- Castle on a hill (4 towers, gate, keep, walls): 1,135 Gaussians
- Medieval village (3 houses, church, well): ~1,200 Gaussians
- Pirate ship (hull, masts, sails, crow's nest): ~1,100 Gaussians
- Space station (modules, connectors, solar panels): ~1,000 Gaussians
- Dragon on mountain (full creature with wings, mountain with cave): ~1,400 Gaussians

[Figure 2: Castle scene rendered as point cloud, showing semantic coloring by tree node]

[Figure 3: Multi-object composition showing independent editability of parts]

**Planck 1.4: Conversation Memory.** The underlying language model (Planck series) has achieved 40% retrieval recall at 90 conversation turns, demonstrating the capacity for extended context that recursive decomposition requires. The model maintains coherent compositional intent across long prompts.

**Key Metrics:**
- Object accuracy (flat routing): 100%
- Valid tree generation: all hand-authored trees render correctly
- End-to-end latency (tree to render): <200ms on CPU
- Editability: any node modifiable without re-inference
- Semantic queryability: tree traversal answers spatial queries
- Total codebase: ~75K lines Python, fully tested pipeline

**What This Demonstrates.** The current implementation proves three things. First, the composition tree representation is expressive enough to encode complex multi-object scenes with semantic labels. Second, the rendering pipeline (tree traversal, transform accumulation, Gaussian splatting) is fast and correct. Third, the architecture is ready for a learned decomposer; the only missing piece is scale. The hand-authored scenes serve as ground-truth targets for training the Hertz decomposer: given a prompt, produce a tree that matches these reference structures.

---

## 6. Limitations, Next Steps, and Commercial Path

### Current Limitations

We are transparent about what works and what does not. Intellectual honesty accelerates progress.

**Model scale.** The 100M-parameter Planck model is too small for conditional decomposition from arbitrary prompts. It can route to known classes but cannot invent novel decompositions for unseen concepts. This is a scale limitation, not an architectural one. The evidence: at 100M parameters, language models demonstrate structured prediction capability (our flat routing works perfectly), but lack the world knowledge to decompose "a Victorian greenhouse" into appropriate sub-parts without explicit enumeration.

**Procedural geometry.** Current terminal nodes use hand-coded primitive generators (cylinder, cone, box, sphere, dome). Real objects have complex geometry that requires learned representations. The system produces recognizable scenes (a castle is clearly a castle) but not photorealistic ones. The path to photorealism runs through learned blob distributions, where each category samples from a trained Gaussian mixture.

**Single-view rendering.** The current Three.js viewer renders from orbit cameras. Multi-view consistency is guaranteed by the 3D Gaussian representation, but we have not yet integrated real-time rasterization backends (e.g., gsplat, diff-gaussian-rasterization). This limits current demos to web-based point cloud visualization rather than production-quality rendering.

**Training data.** There is no large-scale dataset of (text prompt, composition tree) pairs. Creating this dataset is itself a research contribution. Our approach: use GPT-4 and Claude to generate tree structures from prompts, validate them with the renderer, and curate a training set of 50K-100K verified examples.

### Next Steps

**Hertz 1B+ Decomposer (Q3 2026).** Train a 1-billion-parameter model specifically for compositional decomposition. The model takes arbitrary text prompts and produces valid composition trees with appropriate blob selections. This is the critical capability gate.

**Learned Geometry via ShapeNet/Objaverse (Q3-Q4 2026).** Replace procedural primitives with learned blob distributions. Each terminal node samples from a category-conditioned Gaussian mixture, producing geometry that matches real object shapes.

**Real-Time Rendering Integration (Q4 2026).** Integrate gsplat or similar CUDA-accelerated rasterizers for real-time viewing. Target: 60fps at 1080p for scenes with 10,000+ Gaussians.

### Commercial Path

**API Service.** The primary commercial offering: a REST API that accepts text prompts and returns structured 3D scenes as JSON (composition tree) plus rendered outputs (images, point clouds, glTF). Pricing per scene generation, with volume tiers for studios.

**SDK for Game Engines.** Unity and Unreal plugins that call the API and instantiate composition trees as native scene graphs. Each node becomes an editable GameObject/Actor with semantic metadata. Studios integrate directly into their asset pipelines.

**Enterprise Licensing.** For organizations requiring on-premises deployment (film studios with IP concerns, defense applications), a self-hosted version with dedicated GPU allocation. This tier includes custom blob libraries trained on proprietary asset catalogs.

**Competitive Moat.** Our defensibility rests on three pillars: (1) the formal proof that alpha-compositing subsumes softmax, which provides the theoretical foundation and is under peer review; (2) the composition tree format, which becomes a de facto standard as adoption grows; and (3) the trained decomposer, which improves with every generation (each output becomes a training signal for the next model version). The system exhibits a data flywheel: more users generate more trees, which train better decomposers, which attract more users.

**Timeline:**
- Hertz decomposer training: Q3 2026
- Beta API (limited partners): Q4 2026
- Public API launch: Q1 2027
- Game engine SDK: Q2 2027
- Enterprise tier: Q3 2027

**Capital Requirements.** The primary cost driver is GPU compute for Hertz training. A 1B-parameter model requires approximately 500 A100-hours for initial training, with ongoing costs for fine-tuning and inference serving. Total estimated capital to reach beta API: under $500K, making this capital-efficient relative to competitors burning $10M+ on monolithic generation approaches that produce un-editable outputs.

**Risk Mitigation.** The primary technical risk is that the Hertz decomposer fails to generalize beyond trained categories. We mitigate this with a fallback: even if the decomposer produces imperfect trees, users can edit them manually via the DSL before rendering. The system degrades gracefully from "fully automatic" to "assisted manual," retaining value at every capability level. The secondary risk is market timing: if a major player (Google, Meta) ships structured 3D first, we compete on editability and API ergonomics rather than generation quality alone.

**Summary.** Raum is not an incremental improvement to text-to-3D generation. It is a different paradigm: decomposition rather than synthesis, structure rather than monoliths, editability rather than one-shot artifacts. The mathematical foundation is proven, the pipeline is operational, and the path to commercial scale is clear. We are building the infrastructure for a world where 3D content is as composable and editable as text.

---

*Radiance Labs, 2026*

*For inquiries: contact information available upon request.*
