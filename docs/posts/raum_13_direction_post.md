# Recursive Semantic-to-Geometric Decomposition: A New Direction for Text-to-3D

Most text-to-3D systems stop at the object level. You say "chair" and get a chair. You say "castle" and get a castle-shaped mesh. The internal structure, the compositional hierarchy that makes a castle a castle (towers, gates, walls, keeps, battlements), is either implicit in a diffusion prior or absent entirely.

We are exploring a different approach: recursive semantic-to-geometric decomposition, where a language model acts as a structural decomposer and individual Gaussian splats are the terminal primitives at every scale.

The architecture has three components.

First, a contextual decomposer. A small language model (our Planck encoder, 100M params) takes a text prompt and produces a composition tree. "A castle on a hill" decomposes into spatial sub-concepts: tower (x4, corners), gate (front, arched), keep (center, tall), curtain wall (connecting), hill (below, sloped). Each node carries spatial relations to its siblings.

Second, recursive expansion. Each non-terminal node decomposes further. "Tower" becomes base (cylindrical), body (tapered cylinder), crenellations (repeating blocks), window slits (narrow voids). Recursion continues until nodes reach a "primitive" threshold, concepts simple enough to be rendered directly as small clusters of Gaussian splats.

Third, terminal rendering. At the leaves, GloVe word vectors (not subword LM embeddings, for reasons documented in our Raum 1.2 findings) provide discriminative features that map to splat parameters: position, covariance, color, opacity. Each leaf concept becomes a handful of positioned, oriented, colored Gaussians.

This connects directly to the SGS (Semantic Gaussian Splatting) thesis: if Gaussians are the atomic primitive of visual representation, they should be the atomic primitive at every scale of composition. A castle is not one blob. It is a tree of sub-concepts bottoming out at thousands of individually meaningful splats.

How does this compare to existing work? GALA3D and SceneWiz3D compose scenes from objects but treat each object as an opaque generation call via Score Distillation Sampling. DreamGaussian and GaussianDreamer produce Gaussian splats from text but operate at the whole-object level with no compositional structure. PartNeRF and BSP-Net decompose shapes into parts, but the parts are geometric (convexes, implicit surfaces), not semantic.

Nobody, to our knowledge, uses a language model as a recursive decomposer that bottoms out at individual Gaussian splats. The closest analogues are in robotics: SayCan and Code-as-Policies use LLMs for task decomposition, but their "primitives" are robot actions, not geometric elements.

Open questions remain substantial. What constitutes a "primitive" concept? How deep should decomposition go before quality degrades? How do we train the decomposer when ground-truth composition trees do not exist at scale?

This is a research direction, not a shipped product. We expect Raum 1.3 to be a multi-week track with significant architecture exploration. The Raum 1.2 findings (subword tokenization is wrong for word-level discrimination, but language models excel at compositional reasoning) give us confidence that this separation of concerns, LM for structure, word vectors for discrimination, is the right factorization.

The demo, when it ships, will be a web app where you type a scene, watch the composition tree expand in real time, and see Gaussian splats populate the 3D viewport level by level.
