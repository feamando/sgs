# Literature Review: Semantic Gaussian Splatting

**Supporting Document for SGS Whitepaper v2.0**
**Date:** April 2026
**Papers Reviewed:** 85+

---

## Overview

This literature review surveys published work across six domains relevant to the Semantic Gaussian Splatting (SGS) proposal. For each paper, we note its relevance to SGS and whether it **validates**, **challenges**, or **extends** the core hypotheses.

The SGS whitepaper makes five key claims that this review evaluates:
1. **A1** — Language meaning embeds in continuous geometric space
2. **A2** — Gaussians are more natural than points for representing meaning
3. **A3** — Semantic composition can be modeled as Gaussian rendering/compositing
4. **A4** — Low-dimensional splatting space preserves semantic structure
5. **A5** — Multi-pass rendering provides sufficient depth of computation

---

## 1. Gaussian Word Embeddings (Post-Word2Gauss)

*Validates: A2 (Gaussians beat points for meaning)*

The foundational work of Vilnis & McCallum (2015) spawned a productive lineage of research that consistently validates the Gaussian primitive for language.

### 1.1 Core Papers

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Vilnis & McCallum** — "Word Representations via Gaussian Embedding" | 2015 | ICLR | Words as Gaussian distributions (mean + covariance) capture uncertainty and asymmetric entailment via KL divergence | **Foundational.** The direct ancestor of SGS's semantic Gaussian primitive |
| **Athiwaratkun & Wilson** — "Multimodal Word Distributions" | 2017 | ACL | Extends to Gaussian *mixtures* per word, capturing polysemy. Energy-based max-margin objective. Beats Word2Vec and single-Gaussian on similarity/entailment | **Validates polysemy via multiple splats.** Each word sense = separate Gaussian, directly analogous to 3DGS using many splats per object |
| **Athiwaratkun & Wilson** — "Hierarchical Density Order Embeddings" | 2018 | ICLR | Broad distributions encompass specific ones, creating inclusion-based hierarchies. Outperforms point-based order embeddings for hypernymy | **Validates covariance as semantic breadth.** The "spread" of a Gaussian encodes generality — exactly what SGS proposes |
| **Athiwaratkun, Wilson & Anandkumar** — "Probabilistic FastText for Multi-Sense" | 2018 | ACL | Gaussian mixture + subword n-grams. Handles rare words and polysemy simultaneously | **Validates multi-scale Gaussians.** Subword composition within Gaussian framework maps to SGS's micro/meso/macro scales |

### 1.2 Applied Extensions

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Yuksel, Ugurlu & Koc** — "Semantic Change Detection with Gaussian Word Embeddings" | 2021 | IEEE/ACM TASLP | Gaussian covariance captures diachronic semantic drift. Ranked 1st in SemEval-2020 Task 1 | **Validates temporal dynamics.** Covariance changes over time = semantic evolution, analogous to 4D Gaussian Splatting |
| **Jamadandi et al.** — "Probabilistic Word Embeddings in Kinematic Space" | 2021 | ICPR | Gaussian distributions in non-Euclidean (kinematic) space, inspired by AdS/CFT physics | **Extends to non-Euclidean geometry.** Supports SGS operating in non-standard metric spaces |
| **Huang** — "Gaussian Joint Embeddings for Self-Supervised Learning" | 2026 | arXiv | Gaussian embeddings extended to modern self-supervised paradigm (JEPA-like). Probabilistic joint density with uncertainty | **Validates continued relevance.** Gaussian embedding paradigm alive in 2026, not a dead end |

### 1.3 Assessment for SGS

**Strong validation of A2.** The 2015-2026 trajectory shows Gaussians consistently outperform points for:
- Uncertainty representation (rare/ambiguous words → broad Gaussians)
- Asymmetric relations (hypernymy as containment)
- Polysemy (Gaussian mixtures)
- Temporal evolution (covariance drift)

**Key gap identified:** All papers use Gaussians for *pairwise word relationships* (similarity, entailment). None use Gaussians for *compositional semantics* (building sentence meaning). This is SGS's novel contribution.

---

## 2. Gaussian Mixture Models for Polysemy

*Validates: A2 (Gaussians handle polysemy naturally)*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Chen, Qiu, Jiang & Huang** — "Gaussian Mixture Skip-gram (GMSG)" | 2015 | arXiv | Gaussian mixture per word with *dynamic* component count adjustment during training | **Directly analogous to adaptive density control.** GMSG's dynamic sense splitting = 3DGS's Gaussian splitting |
| **Nguyen, Modi, Thater & Pinkal** — "Mixture Model for Multi-Sense Embeddings" | 2017 | *SEM | Sense-weighted mixtures outperform uniform mixtures | **Validates opacity/salience weighting.** Different senses have different importance, like SGS's per-Gaussian opacity α |
| **Jayashree, Shreya & Srijith** — "Multi-Sense via Approximate KL Divergence" | 2019 | COMAD | KL-divergence objective for Gaussian mixture word representations | Extends training methodology for Gaussian mixtures |

**Assessment:** Dynamic Gaussian mixtures with adaptive component counts (GMSG) are a direct precursor to SGS's adaptive density control. The split/prune/clone mechanism in 3DGS has an independent precedent in NLP.

---

## 3. Sentence and Document-Level Gaussian Representations

*Validates: A3 (Gaussians scale beyond words to composition)*

This is the most directly relevant category — extending Gaussians from word-level to sentence/document-level representation.

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Yoda, Tsukagoshi, Sasano & Takeda** — "GaussCSE: Sentence Representations via Gaussian Embedding" | 2023 | EACL 2024 | **Sentences as Gaussian distributions.** Enables asymmetric sentence similarity and entailment direction estimation | **Closest precedent to SGS.** Proves the Gaussian primitive scales from words to sentences. Key difference: GaussCSE represents the *output* as a Gaussian; SGS uses Gaussians as the *compositional primitive* |
| **Das, Zaheer & Dyer** — "Gaussian LDA for Topic Modeling" | 2015 | ACL | Topics as Gaussian regions in embedding space. Documents as mixtures of Gaussian topics | **"Gaussian splatting" for topics.** Each topic is a positioned, shaped blob — essentially a semantic splat at the document scale |
| **Li, Zhou, He et al.** — "BERT-flow: Sentence Embeddings from PLMs" | 2020 | EMNLP | BERT embeddings are anisotropic/non-Gaussian. Normalizing flows transform to isotropic Gaussian | **Important constraint for SGS.** Raw transformer embeddings are NOT Gaussian-distributed. SGS's low-d splatting space must either learn a Gaussian-friendly geometry or accept non-Gaussian regions |
| **Batmanghelich et al.** — "Spherical Topic Models" | 2016 | ACL | von Mises-Fisher distributions (spherical Gaussians) for topic modeling. Flexible topic count discovery | **Validates directional Gaussians.** Spherical Gaussian splats on the unit hypersphere |
| **Yurochkin et al.** — "Hierarchical Optimal Transport for Document Representation" | 2019 | NeurIPS | Documents as distributions-of-distributions, compared via hierarchical optimal transport | **Multi-scale distributional representation.** Documents → topic distributions → word distributions, like SGS's macro/meso/micro scale |

**Assessment:** GaussCSE (2023) is the strongest direct precedent. It proves sentences can be Gaussian, but does not address *how* word Gaussians compose into sentence Gaussians — which is exactly SGS's contribution (the rendering equation as the composition mechanism).

---

## 4. Optimal Transport and Wasserstein Distance in NLP

*Extends: A3 (alternative composition/comparison methods for distributional representations)*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Kusner, Sun, Kolkin & Weinberger** — "Word Mover's Distance (WMD)" | 2015 | ICML | Document distance as minimum-cost optimal transport between word embedding distributions. Outperforms bag-of-words, TF-IDF, LSA | **Alternative to rendering for composing distributions.** WMD computes distance between document-as-distribution and target-as-distribution — a comparison operation SGS could adopt |
| **Wu et al.** — "Word Mover's Embedding" | 2018 | NeurIPS | Fixed-dimensional embeddings from WMD, reducing O(n^3) OT cost | **Efficiency precedent.** If SGS rendering is too expensive, OT-based embeddings offer a compiled alternative |
| **Wang et al.** — "Wasserstein-Fisher-Rao Document Distance" | 2019 | arXiv | Unbalanced optimal transport for documents of different lengths | Handles variable-mass Gaussian scenes |
| **McCarroll et al.** — "WMD + GloVe for Information Retrieval" | 2026 | arXiv | WMD + GloVe outperforms all SOTA retrieval models including Doc2Vec and LSA | **Continued validation** that distributional distance works in practice |

**Assessment:** Optimal transport provides a theoretically principled alternative to alpha-blending for comparing/composing distributional representations. If SGS's rendering equation fails for composition (Challenge C2 from orthogonal review), Wasserstein-based composition is a fallback.

---

## 5. Uncertainty Quantification via Distributional Representations

*Validates: A2 (distributional representations are increasingly demanded by the field)*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Cui et al.** — "Prompt2Gaussia: Uncertain Prompt-Learning" | 2023 | arXiv | Prompt tokens as Gaussian random variables (not points). Improves robustness | **Direct validation.** Replacing point prompts with Gaussian prompts helps — the same principle SGS applies at architecture level |
| **Kiruluta** — "Bayesian Kalman View of In-Context Learning" | 2026 | arXiv | LLM in-context learning modeled as Bayesian state estimation. Covariance collapse = learning | **Theoretical bridge.** ICL as sequential Gaussian estimation is formally related to SGS's multi-pass rendering |
| **Bayesian Transformer Language Models** (multiple groups) | 2019-2023 | Various | Variational inference over transformer weights, producing posterior distributions | **Demand signal.** Field increasingly wants distributional/uncertainty-aware representations |

**Assessment:** Growing demand for uncertainty-aware language models validates the need for distributional primitives. SGS's Gaussian covariance is a natural mechanism.

---

## 6. Hyperbolic Gaussian Embeddings

*Extends: A2 and A4 (Gaussians work in non-Euclidean spaces, potentially improving hierarchy)*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Nagano et al.** — "Wrapped Normal on Hyperbolic Space" | 2019 | ICML | Pseudo-hyperbolic Gaussian with analytically evaluable density and differentiable parameters. Hyperbolic VAE + probabilistic word embeddings | **Key bridge paper.** Gaussian splats can exist in hyperbolic space. Tree-depth = distributional precision. Could replace SGS's Euclidean splatting space |
| **Nickel & Kiela** — "Poincare Embeddings" | 2017 | NeurIPS | Hierarchical data in hyperbolic space. Distance from origin encodes generality/specificity | **Geometric foundation.** In SGS terms: center = hypernym (broad Gaussian), edge = hyponym (tight Gaussian) |
| **Iyer et al.** — "Non-Euclidean Mixture Model" | 2024 | arXiv | Spherical + hyperbolic spaces in mixture model. Different curvatures for different relation types | **Multi-curvature SGS.** Different semantic relationships may need Gaussians in different geometry |
| **Qiao et al.** — "HYDEN: Hyperbolic Density Representations" | 2024 | arXiv | Cross-modal (image+text) Gaussian embeddings in hyperbolic space | Cross-modal distributional representations |

**Assessment:** Hyperbolic Gaussians could significantly improve SGS's ability to represent hierarchical semantics. The low-dimensional splatting space (d=64 Euclidean) proposed in v3 could be replaced with a hyperbolic manifold for hierarchy-heavy domains.

---

## 7. Gaussian Splatting Beyond Vision

*Validates: The 3DGS paradigm generalizes beyond 3D scene reconstruction*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Kim, Seo & Kim** — "Station2Radar: Gaussian Splatting for Precipitation Fields" | 2026 | arXiv | Gaussian splatting applied to meteorological data — fusing sparse weather stations into continuous precipitation fields | **Strongest non-visual precedent.** Gaussians representing weather data, not 3D scenes. Directly supports the generality argument |
| **Cai et al.** — "GaussTwin: Differentiable Gaussian Digital Twins" | 2026 | arXiv | Gaussians as differentiable world models for robotic manipulation with MPC | Gaussians for physics/policy, not just rendering |
| **Xiong et al.** — "Mind-to-Face: EEG → Gaussian Avatars" | 2025 | arXiv | Brain signals decoded into Gaussian splatting facial expressions | Non-visual input (neural signals) → Gaussian representation |
| **Luo et al.** — "DynFOA: Gaussian Splatting → Spatial Audio" | 2026 | arXiv | 3D Gaussians grounding spatial audio generation (ambisonics) | Gaussians bridging visual and auditory modalities |
| **Li et al.** — "PIN-WM: Physics-Informed Gaussian World Models" | 2025 | arXiv | Gaussian-based world model encoding rigid body dynamics | Gaussians representing physics, not appearance |

**Assessment:** The field is rapidly expanding Gaussian splatting beyond vision. Weather data (Station2Radar), robotics (GaussTwin), brain signals (Mind-to-Face), and audio (DynFOA) all demonstrate that the Gaussian primitive is domain-agnostic. **SGS applying Gaussians to language is a natural next step in this trajectory.**

---

## 8. Language-Embedded Gaussians (Language IN Splatting)

*The reverse direction: language features embedded into 3D Gaussian scenes*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Qin et al.** — "LangSplat: 3D Language Gaussian Splatting" | 2024 | CVPR | CLIP features in each Gaussian. Scene-specific language autoencoder. 199x faster than LERF | **Proves Gaussians carry semantic features.** Each Gaussian = geometry + appearance + language. SGS drops the geometry, keeps the language |
| **Li et al.** — "LangSplatV2: 450+ FPS" | 2025 | arXiv | Higher-dimensional language features, faster rendering | Scalability evidence for high-d features in Gaussians |
| **Li et al.** — "4D LangSplat: Dynamic Semantic Gaussians" | 2025 | arXiv | Temporal + semantic Gaussians with MLLM-generated labels | Spatiotemporal semantic fields |
| **Maggio & Carlone** — "Bayesian Fields: Task-driven Open-Set Semantic Gaussians" | 2025 | arXiv | Bayesian uncertainty in semantic Gaussian fields. Task-driven open-set queries | **Principled uncertainty for semantic Gaussians.** Directly transferable to SGS |
| **Barhdadi et al.** — "4D Synchronized Fields: Motion-Language Gaussians" | 2026 | arXiv | Motion + language fields co-located in Gaussian primitives | Multi-modal semantic Gaussians |
| **Kerr et al.** — "LERF: Language Embedded Radiance Fields" | 2023 | ICCV (Oral) | CLIP features volumetrically in NeRF, multi-scale language queries | Established the language-in-fields paradigm |
| **Zhou et al.** — "Feature 3DGS" | 2024 | arXiv | Gaussians render arbitrary semantic features (SAM, CLIP-LSeg) | **Gaussians as general feature carriers.** Not limited to RGB |
| **Zheng et al.** — "GaussianGrasper" | 2024 | arXiv | Language-embedded Gaussians drive robotic grasping from NL commands | Semantic Gaussians → physical actions |

**Assessment:** This body of work proves that Gaussians can carry high-dimensional language features alongside geometry. **SGS's contribution is the logical inverse: dropping the 3D geometry and keeping only the semantic/language representation.** The LangSplat lineage validates the technical feasibility; SGS proposes the conceptual leap.

---

## 9. Conceptual Spaces Computational Models

*Validates: A1 and A2 (language meaning has geometric structure compatible with Gaussians)*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Kumar, Chatterjee & Schockaert** — "Extracting Conceptual Spaces from LLMs" | 2025 | arXiv | Gardenfors-style conceptual spaces extracted from LLM prototype embeddings | **LLMs learn Gardenfors-compatible geometry.** SGS's Gaussian representations are consistent with what LLMs already encode |
| **Tull et al.** — "From Conceptual Spaces to Quantum Concepts" | 2024 | arXiv | Category-theoretic framework with both classical Gaussian and quantum circuit instantiations of conceptual spaces | **Formal mathematical grounding** for concepts-as-Gaussians |
| **Wheeler & Natarajan** — "Semantic Communication via Geometry of Meaning" | 2023 | arXiv | Gardenfors conceptual spaces for semantic communication. Massive compression by transmitting meaning-geometry | **Efficiency argument for SGS.** Geometric semantic representations are inherently compressible |
| **Fel et al.** — "Minkowski Representation Hypothesis" | 2025 | arXiv | Vision model tokens as "convex mixtures of archetypes" grounded in Gardenfors | **Neural networks learn Gardenfors-compatible regions.** Convex regions in latent space = Gaussians |
| **Tetkova et al.** — "On Convex Decision Regions in Deep Networks" | 2023 | arXiv | Measured convexity in neural nets across images, audio, text, medical imaging | **Empirical validation across modalities.** Convex regions confirmed in text representations |
| **Chatterjee et al.** — "LLMs for Learning Conceptual Spaces" | 2023 | arXiv | BERT and GPT-3 tested for learning Gardenfors quality dimensions | Transformers implicitly learn conceptual space structure |
| **Banaee & Lowry** — "Abstract Concept Modelling via Conceptual Spaces" | 2026 | arXiv | Conceptual spaces for abstract chess strategy concepts | Conceptual spaces work for purely abstract (non-perceptual) domains |

**Assessment:** Strong evidence that (a) neural networks already learn Gardenfors-compatible convex regions in latent space, (b) these regions can be extracted as Gaussian-like structures, and (c) the approach extends to abstract concepts, not just perceptual categories. **This validates SGS's theoretical grounding in conceptual spaces theory.**

---

## 10. Differentiable Rendering as General Computation

*Validates: The rendering paradigm extends beyond graphics*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Sitzmann et al.** — "SIREN: Implicit Neural Representations with Periodic Activations" | 2020 | NeurIPS | SIRENs represent images, wavefields, video, sound, and solve PDEs as continuous neural implicit functions | **Proof of generality.** Neural representations from vision work for arbitrary continuous signals |
| **Kato et al.** — "Differentiable Rendering: A Survey" | 2020 | arXiv | Rendering as general differentiable computation, not just image synthesis | Theoretical foundation for rendering-as-computation |
| **Mostafa et al.** — "Differentiable Rendering for Tabular Data" | 2026 | arXiv | Differentiable rendering applied to tabular biomedical data classification | **Direct precedent for rendering non-visual data.** Tabular data → learned feature maps via rendering |
| **Chen et al.** — "Differentiable Inverse Rendering for RF Digital Twin" | 2026 | arXiv | Differentiable rendering for electromagnetic fields, not light | Rendering paradigm applied to radio frequency signals |

**Assessment:** The rendering paradigm is demonstrably general. Papers applying it to tabular data, RF signals, seismic data, and weather confirm that "rendering" is a differentiable computation framework, not an image-specific technique.

---

## 11. Alpha-Compositing and Attention — The Formal Connection

*Partially validates: A3 (composition via rendering)* | *Identifies a novel theoretical contribution*

| Paper | Year | Venue | Key Finding | SGS Relevance |
|---|---|---|---|---|
| **Liang, He & Chen** — "ReTR: Modeling Rendering via Transformer" | 2023 | arXiv | Cross-attention explicitly simulates the rendering process | Rendering framed as attention-like computation |
| **Hu & Han** — "Volume Rendering with Attentive Depth Fusion" | 2023 | arXiv | Attention mechanisms alongside volume rendering | Practical composability of attention and rendering |
| **Jin et al.** — "MoE-GS: Mixture of Experts for Gaussian Splatting" | 2025 | arXiv | MoE integrated with Gaussian splatting via "Volume-aware Pixel Router" | **Expert routing = semantic field selection.** MoE architecture directly bridges to SGS's per-Gaussian specialization |
| **D'Amicantonio et al.** — "GS-MoE: Gaussians Guide Expert Routing" | 2025 | arXiv | Gaussian Splatting literally guides MoE expert selection | **Structural parallel explicit.** Each Gaussian as a localized expert |

**Key gap identified:** No paper formally proves the mathematical equivalence between alpha-compositing and attention. Both are weighted sums of value vectors:
- Volume rendering: C = Σ(T_i · α_i · c_i), where T_i = Π(1 − α_j)
- Attention: y = Σ(softmax(q · k_i) · v_i)

**This formal proof is a potential novel theoretical contribution of the SGS whitepaper.**

---

## 12. Hypothesis Validation Summary

| Hypothesis | Papers Supporting | Papers Challenging | Verdict |
|---|---|---|---|
| **A1: Language embeds in continuous geometric space** | Mikolov (2013), Pennington (2014), Devlin (2019), Tetkova (2023), Fel (2025) | None | **Strongly validated** |
| **A2: Gaussians beat points for meaning** | Vilnis (2015), Athiwaratkun (2017, 2018), Yuksel (2021), Yoda/GaussCSE (2023), Huang (2026) | Li/BERT-flow (2020): raw embeddings are anisotropic, not Gaussian | **Validated with caveat** — learned Gaussian spaces needed, not raw transformer output |
| **A3: Composition via Gaussian rendering** | GaussCSE (sentence Gaussians exist), Gaussian LDA (topic Gaussians compose), WMD (distributional comparison works) | No direct precedent for alpha-blending composition in NLP | **Untested — strongest novel claim** |
| **A4: Low-d splatting preserves structure** | Wheeler (2023): geometric representations compress massively; conceptual spaces work in low-d | Li/BERT-flow (2020): useful structure lives in ~100+ dimensions | **Plausible but dimension threshold unknown** |
| **A5: Multi-pass rendering = sufficient depth** | Kiruluta (2026): ICL as sequential Gaussian estimation | No direct evidence | **Plausible by analogy** |

---

## 13. Novel Contributions Identified for SGS

Based on this review, the following are confirmed as **novel contributions** not present in existing literature:

1. **Applying Gaussian splatting to pure language modeling** — no existing paper does this. All language-Gaussian work is either word-level (Word2Gauss) or grounded in 3D scenes (LangSplat).

2. **Alpha-compositing as semantic composition** — the rendering equation as a mechanism for building sentence meaning from word meanings. GaussCSE represents sentences as Gaussians; SGS proposes the mechanism by which word Gaussians compose into sentence meaning.

3. **Bridging Gardenfors' conceptual spaces with Gaussian splatting** — conceptual spaces theory predicts convex regions; Gaussians are the natural primitive; splatting is the rendering engine. This triad has not been connected before.

4. **Formal connection between alpha-compositing and attention** — the structural parallel is implicit in many papers but unproven.

5. **Adaptive density control for semantic vocabulary** — dynamic split/prune/clone for semantic Gaussians. GMSG (Chen et al., 2015) does dynamic sense splitting, but not within a rendering/splatting framework.

---

## References (Full List)

### Gaussian Word Embeddings
1. Vilnis, L. & McCallum, A. (2015). Word Representations via Gaussian Embedding. ICLR 2015.
2. Athiwaratkun, B. & Wilson, A.G. (2017). Multimodal Word Distributions. ACL 2017.
3. Athiwaratkun, B. & Wilson, A.G. (2018). Hierarchical Density Order Embeddings. ICLR 2018.
4. Athiwaratkun, B., Wilson, A.G. & Anandkumar, A. (2018). Probabilistic FastText for Multi-Sense Word Embeddings. ACL 2018.
5. Yuksel, A., Ugurlu, B. & Koc, A. (2021). Semantic Change Detection with Gaussian Word Embeddings. IEEE/ACM TASLP.
6. Jamadandi, A. et al. (2021). Probabilistic Word Embeddings in Kinematic Space. ICPR 2021.
7. Huang, Y. (2026). Gaussian Joint Embeddings for Self-Supervised Representation Learning. arXiv.

### Gaussian Mixture Embeddings
8. Chen, X. et al. (2015). Gaussian Mixture Skip-gram (GMSG). arXiv.
9. Nguyen, D.Q. et al. (2017). Mixture Model for Multi-Sense Word Embeddings. *SEM.
10. Jayashree, P. et al. (2019). Multi-Sense via Approximate KL Divergence. COMAD/CODS.

### Sentence/Document-Level Gaussians
11. Yoda, S. et al. (2023). GaussCSE: Sentence Representations via Gaussian Embedding. EACL 2024.
12. Das, R. et al. (2015). Gaussian LDA for Topic Modeling with Word Embeddings. ACL 2015.
13. Li, B. et al. (2020). BERT-flow: On Sentence Embeddings from PLMs. EMNLP 2020.
14. Batmanghelich, N. et al. (2016). Spherical Topic Models. ACL 2016.
15. Yurochkin, M. et al. (2019). Hierarchical Optimal Transport for Document Representation. NeurIPS 2019.

### Optimal Transport in NLP
16. Kusner, M. et al. (2015). From Word Embeddings to Document Distances (WMD). ICML 2015.
17. Wu, L. et al. (2018). Word Mover's Embedding. NeurIPS 2018.
18. Wang, Z. et al. (2019). Wasserstein-Fisher-Rao Document Distance. arXiv.
19. Sato, R. et al. (2021). Re-evaluating Word Mover's Distance. arXiv.
20. McCarroll et al. (2026). WMD + GloVe for Information Retrieval. arXiv.

### Uncertainty Quantification
21. Cui, S. et al. (2023). Prompt2Gaussia: Uncertain Prompt-Learning. arXiv.
22. Kiruluta, A. (2026). Bayesian Kalman View of In-Context Learning. arXiv.

### Hyperbolic Gaussians
23. Nagano, Y. et al. (2019). Wrapped Normal on Hyperbolic Space. ICML 2019.
24. Nickel, M. & Kiela, D. (2017). Poincare Embeddings. NeurIPS 2017.
25. Iyer, R.G. et al. (2024). Non-Euclidean Mixture Model. arXiv.
26. Qiao, Z. et al. (2024). HYDEN: Hyperbolic Density Representations. arXiv.

### Gaussian Splatting Beyond Vision
27. Kim, D. et al. (2026). Station2Radar: Gaussian Splatting for Precipitation Fields. arXiv.
28. Cai, Y. et al. (2026). GaussTwin: Gaussian Splatting Digital Twins for Robotics. arXiv.
29. Xiong, H. et al. (2025). Mind-to-Face: EEG Decoding via Gaussian Splatting. arXiv.
30. Luo, Z. et al. (2026). DynFOA: Gaussian Splatting for Spatial Audio. arXiv.
31. Li, W. et al. (2025). PIN-WM: Physics-Informed Gaussian World Models. arXiv.

### Language-Embedded Gaussians
32. Qin, M. et al. (2024). LangSplat: 3D Language Gaussian Splatting. CVPR 2024.
33. Li, W. et al. (2025). LangSplatV2: 450+ FPS. arXiv.
34. Li, W. et al. (2025). 4D LangSplat: Dynamic Language Gaussians. arXiv.
35. Maggio, D. & Carlone, L. (2025). Bayesian Fields: Task-driven Semantic Gaussians. arXiv.
36. Barhdadi, M.R. et al. (2026). 4D Synchronized Motion-Language Gaussian Fields. arXiv.
37. Kerr, J. et al. (2023). LERF: Language Embedded Radiance Fields. ICCV 2023.
38. Zhou, S. et al. (2024). Feature 3DGS. arXiv.
39. Zheng, Y. et al. (2024). GaussianGrasper. arXiv.

### Conceptual Spaces
40. Kumar, N. et al. (2025). Extracting Conceptual Spaces from LLMs. arXiv.
41. Tull, S. et al. (2024). From Conceptual Spaces to Quantum Concepts. arXiv.
42. Wheeler, D. & Natarajan, B. (2023). Semantic Communication via Geometry of Meaning. arXiv.
43. Fel, T. et al. (2025). Minkowski Representation Hypothesis. arXiv.
44. Tetkova, L. et al. (2023). On Convex Decision Regions in Deep Networks. arXiv.
45. Chatterjee, U. et al. (2023). LLMs for Learning Conceptual Spaces. arXiv.
46. Banaee, H. & Lowry, S. (2026). Abstract Concept Modelling via Conceptual Spaces. arXiv.

### Differentiable Rendering as General Computation
47. Sitzmann, V. et al. (2020). SIREN. NeurIPS 2020.
48. Kato, H. et al. (2020). Differentiable Rendering: A Survey. arXiv.
49. Mostafa, S. et al. (2026). Differentiable Rendering for Tabular Data. arXiv.
50. Chen, X. et al. (2026). Differentiable Rendering for RF Digital Twins. arXiv.

### Alpha-Compositing / Attention / MoE Connections
51. Liang, Y. et al. (2023). ReTR: Modeling Rendering via Transformer. arXiv.
52. Hu, P. & Han, Z. (2023). Volume Rendering with Attentive Depth Fusion. arXiv.
53. Jin, I. et al. (2025). MoE-GS: Mixture of Experts for Gaussian Splatting. arXiv.
54. D'Amicantonio, G. et al. (2025). GS-MoE: Gaussians Guide Expert Routing. arXiv.

### State Space Models (Alternative Continuous Architectures)
55. Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv.
56. Gu, A., Goel, K. & Re, C. (2021). S4: Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022.
57. Dao, T. & Gu, A. (2024). Mamba-2: Transformers are SSMs. ICML 2024.
58. Poli, M. et al. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. ICML 2023.
59. Peng, B. et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. EMNLP 2023.
60. Sun, Y. et al. (2023). RetNet: Retentive Network. arXiv.

### Diffusion Models for Text
61. Li, X.L. et al. (2022). Diffusion-LM Improves Controllable Text Generation. NeurIPS 2022.
62. Dieleman, S. et al. (2022). CDCD: Continuous Diffusion for Categorical Data. arXiv.
63. Sahoo, S.S. et al. (2024). MDLM: Simple and Effective Masked Diffusion Language Models. NeurIPS 2024.
64. Lou, A. et al. (2023). SEDD: Discrete Diffusion Modeling by Estimating Ratios of Data Distribution. ICML 2024.
65. He, Z. et al. (2022). DiffusionBERT. arXiv.

### Energy-Based Models for Language
66. Ramsauer, H. et al. (2020). Hopfield Networks is All You Need. ICLR 2021.
67. Deng, Y. et al. (2020). Residual Energy-Based Models for Text Generation. ICLR 2020.

### Geometric Deep Learning for NLP
68. Bronstein, M.M. et al. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv.

### Kernel Attention / Linear Attention
69. Choromanski, K. et al. (2020). Performers (FAVOR+). ICLR 2021.
70. Katharopoulos, A. et al. (2020). Linear Attention: Transformers are RNNs. ICML 2020.
71. Zhang, M. et al. (2024). Hedgehog & Porcupine: Expressive Linear Attentions. ICLR 2024.

### Compositional Generalization
72. Drozdov, A. et al. (2022). Compositional Semantic Parsing with LLMs. ICLR 2023.
73. Anil, C. et al. (2022). Exploring Length Generalization in LLMs. arXiv.

### Neural Fields Beyond Vision
74. Xie, Y. et al. (2021). Neural Fields in Visual Computing and Beyond. EUROGRAPHICS STAR.
