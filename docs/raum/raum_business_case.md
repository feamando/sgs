# Raum: Business Case

**Radiance Labs, 2026-06-04**
**Status:** Draft for internal decision (build / fund / partner). One-author working doc.

## Purpose

Decide whether Raum justifies continued investment as a commercial product, not just a research track. Raum turns natural-language prompts into **structured, editable 3D scenes** (a labelled tree of Gaussian-splat parts) rather than the monolithic blobs that today's text-to-3D systems produce. This doc states the opportunity, the wedge, the economics, and the gates that would kill or greenlight it.

Confidence tags below follow the FPF convention: **CL1** = proven/measured in-repo, **CL2** = strong external evidence, **CL3** = reasoned estimate, **CL4** = speculative.

## Thesis (one paragraph)

Every text-to-3D system generates a sealed radiance field with no parts, so every edit costs a full re-optimization (30-60 GPU-min) and no output is reusable or queryable. Raum decomposes the prompt into a semantic tree where each named part is an independently editable set of Gaussians, so edits are O(1) and scenes are composable and machine-queryable. The same alpha-compositing math that renders the scene is a **formally proven strict superset of softmax attention** (submitted to JMLR), giving one substrate for language and geometry. The wedge is not "prettier 3D" but **editable, structured 3D as an API** for pipelines that today pay humans to reconstruct structure after generation.

## Problem (why this is worth money)

Current monolithic text-to-3D (DreamFusion, Magic3D, and successors) fails three downstream needs:

- **Editability.** Change the roof colour, move the hill, swap a tower: each needs a full re-optimization pass (30-60 GPU-min/edit). Iteration is economically prohibitive. (CL2, well-documented in the field)
- **Compositionality.** No concept of "parts," so no reuse, no component library, no mix-and-match across scenes. Every output starts from scratch. (CL2)
- **Reasoning/queryability.** "What is left of the gate?" is unanswerable: there is no gate, only a gradient field. Game engines, robotics planners, and accessibility tools must rebuild the scene graph by hand, negating the time savings generation promised. (CL2)

Reported studio cost today: **$50-200 per usable scene** with manual cleanup, $50/A100-hr-class compute, 1-4 GPU-hr per single-object generation. (CL3, from secondary reports; verify before quoting externally)

## Solution and what's actually built

Raum = Recursive Semantic-to-Geometric Decomposition. Three components on one math framework:

1. **Decomposer** (LM): prompt to composition tree (named nodes, transforms, primitive types).
2. **Blob library**: per-category Gaussian distributions (procedural now, learned next).
3. **Terminal renderer**: flatten tree to Gaussians, differentiable splatting, with tree provenance so a clicked pixel maps back to its node.

**Proven / measured today (CL1, in-repo):**
- End-to-end pipeline operational: prompt to tree to render. CLI high-fidelity path fixed 2026-06.
- Hand-authored scenes render correctly (castle, village, pirate ship, space station, dragon), ~1,000-1,400 Gaussians each; Raum 1.4 scales to ~50K with subdivision + densification + SGS-native refinement.
- SGS-native refinement beats external on surface coverage (+21%) and compactness.
- Export to `.ply` / `.splat` / `.obj` (UE5, Unity, Blender, web viewers).
- Flat-routing prototype: 100% object-class accuracy on its validation set.
- 109 Sketchfab architecture templates; 3,450 LLM-generated training trees (~$5).
- Editability is structural: any node modifiable without re-inference (O(1) edit).
- Underlying theorem (alpha-compositing supersets softmax) machine-verified in Lean 4, submitted to JMLR.

**Honest gaps (CL1):**
- 100M decomposer is too small for open-vocabulary decomposition; it routes known classes, doesn't invent decompositions for unseen concepts. Needs ~1B scale (Hertz).
- Geometry is procedural primitives, recognizable not photorealistic. Photorealism needs learned blobs (ShapeNet/Objaverse).
- No real-time CUDA rasterizer integrated yet (web point-cloud viewer only).
- No large (prompt, tree) dataset exists; building it is itself a contribution.

## Market and where Raum wins

| Segment | Pain Raum removes | Why structured 3D specifically | Confidence |
|---|---|---|---|
| **Game asset / level prototyping** | Weeks of manual modelling per environment | Parts are editable GameObjects with semantic metadata | CL3 |
| **ArchViz** | Client iterates on CAD, not language | Named elements map to geometry; describe-to-see review loop | CL3 |
| **Film/VFX pre-viz** | 2-5 days/sequence manual blocking | Auto-blocked, labelled, repositionable entities | CL3 |
| **Education / sci-comm** | Static diagrams; no manipulable labelled parts | Tree = persistent labels through rotation; accessibility read-aloud | CL3 |
| **Robotics scene understanding** | Needs a scene graph, not a voxel field | Decomposition *is* the scene graph from language | CL4 (earliest, hardest) |

**Sizing (treat as directional, CL3):** global game-asset market cited at >$3B/yr; a 10% efficiency gain in environment creation implies a ~$300M served opportunity in that one segment. This is a top-down figure and must be rebuilt bottom-up (target studios x scenes/yr x price/scene) before any external use.

**The defensible wedge:** not generation quality (Google/Meta can outspend on that) but **editability + structured output + API ergonomics**. The moat is the data flywheel: every generated-then-edited tree is a training signal for the next decomposer, and the composition-tree format can become a de facto interchange standard.

## Business model

- **API (primary):** prompt to {composition tree JSON + rendered image/point cloud/glTF}. Per-scene pricing, volume tiers for studios.
- **Engine SDK:** Unity/Unreal plugins instantiate trees as native, editable scene graphs.
- **Enterprise/on-prem:** self-hosted for studios with IP concerns or defense; includes custom blob libraries trained on proprietary catalogs.

## Economics

- **Capital to beta API: under $500K** (CL3), dominated by GPU compute for the ~1B Hertz decomposer (~500 A100-hr initial train + fine-tune + serving). Capital-efficient vs competitors reportedly burning $10M+ on monolithic, un-editable approaches.
- **Compute structurally light (CL1/CL2):** SGS inherits transformer infra (FlashAttention-class kernels, AdamW recipes); Planck 105M runs in-browser. We do not invent new training infrastructure.
- **Unit economics (CL3, to be validated):** if a scene costs cents of inference and replaces $50-200 of human cleanup, gross margin per API call is high; the question is volume and conversion, not per-unit viability.

## Roadmap (capability gates, not dates)

| Gate | What must be true | Kills the case if... |
|---|---|---|
| **G1 Decomposer generalizes** | ~1B model produces valid trees for unseen prompts | It can't decompose beyond trained categories even with fallback editing |
| **G2 Learned geometry** | Blobs from ShapeNet/Objaverse beat procedural on realism | Output stays "clearly synthetic" and buyers won't pay |
| **G3 Real-time render** | gsplat-class backend, ~60fps @1080p, 10K+ Gaussians | Latency makes the API unusable in a creative loop |
| **G4 Pilot pull** | A design/studio partner pays for editable-3D-as-API | No one will pay a premium over free monolithic tools |

Stated timeline (CL3): Hertz decomposer Q3 2026, beta API Q4 2026, public API Q1 2027, engine SDK Q2 2027, enterprise Q3 2027.

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Hertz decomposer fails to generalize | High | Graceful degrade: user edits imperfect tree via DSL before render. Value retained at every capability level. |
| Incumbent (Google/Meta) ships structured 3D first | High | Compete on editability + API ergonomics + format standard, not raw generation quality. |
| Market timing / buyers not ready | Medium | Land in one beachhead (game prototyping or sci-comm), prove pull before broadening. |
| Photorealism gap deters buyers | Medium | Target segments where *structure + speed* beats fidelity (pre-viz, education, prototyping). |
| Single-founder / capacity | Medium | Scope to one beachhead; partner for go-to-market rather than build a sales org. |

## Recommendation

**Conditional go, beachhead-first.** The research foundation is real (CL1) and the theoretical moat is unusually strong for a pre-product company (proven theorem under peer review). But the commercial case is unproven at the two gates that matter: **G1 (does a 1B decomposer actually generalize?)** and **G4 (will a real buyer pay for editable-3D-as-API?)**.

Sequence:
1. Pick **one beachhead** (recommend game/indie prototyping or sci-comm/education, where structure+speed beats photorealism).
2. Train the Hertz decomposer and clear **G1** on that beachhead's prompt distribution, not the open world.
3. Put the existing demo + API stub in front of **3-5 design partners** to test **G4** willingness-to-pay before any SDK or enterprise build.
4. Only scale to engine SDK / enterprise after G1 and G4 are green.

This keeps spend under the ~$500K beta envelope, defers the expensive surfaces until pull is proven, and exploits the one durable asset competitors lack: a formally proven, structured, editable representation.

## Open items before external use

- Rebuild the >$3B / $300M sizing bottom-up; do not quote top-down externally.
- Verify the $50-200/scene and $10M+ competitor figures against primary sources.
- Confirm Hertz training cost (~500 A100-hr) against a current quote.
- Name the beachhead and one lighthouse partner.
