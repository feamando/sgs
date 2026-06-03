# Radiance Labs Roadmap

Single source of truth for every **model** and **product** iteration we
ship. The swimlane visualizer at `pm/index.html` reads this file directly.

## Sync rules (read before editing)

- **Bidirectional sync**: if you bump a version in `SETUP.md`, you MUST
  add or update the matching row here. If you add a row here, update the
  relevant `SETUP.md` section so run instructions exist.
- **Versioning**:
  - **Major / minor** (e.g. `Planck 1.1`, `Raum 0.1`) are declared by the
    user manually when starting a new iteration.
  - **Fix versions** (e.g. `Planck 1.1.1`, `Raum 0.0.1`) are for small
    improvements landed within the same minor. Claude may add these
    automatically when doing bug-fix work.
- **Status values**:
  - `open` — planned but not started
  - `in progress` — actively being worked on
  - `done` — shipped (usually with a LinkedIn post)
- **Row format**: one pipe-table row per entry. Do not delete rows; move
  them to `done` and add a new row for the next version. History matters.
- **id**: `<swimlane-id>-<slug>`, kebab-case throughout (e.g.
  `1-planck-1-0`). Dots in version numbers become dashes.

## Swimlanes

| swimlane-id | name | kind | description |
|---|---|---|---|
| 1 | Planck | model | Small SGS language models (~100M params) |
| 2 | Hertz | model | Large SGS language models (~1B+ params) |
| 3 | Helmholtz | model | TBD |
| 5 | Klang | model | Audio synthesis via SGS |
| 6 | Raum | model | Text-to-3D Gaussian-splat bridge |
| 7 | Einstein | model | Frontier model (future) |
| 8 | Prisma | product | TBD |
| 9 | Klang | product | Audio demo / app |
| 10 | Raum | product | Text-to-3D web demo |
| 11 | Satz | product | Text demo: Planck LM + blob retrieval (local web app) |

## Entries

| id | name | type | status | date_created | notes |
|---|---|---|---|---|---|
| 1-planck-1-0 | Planck 1.0 | model | done | 2026-04-07 | Foundation 100M LM; baseline |
| 1-planck-1-1 | Planck 1.1 | model | done | 2026-04-14 | Validated blob concept |
| 1-planck-1-2 | Planck 1.2 | model | done | 2026-04-20 | Six-run ablation matrix shipped 2026-04-28; gate FAIL (val loss 2× worse, speedup 1.07×). Diagnoses at results/planck_12/README.md. Remediation tracked as Planck 1.2.1 |
| 1-planck-1-2-1 | Planck 1.2.1 | model | done | 2026-04-28 | Closed 2026-05-01 as FAIL alongside 1.2.2. Two-track accel remediation ran 2026-04-28; 4 of 8 runs crashed (scheduler TypeError, sk OOM, all collapsed). Root-cause bugs fixed in 1.2.2; see 1.2.2 row for the verdict |
| 1-planck-1-2-2 | Planck 1.2.2 | model | done | 2026-04-29 | Closed 2026-05-01 as FAIL. Three-bug fix pass landed (Optimizer subclass, flat index_select, additive tl floor + renorm) but the SGS-native compound still OOM'd (all_fix step 27.1k, all_plus_fix step 44.3k, sk_fix step 10k) because forward [B·H, L, k, d_f] activation is ~8 GiB bf16 × 3 passes and Windows shared-GPU-memory was spilling until system RAM exhausted. Muon regressed +0.48 nats at matched tokens. No gate passed. Decision: shelve the SGS-native accel track at 100M scale; Hertz 1.2 runs on plain AdamW. Learnings (Optimizer subclass, index_select, harness speedup mask) stay in the tree |
| 1-planck-1-3 | Planck 1.3 | model | done | 2026-04-27 | Shipped 2026-05-26: Wikipedia base retrain (1.3.0, PPL 24.96) + static Wikipedia blob index (1.3.1a, 200k blobs, 960 MB). Live RSS blob index (1.3.1b) deferred post-Hertz. Raum 0.1/1.1 use the 1.3.0 base as frozen token encoder. Details at docs/plans/planck_13_plan.md |
| 1-planck-1-4 | Planck 1.4 | model | done | 2026-04-27 | Shipped 2026-05-21: conversation-memory blobs. DynamicBlobStore + TurnEncoder + HybridRetriever + chat harness. Needle benchmark: 40% retrieval recall at 90+ turn distance (similarity-only), blob-memory retrieves facts truncation loses, wall-clock 5.99s vs 3.68s (within 2x). Cost-slope gate invalid at 512-token context (all modes saturate identically). Results at results/planck14_needle_findings.md |
| 2-hertz-1-0 | Hertz 1.0 | model | done | 2026-04-07 | Paused 2026-04-20, wall-clock infeasible without accel |
| 2-hertz-1-2 | Hertz 1.2 | model | open | 2026-04-20 | Large-LM run on plain AdamW (no accel recipe); unblocked 2026-05-01 after Planck 1.2.1/1.2.2 closed FAIL. Re-sequenced 2026-05-04 to run LAST, after Raum 0.1/1.1 + Satz 0.1 + Klang 1.3 remainder, so every shorter product track ships before GPU-weeks commit. ~10 days for 10B tokens on RTX 4090 at baseline throughput. Details in SETUP_202605.md §7 |
| 5-klang-1-0 | Klang 1.0 | model | done | 2026-04-10 | Initial audio-SGS concept + scene.py scaffold |
| 5-klang-1-1 | Klang 1.1 | model | done | 2026-04-15 | Variants A & B trained; phase warble + sub-200Hz dropout findings |
| 5-klang-1-2 | Klang 1.2 | model | done | 2026-04-20 | Complex-valued Gaussians, transmittance compositing, MRSTFT. Shipped 2026-04-28: Gate A (MSE 0.00156) + Gate B (log-MAE 1.378 vs Variant B 2.057) both pass. Bass dropout + Nyquist whine fixed. Still behind Klang 1.1 Variant A (stft_3000g) on absolute quality → 1.3 |
| 5-klang-1-3 | Klang 1.3 | model | open | 2026-04-28 | Scale Klang 1.2 architecture (complex Gaussians + transmittance + MRSTFT) to 1000-3000 gaussians to beat Klang 1.1 Variant A (stft_3000g, log-MAE 1.255, MCD 308.9). Re-sequenced 2026-05-04: 500g run shipped first and is validated against Variant A; the 1000/2000/3000g sweep is deferred until just before Hertz 1.2 so Raum can take foreground. If 500g already clears Variant A's gates, the swimlane closes without the rest of the sweep. Details in SETUP_202605.md §6 |
| 6-raum-1-0 | Raum 1.0 | model | done | 2026-04-24 | Template-routing bridge; 6-object hexad, 2-object scenes, analytic-label training. Shipped 2026-04-27 as backbone of Raum 0.0 demo |
| 6-raum-1-1 | Raum 1.1 | model | done | 2026-04-27 | Shipped 2026-05-08: frozen Planck 1.3 encoder + 5.15M bridge (d_model=256/6L/8H) + Objaverse 300-class blob library + relation head + DSL v1 + editable demo. 100% accuracy on procedural scenes. Details at docs/plans/raum_11_plan.md |
| 6-raum-1-2 | Raum 1.2 | model | done | 2026-05-09 | PARTIAL. Trained (50k samples, 100 epochs) but blob accuracy 0.1%. Root cause: SentencePiece subword tokenizer collapses related words (35 collision groups across 300 classes). Raw embedding lookup from a subword LM is architecturally wrong for word-level discrimination. Demo ships with GloVe path. Findings documented in SETUP_202605.md §4.5 |
| 6-raum-1-3 | Raum 1.3 | model | done | 2026-05-18 | Shipped 2026-05-21: recursive semantic-to-geometric decomposition. Trained decomposer on 100+ synthetic trees, inference pipeline generates composition trees from arbitrary prompts, procedural Gaussian fill, interactive web UI. 55-60 Gaussians per scene at depth 1. Details at docs/plans/raum_13_plan.md |
| 10-raum-0-0 | Raum 0.0 | product | done | 2026-04-26 | Local web demo; polish + shader fix shipped 2026-04-27. Known limits (3+ objects, chained relations) deferred to 0.1 |
| 10-raum-0-1 | Raum 0.1 | product | done | 2026-04-27 | Shipped 2026-05-19: GloVe + 300-class Objaverse blob library + 5-object scenes + relation head + editable DSL v1. Colors, spatial relations, and object identity all functional. Known limits: multi-word relations, OOV ambiguity ("plane" -> airplane). These motivate Raum 1.3 recursive decomposition. |
| 10-raum-0-3 | Raum 0.3 | product | done | 2026-05-18 | Shipped 2026-05-21: interactive web demo of Raum 1.3 recursive decomposition. Prompt input, "Decompose + Render" button, live composition tree JSON, Three.js 3D viewer with orbit controls. Arbitrary text-to-3D via trained Planck decomposer. |
| 6-raum-1-4 | Raum 1.4 | model | done | 2026-05-26 | Shipped 2026-05-27: high-fidelity scene generation. Template subdivision (60->13K Gaussians) + gradient densification (13K->51K) + SGS-native refinement (per-cluster Chamfer matching to 109 Sketchfab architecture scans). Beats external multiview consistency on surface coverage (+21%), silhouette fill (+5%), compactness. Export pipeline: .ply, .splat, .obj. 43 unit tests. Details at docs/plans/raum_14_implementation.md |
| 10-raum-0-4 | Raum 0.4 | product | in progress | 2026-05-27 | High-fidelity product demo. Text prompt -> 51K+ Gaussian scene with SGS-native refinement. Quality toggle (skeleton/templates/dense), refinement mode selector (sgs/external), export button (.ply/.splat). Details at SETUP_202606.md §3 |
| 6-raum-2-0 | Raum 2.0 | model | open | 2026-05-27 | Physical Gaussians: extend SGS primitive with material embedding (e_p). Two-stage architecture: discrete classifier + lookup (current), continuous prediction (Hertz scale). P6 correlation validated at small scale (hardness R^2=0.54). 6/7 Lean 4 proofs complete (Aristotle). Whitepaper + literature review + formal math spec at docs/papers/physical_gaussians*.md |
| 10-raum-0-5 | Raum 0.5 | product | done | 2026-06-01 | Shipped 2026-06-01: honest castle on a hill. Deterministic scene grammar (atomic parts: stone/crenellation/tower/gate/keep/hill/tree, corner-tower + centered-gate spatial rules) renders a recognizable low-fidelity castle with NO model, via --scene-file in the demo (prebuilt=True skips subdivision; ~2.6K parts -> ~51K splats). GATE MET: visibly a castle on a hill (4 cream towers w/ red roofs, crenellated walls, keep, green dome), no blobs/sphere dust. Proves the fill->densify->refine->render stack hits the target and is the 1.5 training-data generator. Details at SETUP_202606_2.md §3 |
| 6-raum-1-5 | Raum 1.5 | model | done | 2026-06-01 | Shipped 2026-06-02: decomposer retrained on the 0.5 grammar's SHALLOW skeletons (parts as leaves; fill stage re-expands tower->body+cren+roof via shared builders, reconciling the model path with the 0.5 render). 21.6K records, max ~440 tok (model context is 512, not 768). GATE MET on the headline prompt: "a castle on a hill" generates a recognizable castle FROM A SENTENCE (towers, crenellated walls, red roofs, green hill). Known gaps -> 1.6: accuracy/prompt-faithfulness (collapses onto the dominant template; "a wall with a gate" -> whole castle, "a tower with a gate" -> no gate + unrequested trees) and realism (flat round-disc splats). Details at SETUP_202606_2.md §4 |
| 6-raum-1-6 | Raum 1.6 | model | done | 2026-06-03 | Shipped 2026-06-03: compositional accuracy. Decomposer retrained (Planck 1.3 frozen) on the SceneSpec compositional grammar; val_loss 0.107. PROMPT-FAITHFUL: "tower on a hill" -> tower, "a wall and a gate" -> wall not castle, counts honored. Greedy decoding + parse-retry killed the flaky JSON failures; structural JSON recovery as net. Grammar-validated decoding drops hallucinated leaves; kind-based layout snap caps towers at 4 on the ring. Details at SETUP_20260603.md §3-4 |
| 10-raum-0-6 | Raum 0.6 | product | in progress | 2026-06-03 | Realism. DONE: §5.1 oriented-ellipsoid lit splats (InstancedMesh, per-instance quaternion+scale+color, Lambert + sun/hemi/ambient) replacing flat discs; §5.3 windows/arrow-slits carved into wall faces; §5.4 lighting + fake AO; refine defaulted OFF (SGS Chamfer-to-template distorted clean snapped geometry); tower proportions + hill-vs-footprint sizing hand-tuned. OUTSTANDING: §5.2 real-scan part routing (route each part-leaf to its matched Sketchfab category scan so a tower is a real scanned tower) -- needs the 109-scan library on the 4090; this is the realism payoff and overlaps the 1.7 systemic pivot. Details at SETUP_20260603.md §5 |
| 6-raum-1-7 | Raum 1.7 | model | open | 2026-06-03 | Systemic geometry (replaces hand-tuned grammar). The hand-tuned procedural grammar is the quality ceiling: tower ratio, roof height, hill size are magic numbers that don't generalize. Pivot: (1) train decomposer to emit scales/positions the renderer respects with scan-derived ground truth, instead of snapping to constants; (2) optional differentiable/render-scored objective vs reference images. NOTE: the scan-routing primitive (0.6 §5.2) is the shared building block -- 0.6 lands it as a fill swap; 1.7 makes the model reason about it. Rationale in SETUP_20260603.md "Limit reached" section |
| 10-raum-0-7 | Raum 0.7 | product | open | 2026-06-03 | Product demo on the 1.7 systemic geometry: scan-driven parts so towers/walls are real scanned proportions, not procedural stone-stacks. Depends on 1.7 |
| 11-satz-0-1 | Satz 0.1 | product | open | 2026-05-01 | Local web demo of Planck LM + blob retrieval. Prompt textbox, streamed generation, right-panel retrieved-blob list with transmittance weight bars, k-slider. No new training. Updated 2026-05-04: primary path is Planck 1.3 + Wikipedia blobs (after §2 ships); Planck 1.1 + TinyStories is a flagged-placeholder fallback. Runs AFTER Planck 1.3 + Raum 0.1/1.1, BEFORE Klang 1.3 remainder and Hertz 1.2. Details at docs/plans/satz_01_plan.md |
