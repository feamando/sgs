# SGS Roadmap — All Directions

**Date:** 2026-04-10

---

## Active Tracks

### Track A1: Theorem Paper (Publication)
**Status:** Draft complete
**File:** `paper/theorem_paper.md`
**Target:** Theory workshop (ICML Theory, NeurIPS MathAI), 4-page format
**What:** Standalone paper on Softmax ⊂ Alpha-Compositing, Lean verified
**Next:** Format as LaTeX, submit

### Track A2: Full Paper (Publication)
**Status:** Draft v3 complete, orthogonally challenged twice
**File:** `paper/semantic_gaussian_splatting.md`
**Target:** EMNLP 2026 or NeurIPS workshop
**What:** Full SGS architecture + all experiments
**Next:** LaTeX formatting, finalize after A1 feedback

### Track A3: Interactive Visualizer (Demo)
**Status:** Plan ready
**File:** `docs/plans/a3_visualizer_plan.md`
**What:** Web app showing Gaussian composition in 3D
**Timeline:** 5 days (MVP) or 12 days (full)
**Viral potential:** HIGH — shareable, educational, novel

### Track A4: Few-Shot Domain Application (Research)
**Status:** Research in progress
**What:** Apply SGS to a domain where few-shot matters commercially
**Timeline:** Depends on domain choice
**Impact:** Practical validation of SGS's inductive bias advantage

### Track B1: Generative Model — Browser Demo (100M)
**Status:** Plan ready
**File:** `docs/plans/b1_generative_model_plan.md`
**What:** 100M-param SGS language model on TinyStories, deployed in browser via ONNX
**Timeline:** 7 days code + 12h GPU + 5 days browser deploy
**Hook:** "100M-param language model using 3D rendering math. Runs in your browser."

### Track B1-1: Generative Model — Internal Benchmark (1B)
**Status:** Plan ready
**File:** `docs/plans/b1_generative_model_plan.md` (B1-1 section)
**What:** 1B-param SGS language model on FineWeb-Edu, benchmark against TinyLlama/Pythia
**Timeline:** 3 days code + 3-5 days GPU
**Question:** Can SGS scale to 1B? How does it compare to matched transformers?

---

## Priority Order

| Priority | Track | Why First |
|---|---|---|
| **P0** | A1 (Theorem paper) | Fastest to complete. Establishes priority on the novel theorem. |
| **P1** | A3 (Visualizer MVP) | 5 days to something shareable. Generates attention for all other tracks. |
| **P2** | B1 (100M generative, browser) | 7 days code + 12h GPU + 5 days deploy. The public demo. |
| **P3** | B1-1 (1B benchmark, internal) | 3 days code + 5 days GPU. Answers "can SGS scale?" |
| **P4** | A4 (Domain application) | Cyber or SciLit PoC after B1 proves generation works. |
| **P5** | A2 (Full paper) | Parallel with everything. Submit with B1/B1-1 results. |

---

## Resource Allocation

| Week | What | GPU | Human |
|---|---|---|---|
| Week 1 | A1: finalize theorem paper | 0 | 2 days |
| Week 1-2 | A3: visualizer MVP | 0 | 5 days |
| Week 2-3 | B1: build + train 100M model | 12h | 7 days |
| Week 3 | B1: browser deployment | 0 | 5 days |
| Week 3-4 | B1-1: scale to 1B + train | 3-5 days | 3 days |
| Week 4+ | A4: domain PoC (cyber or scilit) | varies | varies |
| Ongoing | A2: full paper refinement | 0 | background |
