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

### Track B1: Tiny Generative Model (Exploration)
**Status:** Plan ready
**File:** `docs/plans/b1_generative_model_plan.md`
**What:** Train a small text generator using SGS rendering on TinyStories
**Timeline:** 7 days code + 1 day GPU
**Hook:** "Language model trained with 3D rendering math for $10"

---

## Priority Order

| Priority | Track | Why First |
|---|---|---|
| **P0** | A1 (Theorem paper) | Fastest to complete. Establishes priority on the novel theorem. |
| **P1** | A3 (Visualizer MVP) | 5 days to something shareable. Generates attention for all other tracks. |
| **P2** | B1 (Generative model) | 7+1 days. If it works, it's the most compelling demo. If it fails, we learn about SGS's generation limits. |
| **P3** | A4 (Domain application) | Requires choosing a domain and accessing domain-specific data. Longer setup. |
| **P4** | A2 (Full paper) | Parallel with everything else. Submit when A1/A3/B1 provide additional signal. |

---

## Resource Allocation

| Week | What | GPU | Human |
|---|---|---|---|
| Week 1 | A1: finalize theorem paper | 0 | 2 days |
| Week 1-2 | A3: visualizer MVP | 0 | 5 days |
| Week 2-3 | B1: build generative model | 24h | 7 days |
| Week 3+ | A4: domain application | varies | varies |
| Ongoing | A2: full paper refinement | 0 | background |
