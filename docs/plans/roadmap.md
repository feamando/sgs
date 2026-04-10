# SGS Master Plan

**Project:** Semantic Gaussian Splatting
**Started:** 2026-04-07
**Updated:** 2026-04-10
**Repo:** github.com/feamando/sgs

---

## What We've Built So Far

| Milestone | Date | Key Result |
|---|---|---|
| Whitepaper v1-v4 | Apr 7 | Architecture defined, 85+ paper literature review |
| 13 Lean 4 proofs | Apr 7 | All mathematical foundations formally verified |
| Novel theorem | Apr 7 | Softmax ⊂ Alpha-Compositing (Lean verified) |
| Phase 1 experiments | Apr 8 | SGS beats softmax +0.027 on STS-B (3 seeds, significant) |
| Phase 1.5 | Apr 8 | IDF/PC1/multi-head all hurt — simplest config is best |
| Phase 2 (NLI + d_s) | Apr 8 | SGS converges with softmax at scale (0.726 vs 0.729) |
| Phase 3 (SCAN) | Apr 8-9 | SGS+GRU 27.2% vs Transformer 0.0% on length generalization |
| SCAN ablation | Apr 9 | GRU decoder is primary driver; SGS encoder ≈ Transformer encoder |
| Kernel isolation (M2) | Apr 9 | ~2/3 kernel advantage, ~1/3 rendering equation |
| Hybrid (M6) | Apr 9 | SGS+Softmax hybrid doesn't beat either alone |
| Paper v3 | Apr 9 | Battle-tested through 3 orthogonal challenges |

---

## The SGS Advantage (Proven)

| Setting | SGS vs Softmax | Confidence |
|---|---|---|
| Zero-shot composition | **+0.08** | High (no training confound) |
| Few-shot (5.7K pairs) | **+0.027** (3 seeds, significant) | High |
| At scale (314K pairs) | Tied (Δ = 0.003) | High |
| Training stability | **5× tighter** error bars | High |
| Expressiveness | **Strict superset** (Lean proved) | Certain |

---

## Active Tracks

### RESEARCH & PUBLICATION

#### A1: Theorem Paper
| | |
|---|---|
| **Status** | Draft complete |
| **File** | `paper/theorem_paper.md` |
| **What** | Standalone paper: Softmax ⊂ Alpha-Compositing, Lean verified |
| **Target** | ICML Theory / NeurIPS MathAI / arXiv preprint |
| **Remaining** | LaTeX formatting, venue selection, submit |
| **Effort** | 2 days |

#### A2: Full SGS Paper
| | |
|---|---|
| **Status** | v3, orthogonally challenged 3×, all claims battle-tested |
| **File** | `paper/semantic_gaussian_splatting.md` |
| **What** | Complete paper: architecture + theorem + all experiments + negative results |
| **Target** | EMNLP 2026 / NeurIPS 2026 workshop |
| **Remaining** | LaTeX formatting, incorporate B1/B1-1 results when available |
| **Effort** | Ongoing (parallel with everything) |

---

### DEMOS & PRODUCTS

#### A3: Interactive Visualizer
| | |
|---|---|
| **Status** | Plan ready (`docs/plans/a3_visualizer_plan.md`) |
| **What** | Web app: type sentence → watch Gaussians compose in 3D |
| **MVP** | Pre-computed examples, pure frontend, GitHub Pages. 5 days. |
| **Full** | Live inference, comparison view, query slider. 12 days. |
| **Merge** | Integrates with B1 to show generation live |
| **Monitoring** | Plausible Analytics, Sentry, GitHub Issues |
| **Rollout** | Alpha → Beta (Discord) → Launch (Twitter/HN) |
| **Viral hook** | "What if we rendered sentences like 3D scenes?" |

#### B1: Browser Language Model (100M)
| | |
|---|---|
| **Status** | Plan ready (`docs/plans/b1_generative_model_plan.md`) |
| **What** | 100M-param SGS language model, trained on TinyStories, deployed in browser |
| **Architecture** | d_s=128, d_f=512, 3 passes, 4 heads, 512 context |
| **Training** | TinyStories, ~12h on RTX 4090, ~$10 |
| **Deployment** | ONNX int8 (~105MB), runs in browser via ONNX Runtime Web |
| **Baseline** | GPT-2 Small (117M) — direct head-to-head |
| **Monitoring** | wandb (training), Plausible + Sentry (browser) |
| **Rollout** | Internal eval → Alpha → Beta → Public launch |
| **Viral hook** | "100M-param language model using 3D rendering math. Runs in your browser. Trained for $10." |

---

### SCALING & BENCHMARKING

#### B1-1: 1B Internal Benchmark
| | |
|---|---|
| **Status** | Plan ready (`docs/plans/b1_generative_model_plan.md`, B1-1 section) |
| **What** | 1B-param SGS language model, benchmark against TinyLlama/Pythia |
| **Architecture** | d_s=256, d_f=1024, 5 passes, 8 heads, 1024 context |
| **Training** | FineWeb-Edu 10B tokens, ~3-5 days on RTX 4090, ~$30-50 |
| **VRAM** | ~12GB peak (fits 4090's 24GB) |
| **Baselines** | TinyLlama-1.1B, Pythia-1B, OLMo-1B |
| **Metrics** | Perplexity, HellaSwag (0-shot), ARC-Easy (0-shot), generation quality |
| **Key question** | Can SGS scale? Within 10% of TinyLlama = major finding. |

---

### DOMAIN APPLICATIONS

#### A4-Cyber: Cybersecurity Threat Intel
| | |
|---|---|
| **Status** | Ideas ready (`docs/plans/a4_cybersecurity_ideas.md`) |
| **Lead idea** | ZeroDay Match: zero-shot CVE → MITRE ATT&CK mapping |
| **Data** | MITRE ATT&CK + NVD/CVE (all public) |
| **SGS angle** | Every zero-day is a zero-shot problem. Gaussian covariance captures technique breadth. |
| **PoC effort** | 3-5 days |
| **Market** | $180B cybersecurity |
| **Other ideas** | Threat narrative clustering, anomaly scoring via residual transmittance |

#### A4-SciLit: Scientific Literature Triage
| | |
|---|---|
| **Status** | Ideas ready (`docs/plans/a4_scientific_literature_ideas.md`) |
| **Lead idea** | EmergingLit: zero-shot paper matching for fields too new for citation-trained models |
| **Data** | arXiv API (~2.5M abstracts, free) |
| **SGS angle** | New fields have zero labeled pairs. Zero-shot is the only option. |
| **PoC effort** | 3-5 days |
| **Market** | $2B+ pharma literature screening |
| **Other ideas** | Cross-field bridge finder, systematic review screening, TrendSplat (animated field evolution) |

#### A4-Future: Other Domains (Phase 2+)
| Domain | Why | Data Access | Priority |
|---|---|---|---|
| Rare Disease Diagnosis | Highest humanitarian impact | Hard (IRB/ethics) | After cyber/scilit prove the approach |
| Low-Resource Languages | 1.2B people underserved | Medium | Needs multilingual SGS |
| Legal (Niche Jurisdictions) | Novel regulations = zero training data | Medium (paywalled) | After domain infra built |
| Financial Regulatory | Cross-jurisdictional compliance | Medium | Commercial potential |

---

## Execution Timeline

### Phase I: Foundation (Weeks 1-2)

| Week | Track | Deliverable |
|---|---|---|
| W1 | **A1** | Theorem paper → LaTeX → submit to arXiv + workshop |
| W1-2 | **A3** | Visualizer MVP → deploy on GitHub Pages → share |
| W2 | **B1** | Build generative model codebase (shared with B1-1) |

### Phase II: Generation (Weeks 2-4)

| Week | Track | Deliverable |
|---|---|---|
| W2-3 | **B1** | Train 100M model on TinyStories (12h GPU) |
| W3 | **B1** | Evaluate + compare to GPT-2 Small |
| W3 | **B1** | Browser deployment (ONNX export + web app) |
| W3-4 | **B1-1** | Scale to 1B, train on FineWeb-Edu (3-5 days GPU) |
| W4 | **B1-1** | Benchmark vs TinyLlama/Pythia → scaling verdict |

### Phase III: Applications (Weeks 4-6)

| Week | Track | Deliverable |
|---|---|---|
| W4-5 | **A4-Cyber** | ZeroDay Match PoC → evaluate on held-out CVEs |
| W4-5 | **A4-SciLit** | EmergingLit PoC → evaluate on SciDocs |
| W5-6 | **A2** | Full paper updated with B1/B1-1/A4 results → submit |

### Phase IV: Scale & Community (Weeks 6+)

| Track | Deliverable |
|---|---|
| **A3 + B1 merge** | Visualizer shows generation live (the killer demo) |
| **Public launch** | Twitter/HN/Reddit: browser demo + paper + visualizer |
| **A4 expansion** | Rare disease or low-resource language PoC |
| **B1-1 paper** | If 1B results are strong → standalone scaling paper |
| **Open source** | Clean up repo, documentation, make it easy for others to build on |

---

## Resource Budget

| Resource | Available | Allocated |
|---|---|---|
| **RTX 4090** | 24/7 | B1: 12h, B1-1: 5 days, A4: varies |
| **Human time** | Part-time (~4h/day?) | ~6 weeks of work across all tracks |
| **Compute cost** | ~$100 budget | B1: $10, B1-1: $50, A4: $20 |
| **Hosting** | Free tiers | Vercel, GitHub Pages, Plausible |

---

## Decision Gates

| Gate | Condition | Decision |
|---|---|---|
| **After B1 training** | Model generates coherent text? | Yes → deploy browser app + start B1-1. No → debug generation. |
| **After B1-1 training** | Within 10% of TinyLlama? | Yes → major paper. 10-30% → investigate. >30% → ceiling found. |
| **After A4 PoC** | SGS beats SBERT zero-shot on domain task? | Yes → develop into product. No → SGS advantage doesn't transfer. |
| **After A3 + B1 merge** | Demo is visually compelling? | Yes → public launch. No → iterate on UX. |

---

## North Star

**Near-term (3 months):** Published theorem + browser demo + domain PoC. SGS is known in the research community as a novel composition mechanism with proven properties.

**Medium-term (6 months):** If B1-1 scaling works: SGS as a viable alternative architecture for small/medium language models, with advantages in few-shot, interpretability, and compositional generalization. Open-source library for researchers.

**Long-term (12+ months):** If the approach continues to validate: SGS as the composition layer in hybrid architectures that combine rendering's structural inductive bias with attention's flexible capacity. Think "the BatchNorm of composition" — a module you add to any architecture to get better initialization and few-shot performance.

---

## Files Index

| File | What |
|---|---|
| **Papers** | |
| `paper/theorem_paper.md` | A1: Softmax ⊂ Alpha-Compositing standalone paper |
| `paper/semantic_gaussian_splatting.md` | A2: Full SGS paper (v3, battle-tested) |
| `paper/orthogonal_challenge.md` | First challenge on A2 |
| `paper/orthogonal_challenge_v2.md` | Second challenge on A2 |
| **Plans** | |
| `docs/plans/roadmap.md` | This file |
| `docs/plans/a3_visualizer_plan.md` | Interactive visualizer plan (v2, with ops) |
| `docs/plans/b1_generative_model_plan.md` | Generative model plan (v4, 100M + 1B, browser deploy) |
| `docs/plans/a4_domain_research.md` | Domain research: 6 domains ranked |
| `docs/plans/a4_cybersecurity_ideas.md` | 3 cybersecurity product ideas |
| `docs/plans/a4_scientific_literature_ideas.md` | 4 scientific literature product ideas |
| **Analysis** | |
| `docs/analysis/phase1_results_v3_final.md` | Phase 1 final (post-challenge) |
| `docs/analysis/phase1_5_results.md` | Phase 1.5: SGS beats softmax, tricks hurt |
| `docs/analysis/phase2_results.md` | Phase 2: data bottleneck, softmax catches up |
| `docs/analysis/phase2_final.md` | Phase 2 final (post-challenge) |
| `docs/analysis/phase3_results.md` | Phase 3: SCAN + NLI + gap closing |
| `docs/analysis/scan_multiseed_challenge.md` | SCAN multi-seed challenge |
| `docs/analysis/paper_fixes_results.md` | M2/M6 results |
| **Proofs** | |
| `docs/proofs/proof_results.md` | All 13 Aristotle results |
| `docs/proofs/softmax_subset_alpha_compositing.md` | Novel theorem standalone doc |
| `docs/proofs/lean/` | 13 Lean 4 source files |
| **Foundations** | |
| `docs/whitepaper/v4_literature_validated.md` | Whitepaper (current) |
| `docs/literature_review.md` | 85+ papers |
| `docs/fpf_atomic_specification.md` | 7 atoms, FPF analysis |
| `docs/course/` | 7-article SGS 101 course |
| **Code** | |
| `src/kernel.py` | Gaussian kernel (Atom A2) |
| `src/rendering.py` | Rendering equation (Atom A3) |
| `src/gaussian.py` | Gaussian vocabulary (Atom A1) |
| `src/model.py` | SGS encoder + all ablation models |
| `src/seq2seq.py` | SCAN seq2seq: SGS, Transformer, RPE, ablations |
| `src/scan.py` | SCAN dataset |
| `src/data.py` | GloVe + STS-B + AllNLI |
| `scripts/` | All experiment scripts |
