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
| 11 | Satz | product | TBD |

## Entries

| id | name | type | status | date_created | notes |
|---|---|---|---|---|---|
| 1-planck-1-0 | Planck 1.0 | model | done | 2026-04-07 | Foundation 100M LM; baseline |
| 1-planck-1-1 | Planck 1.1 | model | done | 2026-04-14 | Validated blob concept |
| 1-planck-1-2 | Planck 1.2 | model | done | 2026-04-20 | Six-run ablation matrix shipped 2026-04-28; gate FAIL (val loss 2× worse, speedup 1.07×). Diagnoses at results/planck_12/README.md. Remediation tracked as Planck 1.2.1 |
| 1-planck-1-2-1 | Planck 1.2.1 | model | done | 2026-04-28 | Closed 2026-05-01 as FAIL alongside 1.2.2. Two-track accel remediation ran 2026-04-28; 4 of 8 runs crashed (scheduler TypeError, sk OOM, all collapsed). Root-cause bugs fixed in 1.2.2; see 1.2.2 row for the verdict |
| 1-planck-1-2-2 | Planck 1.2.2 | model | done | 2026-04-29 | Closed 2026-05-01 as FAIL. Three-bug fix pass landed (Optimizer subclass, flat index_select, additive tl floor + renorm) but the SGS-native compound still OOM'd (all_fix step 27.1k, all_plus_fix step 44.3k, sk_fix step 10k) because forward [B·H, L, k, d_f] activation is ~8 GiB bf16 × 3 passes and Windows shared-GPU-memory was spilling until system RAM exhausted. Muon regressed +0.48 nats at matched tokens. No gate passed. Decision: shelve the SGS-native accel track at 100M scale; Hertz 1.2 runs on plain AdamW. Learnings (Optimizer subclass, index_select, harness speedup mask) stay in the tree |
| 1-planck-1-3 | Planck 1.3 | model | open | 2026-04-27 | Fresh-knowledge blobs: generic grammar base + live RSS blob store (Reuters/AP), beats frontier on 24h-fresh QA |
| 1-planck-1-4 | Planck 1.4 | model | open | 2026-04-27 | Conversation-memory blobs: per-turn blob writer + hybrid recency/similarity retrieval; flat cost-per-turn vs. growing context |
| 2-hertz-1-0 | Hertz 1.0 | model | done | 2026-04-07 | Paused 2026-04-20, wall-clock infeasible without accel |
| 2-hertz-1-2 | Hertz 1.2 | model | open | 2026-04-20 | Large-LM run on plain AdamW (no accel recipe); unblocked 2026-05-01 after Planck 1.2.1/1.2.2 closed FAIL. Runs after Klang 1.3 and Raum 0.1/1.1 so the Raum demo ships before GPU-weeks commit. ~10 days for 10B tokens on RTX 4090 at baseline throughput. Details in SETUP_202605.md §5 |
| 5-klang-1-0 | Klang 1.0 | model | done | 2026-04-10 | Initial audio-SGS concept + scene.py scaffold |
| 5-klang-1-1 | Klang 1.1 | model | done | 2026-04-15 | Variants A & B trained; phase warble + sub-200Hz dropout findings |
| 5-klang-1-2 | Klang 1.2 | model | done | 2026-04-20 | Complex-valued Gaussians, transmittance compositing, MRSTFT. Shipped 2026-04-28: Gate A (MSE 0.00156) + Gate B (log-MAE 1.378 vs Variant B 2.057) both pass. Bass dropout + Nyquist whine fixed. Still behind Klang 1.1 Variant A (stft_3000g) on absolute quality → 1.3 |
| 5-klang-1-3 | Klang 1.3 | model | open | 2026-04-28 | Scale Klang 1.2 architecture (complex Gaussians + transmittance + MRSTFT) to 1000-3000 gaussians to beat Klang 1.1 Variant A (stft_3000g, log-MAE 1.255, MCD 308.9). Absolute-quality pass; if it clears Variant A the Klang track ships |
| 6-raum-1-0 | Raum 1.0 | model | done | 2026-04-24 | Template-routing bridge; 6-object hexad, 2-object scenes, analytic-label training. Shipped 2026-04-27 as backbone of Raum 0.0 demo |
| 6-raum-1-1 | Raum 1.1 | model | open | 2026-04-27 | Executor-side expansion to pair with Raum 0.1: 30-object template library, 3+ objects with anchor pointers, DSL consumer, relation graph head |
| 10-raum-0-0 | Raum 0.0 | product | done | 2026-04-26 | Local web demo; polish + shader fix shipped 2026-04-27. Known limits (3+ objects, chained relations) deferred to 0.1 |
| 10-raum-0-1 | Raum 0.1 | product | open | 2026-04-27 | Complex scenes, common-object vocab, OOV policy, planner+executor split (Planck-class planner → DSL → Raum 1.1 executor; nano-banana-style edit loop). Run order updated 2026-05-01: ships AFTER Klang 1.3 but BEFORE Hertz 1.2 so the demo is out the door before GPU-weeks commit to Hertz |
