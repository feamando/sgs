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
| 1-planck-1-2 | Planck 1.2 | model | open | 2026-04-20 | Acceleration-recipe validation (gates Hertz 1.2) |
| 1-planck-1-3 | Planck 1.3 | model | open | 2026-04-27 | Fresh-knowledge blobs: generic grammar base + live RSS blob store (Reuters/AP), beats frontier on 24h-fresh QA |
| 1-planck-1-4 | Planck 1.4 | model | open | 2026-04-27 | Conversation-memory blobs: per-turn blob writer + hybrid recency/similarity retrieval; flat cost-per-turn vs. growing context |
| 2-hertz-1-0 | Hertz 1.0 | model | done | 2026-04-07 | Paused 2026-04-20, wall-clock infeasible without accel |
| 2-hertz-1-2 | Hertz 1.2 | model | open | 2026-04-20 | Large-LM run with SGS accel recipe from Planck 1.2 |
| 5-klang-1-0 | Klang 1.0 | model | done | 2026-04-10 | Initial audio-SGS concept + scene.py scaffold |
| 5-klang-1-1 | Klang 1.1 | model | done | 2026-04-15 | Variants A & B trained; phase warble + sub-200Hz dropout findings |
| 5-klang-1-2 | Klang 1.2 | model | open | 2026-04-20 | Complex-valued Gaussians, transmittance compositing, MRSTFT |
| 6-raum-1-0 | Raum 1.0 | model | in progress | 2026-04-24 | Template-routing bridge; retraining 2026-04-27 after X-rel-Y fix |
| 10-raum-0-0 | Raum 0.0 | product | done | 2026-04-26 | Local web demo; polish + shader fix shipped 2026-04-27. Known limits (3+ objects, chained relations) deferred to 0.1 |
| 10-raum-0-1 | Raum 0.1 | product | open | 2026-04-27 | Complex scenes, common-object vocab, OOV policy |
