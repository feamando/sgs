# Raum: Bottom-Up Market Sizing

**Radiance Labs, 2026-06-04**
**Beachhead:** indie / small game studios doing **environment & level prototyping**.
**Companion to:** `raum_business_case.md` (replaces the top-down $300M figure flagged there).

## Why this beachhead

Picked over ArchViz / VFX / sci-comm because it maximizes the first-revenue probability:

- **Structure + speed beats photorealism** here. Prototyping/greyboxing values "a labelled, editable layout in minutes" over render quality. Raum's current procedural-geometry limitation is least disqualifying. (CL2)
- **Low procurement friction.** Indies buy tools with a credit card; no enterprise sales cycle. (CL2)
- **Large, countable population** and an existing habit of paying for asset/tooling (Unity/Unreal asset stores, Quixel, Synty, etc.). (CL2)
- **API/SDK-native workflow** already exists (engine plugins are a known distribution channel). (CL2)

## Method

Bottom-up = (addressable customers) × (adoption) × (usage) × (price), built as a funnel with explicit levers. Three scenarios (Conservative / Base / Optimistic). Confidence tags: **CL1** measured, **CL2** strong external, **CL3** reasoned estimate, **CL4** speculative. **Every number below CL2 is a lever to be replaced with real data before any external use.**

## Step 1 — Customer universe (TAM count)

| Lever | Value | Basis | Conf |
|---|---|---|---|
| Game studios/teams worldwide making 3D games | ~150,000 | Steam alone ships ~14-19K games/yr; large long tail of studios, schools, solo devs on itch/console/mobile. Order-of-magnitude. | CL3 |
| Share that build **custom 3D environments** (not 2D/asset-flip only) | 40% | Many mobile/2D/hypercasual excluded | CL3 |
| **Universe (TAM, accounts)** | **~60,000** | 150,000 × 40% | CL3 |

## Step 2 — Serviceable (SAM count)

| Lever | Value | Basis | Conf |
|---|---|---|---|
| Reachable via self-serve web + Unity/Unreal marketplace (English, has GPU/cloud budget) | 50% of TAM | Excludes regions/teams we can't transact with at launch | CL3 |
| **SAM (accounts)** | **~30,000** | 60,000 × 50% | CL3 |

## Step 3 — Obtainable (SOM funnel, 3-year)

Adoption ramp among SAM as the product clears its capability gates (G1 decomposer, G2 learned geometry, G3 real-time render):

| | Conservative | Base | Optimistic | Conf |
|---|---|---|---|---|
| Yr1 paid adoption (post-beta) | 0.3% | 0.7% | 1.5% | CL3 |
| Yr2 | 1.0% | 2.5% | 5% | CL3 |
| Yr3 | 2.0% | 5% | 10% | CL3 |
| **Yr3 paying accounts** | **~600** | **~1,500** | **~3,000** | derived |

## Step 4 — Usage & price (two models, pick one)

### Model A — Seat subscription (recommended for indies; predictable, low-friction)

| Lever | Conservative | Base | Optimistic | Conf |
|---|---|---|---|---|
| Avg seats / paying account | 1.5 | 2 | 3 | CL3 |
| Price / seat / month | $20 | $35 | $50 | CL3 |
| **ARPA / month** | $30 | $70 | $150 | derived |
| **ARPA / year** | $360 | $840 | $1,800 | derived |

### Model B — Usage (per-scene) — sanity check on willingness-to-pay

| Lever | Conservative | Base | Optimistic | Conf |
|---|---|---|---|---|
| Prototype scenes / account / yr | 100 | 300 | 800 | CL3 |
| Price / scene | $0.50 | $1.00 | $2.00 | CL3 |
| **Usage rev / account / yr** | $50 | $300 | $1,600 | derived |

Value anchor: a Raum scene replaces minutes-to-hours of greyboxing and, vs monolithic tools, $50-200 of cleanup per *usable* scene (CL3, verify). Even at $1/scene the buyer captures most of the surplus, which is the point for adoption.

> Seat (Model A) and usage (Model B) land in the same ballpark per account in the Base case (~$840 vs ~$300/yr); seat is recommended for predictability and lower buyer anxiety. Numbers below use **Model A**.

## Step 5 — Revenue (Year 3, beachhead only)

ARR = paying accounts × ARPA/yr.

| Scenario | Accounts (Yr3) | ARPA/yr | **Yr3 ARR (beachhead)** |
|---|---|---|---|
| Conservative | 600 | $360 | **~$0.22M** |
| Base | 1,500 | $840 | **~$1.26M** |
| Optimistic | 3,000 | $1,800 | **~$5.4M** |

Full SAM capture at Base ARPA (ceiling on this beachhead): 30,000 × $840 = **~$25M ARR** (CL3). That is the realistic SAM revenue ceiling for indie game prototyping alone, *not* a near-term target.

## Step 6 — Reconcile vs the top-down figure

- Old top-down (business case): "$3B game-asset market × 10% = $300M opportunity." That conflates the whole asset economy with our slice and assumes Raum captures efficiency value as revenue. **Discard for external use.**
- Bottom-up SAM ceiling, this beachhead: **~$25M ARR** (CL3).
- 3-yr SOM: **$0.2M-$5.4M ARR.**

The two-order-of-magnitude gap is the point: $300M was a category number; ~$25M is what indie game prototyping can actually pay Raum at plausible price/adoption. Adjacent beachheads (ArchViz, VFX pre-viz, sci-comm, robotics) are *additional* bottom-up models layered on top, not multipliers on this one.

## Step 7 — What this implies for the decision

- **Beachhead alone does not make a venture-scale company.** ~$25M SAM ceiling and a ~$1.3M Base Yr3 ARR means indie game prototyping is a **proof-of-pull beachhead and revenue floor**, not the whole story. The investable thesis requires the *format/flywheel* and *multi-segment* expansion (engine SDK, enterprise, then ArchViz/VFX/robotics), each its own bottom-up build.
- **It is enough to clear Gate 4 (buyer pull) cheaply.** Reaching even Conservative Yr1 (~90-150 accounts) validates willingness-to-pay inside the ~$500K beta envelope. That is the actual job of this beachhead.
- **Sensitivity:** revenue is most sensitive to **adoption %** (Step 3) and **ARPA** (Step 4), both CL3. The single highest-value de-risking action is putting the demo in front of 3-5 design partners to replace those two levers with observed numbers.

## Levers to replace with real data (priority order)

1. **Adoption % / willingness-to-pay** — design-partner test (G4). Highest leverage, lowest cost.
2. **Customer universe count** — replace the ~150K with sourced figures (Steam annual releases, Unity/Unreal active-seat disclosures, itch.io/console dev counts).
3. **ARPA / pricing** — A/B seat vs usage with the first partners.
4. **Scenes/account/yr** — instrument the beta to measure real generation volume.
5. **$50-200/scene cleanup cost** — verify the value anchor against a real studio workflow.

## Assumptions register (for editing)

```
TAM_studios            = 150000      # CL3 — REPLACE
pct_custom_3d          = 0.40        # CL3
SAM_reach              = 0.50        # CL3
adopt_yr3              = {cons:0.02, base:0.05, opt:0.10}   # CL3 — HIGHEST LEVERAGE
seats_per_account      = {cons:1.5, base:2, opt:3}          # CL3
price_per_seat_month   = {cons:20, base:35, opt:50}         # CL3
# Yr3 ARR = TAM*pct_custom_3d*SAM_reach * adopt_yr3 * seats * price*12
```
