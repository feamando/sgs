# SGS VSP v1 — deliver Planck 2.0 (first VSP model), July 2026

*Windows + RTX 4090. Other platforms not validated.*

**Objective:** ship **Planck 2.0**, the first SGS language model whose token
embedding is a bundled **V/S/P** vector (Visual + Semantic + Physical) instead of
a text-only lookup row. This doc is the build plan from a PASSED grounding gate
through tokenizer → training → the disambiguation benchmark that proves the win.

Origin: the VSP grounding gate PASSED 2026-07-06 (see below, and
[[project_sgs_vsp_gate]]). This supersedes the exploratory §2-4 of
`SETUP_202607_path2.md` (which stays as the gate/probe record). Companion:
[[project_sgs_path1_outcome]] (path1 showed base-scaling saturates on a fixed
task; representation is the lever, hence VSP).

DISCIPLINE: gate-and-kill, same as the probe. Prove each step cheaply before the
next. Kill fast if a gate fails; don't build the trainer for a tokenizer that
doesn't round-trip.

## 0. What we KNOW from the VSP experiment (carry these forward)

These are settled findings; the build must respect them.

1. **The representation separates senses; text alone does not.** Gate result:
   text-only 0.13 (FAIL) → auto visual+physical 0.37 (PASS), max-agg over V,S,P,
   20 polysemous words, nothing hand-labeled.
2. **A token is a SENSE, not a word.** crane-bird and crane-machine are two
   tokens. Word-level tokens collapse polysemy (every text regime failed). Sense
   discovery = Gemma-enumerate candidate senses → SD-image per sense → CLIP-embed
   → DEDUP in embedding space (cosine ≥ thr = same sense). NOT a subword splitter.
3. **V = CLIP-image, auto-derived.** SD generates an image per sense PHRASE, CLIP
   image-encodes, mean over views. Open-vocabulary, no ShapeNet ceiling. NOT a
   synset one-hot (tautological), NOT a text vector (collapses).
4. **P = derived, not looked up.** P6 physics MLP (GloVe→8 material props, trained
   on MATERIAL_TABLE) predicts P from the sense phrase. `derive_p6_from_vector.py`.
   Do NOT reintroduce the hand material-tag table — it's a tautology and P must
   stay auto to keep the "one vector, automatically" claim honest.
5. **S = GloVe** (later: the model's own learned S). Identical across senses of
   the same surface word — that's the collapse the bundle fixes.
6. **Separation metric = MAX-aggregation over blocks**, not concat/mean. A sense
   pair is separable if ANY modality distinguishes it (grey fur ball vs grey
   boulder: V collapses, P/S separate). Concat-of-unit-blocks == mean-of-cosines,
   which lets an identical block dilute/veto an informative one. Use
   `vsp_gating_probe.py --aggregate max`.
7. **Separation is carried mostly by V; derived-P is a small honest bonus.** Don't
   overclaim P. If a sense has no groundable image, it falls back toward S-only.
8. **Prompt hygiene matters.** SD sense phrases MUST be concrete + unambiguous
   ("a crane bird with long legs", not "crane") or SD draws its dominant prior
   for every sense and dedup merges them. This is the #1 data bug.
9. **Abstract senses (justice, the, however) are correctly S-only.** No V/P is a
   feature, not a gap. Two-tier vocab (grounded vs abstract).

## 0.1 Environment

```powershell
cd sgs
# tokenizer + vocab build: main .venv (torch 2.6). Grounding (SD/CLIP): .venv-sds.
.venv\Scripts\Activate.ps1
python -c "import torch, sentencepiece; print('base ok')"
# for grounding at corpus scale (SD + CLIP), switch to .venv-sds as in path2 §0.
```

## 0.2 Corpus: filtered Wikipedia (disambiguation senses), NOT COCO/TinyStories

Decision history:
- TinyStories rejected: deliberately SIMPLE vocab = no polysemy to disambiguate.
- COCO captions TRIED then rejected (2026-07-06): visually dense with native
  images, BUT people-centric and LOW-polysemy (200k captions, only 16.9k unique
  words; top nouns man/woman/street/table). It doesn't CONTAIN both senses of the
  words VSP targets, and it's off-domain for Raum. (COCO scripts kept, reusable.)
- **Wikipedia CHOSEN**: genuine polysemy (an encyclopedia has crane-bird AND
  crane-machine), concrete/material/structural vocab (Raum's domain when
  filtered), and the sense inventory is FREE + title-labeled. Wikipedia's
  DISAMBIGUATION PAGES enumerate senses directly: "Crane" -> {Crane (bird),
  Crane (machine), Crane (surname)}, each its own article with its own lead
  image. No Gemma enumeration, no caption sense-guessing; native V is
  BETTER-labeled than COCO (the article title states the sense).

## 0.3 Sense inventory + V/P at corpus scale (Wikipedia pipeline)

1. **Sense inventory (FREE, no Gemma)**: `prepare_wikipedia_senses.py` scans the
   dump, finds disambiguation pages, parses "Word (qualifier)" links -> per-word
   senses with a concrete grounding phrase ("a crane bird"). Keeps words with
   >=2 senses. Also writes a domain-filtered wordfreq for the abstract tier.
2. **V**: native lead image per sense-article via Wikimedia REST (`--with-images`),
   CLIP-embed; SD-generate from the phrase as fallback (derive_vsp_clip.py, the
   proven probe path). Non-visual senses (crane=surname) get zero-V -> S-only,
   correct.
3. **P, derived**: derive_p6_from_vector.py -- P6 MLP from the sense phrase. NO
   hand material table (the tautology lesson).
4. **S**: GloVe (later: the model's own learned S).
5. **Cache** V/S/P per sense token so training reads a lookup, not a live pass.

```powershell
# 1. sense inventory + abstract-tier wordfreq (prepare_wikipedia_senses.py BUILT)
python scripts/prepare_wikipedia_senses.py --hf-cache data/wikipedia/hf `
  --with-images `                              # native Wikimedia lead images for V
  --out data/vsps/senses_wiki.json --wordfreq-out data/wiki_vsp/wordfreq.json
# 2. V (native images if present, else SD), then derived P:
python scripts/derive_vsp_clip.py --senses data/vsps/senses_wiki.json `
  --v-source clip-image --save-views results/vsp_views_wiki `
  --out data/vsps/wiki_clip.json               # (extend to embed image_refs when present)
python scripts/derive_p6_from_vector.py --in data/vsps/wiki_clip.json `
  --glove data/glove.6B.300d.txt --out data/vsps/wiki_pderiv.json
```

(Gemma is NOT needed for Wikipedia sense enumeration; enumerate_senses_gemma.py
and the COCO prep scripts stay for caption corpora if that path is revisited.)

## 1. VSPS vocabulary (TO BUILD: scripts/build_vsps_vocab.py)

Mint a two-tier vocabulary. Reuse the gate's grounding pipeline at corpus scale.

- **Grounded tokens** — one per concrete SENSE, carrying (CLIP-image V, derived P,
  GloVe S). Senses come from Wikipedia disambiguation pages (§0.3), NOT a splitter.
- **Abstract tokens** — function words / abstractions, S-only, no V/P.

```powershell
# corpus-scale senses (Wikipedia) for grounded tier; wordfreq for abstract tier
python scripts/build_vsps_vocab.py `
  --senses data/vsps/wiki_pderiv.json `            # CLIP-image V (native photos) + DERIVED P
  --glove data/glove.6B.300d.txt `                 # S
  --corpus-vocab data/wiki_vsp/wordfreq.json `     # abstract (S-only) tier
  --out data/vsps/vocab.json
# (build_vsps_vocab.py is BUILT + selftested; ran on the 20-word probe = 44 tokens)
```

GATE: grounded vocab covers the common concrete-noun space at a manageable blowup
(most words 1-3 senses → nouns expand ~1-3x; abstract vocab unchanged). Measure
actual coverage on the Wikipedia sense inventory before assuming. Open item:
groundable words with no lead image AND no SD asset → S-only fallback.

## 2. VSPS tokenize the corpus (tokenize_vsps.py — BUILT)

Sense-tag the Wikipedia corpus (genuine polysemy, Raum-domain) before Planck 2.0.

- **Sense-tag each grounded word occurrence**: embedding-Lesk — pick the sense
  whose descriptive term best matches the sentence context. WSD at tokenize time,
  the step SentencePiece skips. (BUILT + selftested: 5/5 on minimal pairs.)
- **Cache V/S/P per token** so training reads a lookup, not a live CLIP pass.

```powershell
python scripts/tokenize_vsps.py --corpus data/wiki_vsp `
  --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt `
  --out data/wiki_vsps
# NOTE: load GloVe over the FULL corpus vocab so context words contribute to WSD
# (the probe run only loaded term+surface words).
```

GATE: VSPS round-trips Wikipedia text, and polysemous words get SENSE-CORRECT
tokens (the disambiguation the thesis promises) where SentencePiece merged them.

## 3. Planck 2.0 training (train_planck2.py — BUILT + smoke-passed)

Token embeddings (tok_mu[d_s], tok_features[d_f]) are INITIALIZED from a
projection of the cached [V|S|P] bundle -- the grounded warm start (meaning
injected before training). Details:

1. **Bundle init**: `init_from_bundles` seeds tok_mu + tok_features from a
   projection of concat(V,S,P). `--random-init` skips it = the matched-compute
   SentencePiece-style BASELINE (the control the gate compares against).
2. **Frozen warm start** = ZERO the seeded tables' grads for `--freeze-vp-steps`
   opt-steps (NOT requires_grad+add_param_group, which desyncs SequentialLR). All
   params stay in the optimizer from step 0.
3. **Plain AdamW, stdout** ([[project_sgs_accel_shelved]], [[feedback_sgs_wandb_default]]).
4. **Hard opt-step budget** (`--opt-steps`), the [[project_hertz_resume_epoch_restart]]
   lesson: the loop stops at the budget, not on loader exhaustion.
5. **Data = memmapped tokens.bin** (uint32; vocab >65535 so NOT uint16). Never
   json.load the 2.1B-token stream (~60GB RAM).

REALITY (validated 2026-07-10): vocab = **82,867 tokens** (15.7k grounded senses
+ 40k abstract + 32k subword), so the model is **~215M params** (embedding +
lm_head dominate; the "~100M" target was pre-vocab-sizing -- the transformer core
is small, the vocab tables are big). Smoke test (200 steps) PASSED: loss
9.38->7.44, freeze->unfreeze fires clean at the boundary, checkpoints save.
~1.04 step/s on the 4090 -> 40k steps ~= 10.7h (~1.2 epochs over 2.14B tokens).

The experiment needs BOTH runs (matched compute) or there's no claim:

```powershell
# TREATMENT: VSP bundle init (overnight 1)
python scripts/train_planck2.py --tokens data/wiki_vsps --vocab data/vsps/vocab.json `
  --opt-steps 40000 --freeze-vp-steps 2000 --save-dir checkpoints/planck2_vsp
# BASELINE: random init, everything else identical (overnight 2)
python scripts/train_planck2.py --tokens data/wiki_vsps --vocab data/vsps/vocab.json `
  --opt-steps 40000 --random-init --save-dir checkpoints/planck2_baseline
```

## 4. The gate that proves the win (eval_disambiguation.py — BUILT)

**Val loss will NOT show the win.** The benchmark scores sense-correct next-token
prediction on minimal pairs: same polysemous word, two sense-contexts, each with
a sense-appropriate vs -inappropriate continuation. Score = fraction where
logP(right|ctx) > logP(wrong|ctx). Chance = 0.5. Tokenizer-AGNOSTIC (VSP vocab
or a SentencePiece baseline via --sp-baseline), so it's a FAIR cross-model gate.
Pairs in scripts/assets/disambig_pairs.json.

```powershell
# --pairs defaults to scripts/assets/disambig_pairs.json (105 pairs); --out MUST
# differ per run or the second clobbers the first (both default to disambig_eval.json).
python scripts/eval_disambiguation.py --checkpoint checkpoints/planck2_vsp/final.pt `
  --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt `
  --subword-model data/hertz12_data/tokenizer.model --out results/disambig_vsp.json
# control: same command on the --random-init baseline checkpoint
python scripts/eval_disambiguation.py --checkpoint checkpoints/planck2_baseline/final.pt `
  --vocab data/vsps/vocab.json --glove data/glove.6B.300d.txt `
  --subword-model data/hertz12_data/tokenizer.model --out results/disambig_baseline.json
```

GATE (the publishable result): Planck 2.0 (VSP) beats the --random-init baseline
at matched params/tokens. That delta is the paper. If no delta → the bundle
didn't help the MODEL (distinct from "the representation separates", already
proven); diagnose embedding wiring vs frozen-V starving the model.

### RESULT — full-compute run (2026-07-17): VSP did NOT win, −3.8 pts

| Model | 105-pair acc | correct |
|-------|--------------|---------|
| Baseline (SentencePiece, random init) | **0.829** | 87/105 |
| VSP (Planck 2.0, bundle init) | **0.790** | 83/105 |
| **Delta (VSP − baseline)** | **−0.038** | not significant (McNemar p≈0.22, 6 discordant) |

Paired contingency: 82 both-correct, 1 VSP-only win, 5 VSP-only regressions
(club/seal/bat/mouse — all on the RARER sense), 17 neither.

**Diagnosis (the §4 fork, resolved): warm start WASHED OUT, not frozen-V starvation.**
- Decisive signal: the two models ended **0.964 output-correlated** (mean |Δlogp|
  0.86). 40k steps / ~2B tokens overwrote the grounded init; both converged to
  the same function. `--freeze-vp-steps 2000` protected the seed for only 5% of
  training, so V was NOT starved — it had 38k steps and still washed out.
- Why slightly NEGATIVE not neutral: two friction sources the baseline lacks —
  (a) seeded embeddings landed at **2.0× native std** (raw nn.Linear, unscaled);
  (b) the 820→128 throwaway projection preserved bundle geometry at only corr
  0.848, then was discarded. ~15% of grounded structure scrambled at t=0 for no
  payoff.
- **Key insight:** the VSP efficiency bet is a LOW-compute / LOW-data claim (warm
  start lets a small model skip learning disambiguation from scratch). 40k/2B is
  the one regime where a warm start CANNOT matter — the model learns it anyway.
  This run tested the thesis where it was designed to fail.

**FIXES APPLIED (train_planck2.py, 2026-07-17):**
1. `init_from_bundles` now rescales seeded tables to native std (2.0× → 1.00×);
   bundle geometry preserved, only scale corrected.
2. `--freeze-vp-forever` added; at low compute set freeze >= opt-steps so the run
   TESTS the warm start instead of overwriting it.

**NEXT — low-compute gate (the honest test of the thesis):**
### RESULT 2 — efficiency curve (2026-07-22): trend APPEARED, then FAILED to reproduce

Single-seed run looked monotonic (below, seed 0). But the 2k point did NOT
reproduce under a second seed, so the magnitude claim is DEAD and only a weak
directional trend survives. Do not cite +5.7.

| opt-steps | VSP acc | baseline acc | delta (seed 0) | McNemar p |
|-----------|---------|--------------|----------------|-----------|
| 2k  | 0.695 | 0.638 | +5.7 pts | 0.18 |
| 5k  | 0.733 | 0.724 | +1.0 pt  | 1.00 |
| 40k | 0.790 | 0.829 | −3.8 pts | 0.22 |

**REPRODUCTION FAILED (2k, seed 1): delta −1.0 pt (0.657 vs 0.667), p=1.00.**
Two-seed 2k: deltas +5.7 / −1.0, mean **+2.4**. Pooled 210 trials: VSP-only 15 vs
base-only 10, **McNemar p≈0.42**. The seed-to-seed swing (6.7pts) is LARGER than
the effect — the +5.7 was substantially a lucky draw.

WHAT SURVIVES: only a directional hint that low-compute deltas are less negative
than the −3.8 full-compute loss (consistent with washout). That is a weak
hypothesis, NOT a result and NOT a paper figure. Cannot claim VSP beats baseline
at any single compute point.

### RESULT 3 — 6-seed 2k (2026-07-27): EMBEDDING-INIT KILLED, effect is noise

Ran seeds 0-5 at 2k (--freeze-vp-forever vs --random-init), aggregated with
scripts/aggregate_disambig_seeds.py:

| regime | seeds | mean Δ | std | 95% CI | t | per-seed deltas |
|--------|-------|--------|-----|--------|---|-----------------|
| 2k | 6 | **+0.016** | 0.027 | **[−0.013, +0.045]** | 1.42 | +5.7 −1.0 +3.8 0.0 +1.9 −1.0 |

**The 95% CI includes zero (t=1.42, not significant).** The +5.7 was just the max
of six draws; two seeds are negative. Combined with 40k = −3.8, there is NO
compute regime where VSP-as-embedding reliably beats the baseline.

**VERDICT: the embedding-init delivery mechanism is DEAD.** The VSP representation
is real (grounding gate 0.37 stands), but initializing token embeddings from the
V/S/P bundle does not help a trained model — LM gradients wash the warm start out
(40k) and the residual low-compute benefit is indistinguishable from seed noise
(2k). Skipped the 5-seed 40k round: the 2k mean already can't clear "2k > 40k"
since its CI includes 0, so 40k reseeds (~50 GPU-h) would only sharpen a −3.8 we
already trust and cannot change the verdict.

**PIVOT → VSP-as-auxiliary-signal** (grounding never forced through the optimizer
as init): (a) contrastive aux-loss; (b) inference-time rerank (cheapest, tried
first). Scripts: scripts/rerank_disambiguation.py.

### RESULT 4 — rerank probe + low-context salvage (2026-07-27): LINE CLOSED

Rerank on the standard 105 (score = logP + λ·max-block-cosine consistency):
best λ=4 gave +2.9pts but **p≈0.55** (7 gains/4 losses); pure-consistency ceiling
0.657. Margin-slice: the whole benefit sat in low-LM-confidence pairs (+8.3 at
λ=4) and was 0 where the LM was confident — correct SHAPE but underpowered (~11
discordant pairs).

Salvage A — built a 260-pair LOW-CONTEXT benchmark
(scripts/assets/disambig_pairs_lowctx.json, 3-5 word contexts, weak sense cue) so
the LM is uncertain by construction and grounding has room to help. The test was
VALID (base acc dropped 0.829→**0.581**, the model IS unsure here) and the signal
DIED:
- accuracy flat across λ (0.562–0.585 vs 0.581 base)
- **honest held-out λ (pick on half, report other half): mean delta −0.004**
- pure-consistency favors-right **54.1%** (barely above 0.5; was 66% on easy set)
- no λ significant anywhere; low-margin subset (n=51) point estimates mostly
  NEGATIVE, p≥0.77.

WHY THE +8.3 EVAPORATED: on the easy set the "low-margin" pairs still carried
residual text cue (base 0.69, consistency-favors-right 67%); grounding rode along
with that text signal. Strip context to where the LM is truly lost (base 0.45-0.58)
and the bundle is lost too (54%). The +8.3 was the last disguise of the same noise
as the +5.7 and +2.9.

**VERDICT: VSP-for-LM is CLOSED (negative).** Two delivery mechanisms
(embedding-init, inference-rerank) × three benchmarks. The grounding as derived
(CLIP-image V + GloVe S + P6 P, mean-pooled) does NOT carry sense information a
trained LM can use: redundant with text where text suffices, near-chance where it
doesn't. The 0.37 separation gate STANDS as its own finding (curated probes) but
does not transfer to helping a language model. Writeup: docs/papers/
vsp_negative_result.md. GPU work parked.

PAIR SET: disambig_pairs.json now has **105 pairs / 42 polysemous words** (2026-07-10),
two styles: cloze (the sense word IS the right answer, e.g. "sat on the grassy
river ___" -> bank) and continuation (word in context, predict a sense-matched
next word). Report the VSP-minus-baseline DELTA, not either absolute number.
Random-model variance dropped from the 8-pair 0.38-0.88 swing to ~0.6 at 105
pairs, still add more if the delta is within a few points.

## Sequencing (gate-and-kill)

| Phase | What | Status / kill-if |
|-------|------|------------------|
| 0 | grounding gate (V+S+P separate senses) | **PASS 2026-07-06, 0.37** (auto V + derived P) |
| 0.2 | Wikipedia corpus + disambiguation senses | **DONE (sample): 300k articles -> 7,678 polysemous words** at ~4k art/s. Full dump default now (~25-30 min). COCO tried+rejected (low polysemy). |
| 0.3 | native V from Wikimedia lead images | wire derive_vsp_clip.py to embed image_refs (SD fallback already works); KILL if disambiguation sense yield too low |
| 1 | VSPS vocab (two-tier) | **build_vsps_vocab.py BUILT + selftested** (probe: 44 tokens, 2x blowup). Run on Wikipedia senses. |
| 2 | VSPS tokenize Wikipedia | **tokenize_vsps.py BUILT** (5/5 minimal pairs). Run on Wikipedia; load GloVe over full corpus vocab. |
| 3 | Planck 2.0 training | **train_planck2.py BUILT + smoke-passed** (215M, loss 9.4->7.4, ~1.04 step/s). Run VSP + --random-init baseline (matched compute). |
| 4 | disambiguation benchmark | **CLOSED 2026-07-27 (negative).** embedding-init KILLED (6-seed 2k CI incl 0, 40k −3.8); rerank + 260-pair low-context salvage KILLED (held-out delta −0.004, pure-consistency 54% where LM unsure). Grounding redundant-with/subsumed-by text for a trained LM. 0.37 gate stands alone. Writeup: docs/papers/vsp_negative_result.md (see §4 RESULT 3+4). |

## Papers this feeds

- **VSPS Tokenization** — phases 1-2 (the sense-token + grounding pipeline).
- **VSP-based Models** — phase 4 (extends the Physical Gaussians paper).
- (Alpha-Compositing > Softmax already at JMLR, [[project_sgs_jmlr_submission]], not this.)

## Honest scope reminder

The gate proved the REPRESENTATION separates senses on a 20-word probe. It did
NOT prove a trained model benefits. Phases 1-4 are unbuilt. The real result is
phase 4's delta; everything before it is plumbing toward that measurement.
