# Grounded Token Bundles Separate Word Senses but Do Not Help a Language Model: A Negative Result

## Abstract

We test whether a grounded token representation, bundling a word sense's visual
(V), semantic (S), and physical (P) features into one vector, improves word-sense
disambiguation in a small language model. A representation-level probe passes: on
curated polysemous words the V+S+P bundle separates senses that a text embedding
collapses (separation 0.37 for V+S+P versus 0.00 for text alone). But across two
delivery mechanisms of the same bundle (embedding initialization and
inference-time reranking) and three benchmarks, the grounding produces no
improvement a trained model retains: the full-compute initialization deficit
reproduces across three seeds (-3.5pts, CI [-4.9, -2.1]), and a scrambled-bundle
control shows the signal is genuine (beating a permuted bundle by +10.5pts at low
compute) yet washed out by training (losing to a random init at full compute). The
signal is redundant with text
where text is sufficient, and collapses to near-chance where text is
insufficient, the exact regime where grounding was supposed to help. We report
the full chain, the diagnosis, and the methodological point that turned a string
of promising-looking effects into a clean negative: reproduction across seeds and
held-out hyperparameter selection.

## 1. Motivation

Tokenizers collapse a word's senses onto one token: "crane" the bird and "crane"
the machine share an embedding row, so a language model must re-derive the sense
from context every time. The VSP hypothesis: bundle three signals into the token
representation so the sense is present before training, S the text embedding, V a
visual grounding (an image generated per sense, then embedded), P a physical
grounding (predicted material properties). If disambiguation lives in the
representation, a model need not spend capacity learning it, promising smaller
models and multimodal understanding built in rather than bolted on.

## 2. The representation separates senses (the probe passes)

On 20 polysemous words, entirely auto-derived (no hand labels), the bundle
separates senses with a max-block aggregate separation of **0.37** (a max over
V, S, P blocks of the `1 - cos` between the two senses). The text (S) block alone
scores **0.00**: both senses of a surface form receive an identical text embedding
by construction, so text cannot separate them at all. The 0.37 is carried almost
entirely by the visual block (a picture of a bird and of a construction crane are
visually unrelated). This result stands on its own and is not contradicted by
anything below. Separation is a property of the representation on curated probes;
it is not evidence that a trained model benefits.

## 3. Delivery mechanism 1: embedding initialization

We initialize a small SGS language model's token embeddings from a linear
projection of each token's V+S+P bundle, then train, and compare against an
identical model with random init at matched compute (a 105-pair minimal-pair
disambiguation benchmark; score = fraction where logP(right context) >
logP(wrong context); chance 0.5).

**Full compute (40k steps, ~2B tokens): VSP 0.790 vs baseline 0.829, delta
-3.8pts at the original seed.** To rule out the projection-scale mismatch in that
run, we retrained two more seeds under corrected init scaling: all three
reproduce, **mean -3.5pts, 95% CI [-4.9, -2.1]** (per-seed -3.8/-2.9/-3.8). The
deficit is not a scaling artifact. On a single seed the paired difference is not
significant (6 discordant pairs, McNemar p=0.22), so the honest reading is that
grounding does not help at full compute and if anything slightly hurts. The two
trained models ended 0.964 output-correlated, consistent with the init being
washed out; the scrambled control below is the stronger washout evidence.

**Low compute (a plausible regime for a warm start to matter).** A single-seed
sweep looked monotonic and encouraging, +5.7pts at 2k steps, +1.0 at 5k, -3.8 at
40k. It did not survive reproduction. A second seed at 2k gave -1.0. Six seeds at
2k gave a **mean delta of +1.6pts with 95% CI [-1.3, +4.5]** (t=1.42, not
significant): +5.7, -1.0, +3.8, 0.0, +1.9, -1.0. The +5.7 was the maximum of six
draws; two seeds were negative. The seed-to-seed swing exceeded the effect. No
compute regime shows a reliable win. Embedding initialization is dead.

**Scrambled-bundle control (is the signal real, or is the pipeline broken?).** We
reinitialize from the same bundles permuted across tokens (each token gets a real
bundle belonging to a different word): same norm and covariance, no
token-correspondence. If the grounding is genuine, the true bundle should beat the
scrambled one. **At 2k it does, significantly: 0.695 vs 0.590, +10.5pts, McNemar
p=0.007** (13 gains, 2 losses of 15 discordant). So the signal is real and the
pipeline is sound. **At 40k the gap decays to +2.9pts (n.s., p=0.51): grounded
0.790 vs scrambled 0.762, with random init (0.829) now above both.** The true
bundle's edge over the scrambled one shrinks from significant (+10.5) at 2k to
non-significant (+2.9) at 40k, and by full compute a plain random start beats
either structured init, which is direct washout evidence, stronger than the 0.964
correlation: an init that measurably matters early stops mattering once the model
can learn the distinction from text.

## 4. Delivery mechanism 2: inference-time reranking

Since gradients overwrite the init, we deliver the grounding at inference instead,
leaving the trained baseline untouched. We rerank next-token candidates by
sense-consistency:

    score(candidate | context) = logP(candidate | context)
                               + lambda * consistency(candidate, context)

where consistency is the max over the V, S, P blocks of the cosine between the
context's grounded-token bundle and the candidate's. The candidate set is exactly
{right, wrong}, so there is no top-k truncation, and the test isolates whether the
signal survives. lambda = 0 recovers the pure baseline (a built-in sanity check
that reproduced the baseline accuracy exactly).

On the standard 105-pair set, the best lambda gave +2.9pts, but at **p ~ 0.55** (7
gains, 4 losses among 11 discordant pairs), and the pure-consistency ceiling
(lambda -> infinity, ignore the LM) was only 0.657. Slicing by base-model
confidence showed the entire benefit concentrated in low-confidence pairs (+8.3pts)
and exactly zero where the model was already confident, the correct shape for a
real effect, but resting on ~11 discordant pairs, far too few to reach
significance.

## 5. The salvage, and why it failed

The correct-shaped +8.3 motivated one targeted test: build a benchmark that is
low-context by construction so the model is uncertain, giving grounding room to
help. We authored 260 minimal-context pairs (3 to 5 word contexts, weak sense
cue, 42 words, both senses).

The test was valid: base accuracy dropped from 0.829 on the standard set to
**0.581** on the low-context set, the model is genuinely uncertain here. And the
signal died:

- accuracy flat across lambda (0.562 to 0.585 vs 0.581 base);
- **honest held-out lambda (selected on one half, reported on the other): mean
  delta -0.004;**
- pure-consistency favors the correct sense only **54.1%** of the time (versus 66%
  on the easier set, and chance 50%);
- nothing significant at any lambda; on the low-margin subset the point estimates
  are mostly negative.

The +8.3 evaporated because on the easier set the "low-margin" pairs still carried
residual text cue (base accuracy 0.69 there, consistency-favors-right 67%); the
grounding was riding along with text signal that happened to correlate. Strip the
context down to where the model is truly lost (base 0.45 to 0.58) and the bundle
is lost too (54%). The grounding was never disambiguating where text could not; it
was echoing text where text already could.

## 6. Verdict

Across two delivery mechanisms and three benchmarks, the VSP grounding as derived
here (CLIP-image V, GloVe S, P6-predicted P, mean-pooled) does not carry sense
information a trained language model can **retain and use**. The full-compute
deficit reproduces across three seeds (-3.5pts, CI [-4.9, -2.1]), and the
scrambled control shows the signal is real (true bundle beats permuted by +10.5pts
at 2k, p=0.007) but is washed out by training: by 40k it loses even to a random
init. Where the trained model is uncertain, the grounding is **redundant with text
where text suffices, and near-chance where text does not.** The
representation-level separation (0.37 vs 0.00 for text) is real but does not
transfer to helping a model. The VSP-for-language-models line is closed.

This does not refute grounding in general. It bounds a specific claim: bundling
these particular auto-derived V/S/P features into the token representation, by
init or by rerank, does not beat a text baseline on next-token disambiguation at
this scale. A different visual source, a non-mean-pooled bundle, a
contrastive-training integration, or a task where text is genuinely insufficient
(not merely sparse) could each behave differently and are not tested here.

## 7. Methodological note (the actual contribution)

Three separate effects looked publishable and were not: +5.7pts (embedding init,
2k), +2.9pts (rerank), +8.3pts (rerank, low-confidence slice). Each was a
plausible point estimate on a small number of discordant trials with a
hyperparameter or seed chosen post hoc. Two disciplines caught all three:

1. **Reproduction across seeds.** The +5.7 collapsed to a 6-seed mean of +1.6 with
   a CI spanning zero. Report the mean and spread, never the best draw.
2. **Held-out hyperparameter selection.** The rerank lambda tuned on the test set
   showed +2.9 to +8.3; chosen on held-out data it showed -0.004. Tuning and
   reporting on the same data manufactures effects.

The negative result is only trustworthy because these were applied before, not
after, believing the positive ones. That is the transferable lesson.

## Artifacts

Repository: feamando/sgs. Scripts: `scripts/train_planck2.py` (init + `--seed`,
`--freeze-vp-forever`), `scripts/eval_disambiguation.py`,
`scripts/rerank_disambiguation.py`, `scripts/aggregate_disambig_seeds.py`.
Benchmarks: `scripts/assets/disambig_pairs.json` (105),
`scripts/assets/disambig_pairs_lowctx.json` (260, low-context). Results:
`results/disambig_*` (per-seed and per-regime), `results/rerank_baseline*.json`.
Build log and full result tables: `SETUP_202607_VSP_v1.md` (RESULT 1 to 4).
