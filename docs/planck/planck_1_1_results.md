# Planck 1.1 experiment results

Track 2 of the SGS run order (see `SETUP.md` §6.2). Planck 1.1 introduces
Hierarchical SGS: a frozen blob store over TinyStories with a learned
gating head on top of Planck 1.0.

## 1. Setup

- Base model: Planck 1.0 (`checkpoints/planck/best.pt`), 100.9M params,
  `d_s=128 d_f=1000 n_heads=4 n_passes=3 context_len=512`.
- Blobs: 50,000 entries built from Planck 1.0 hidden states on the
  TinyStories training set (k-means++ → `data/blobs/tinystories/blobs.pt`).
- Blob retrieval: top-k = 8, `t_max = 0.3`.
- Fine-tune: Planck 1.1 is Planck 1.0 + a blob projection + a blob gate
  trained on TinyStories with the base **unfrozen**.
- Validator: `scripts/validate_planck11.py`, 100 eval batches, 50 samples
  per model for repetition gates.

## 2. Results by gate

| Gate | Result | Value | Threshold | Comment |
|---|---|---|---|---|
| 3. Perplexity improves | PASS | val_loss 1.7504 → 1.6672 (Δ = +0.0832), ppl 5.76 → 5.30 | Planck 1.1 strictly lower | 14% relative ppl improvement. Blobs demonstrably help the likelihood. |
| 2. Blobs used | PASS | mean effective blob weight = 0.0593 | > 0.05 | Blob pathway contributes non-trivial signal at inference, not a dead branch. |
| 1. Base intact | FAIL (gate design) | `|Δ|` = 0.0835 | ≤ 0.05 | See §3. Not a model regression. |
| 4a. Intra-sample repetition | FAIL | Planck 1.0 mean 5.16/sample → Planck 1.1 mean 8.34/sample (Δmean = +3.18) | Planck 1.1 ≤ Planck 1.0 | Planck 1.1 loops more *within* a single generation. This is a real regression, not a gate-design artefact. |
| 4b. Cross-sample diversity | informational | Planck 1.0 unique_ratio 0.864, Jaccard mean 0.0133; Planck 1.1 unique_ratio 0.825, Jaccard mean 0.0122 | no pass/fail | Small drift toward consistency (lower unique ratio, slightly lower Jaccard). Far smaller than Gate 4a degradation. The retrieval-side upside is not strong enough to compensate. |

Source JSON: `results/planck11_validation.json`.

## 3. Gate 1 is a false fail

The gate checks: with `t_max = 0.0` (blobs silenced), does Planck 1.1 match
Planck 1.0 within `|Δ val_loss| ≤ 0.05`?

Observed: Planck 1.1 (`t_max = 0.0`) val_loss = 1.6669 vs Planck 1.0 = 1.7504.

The base got **better**, by about the same amount the blobs did. The gate
logic assumes turning blobs off reduces 1.1 to 1.0; that is only true if
the fine-tune froze base weights. Ours did not. So what the gate is
actually measuring is "continued training of the base on TinyStories".
That this came out lower is consistent with extra training helping the
base, not with blob drift pathology.

Fix path (scheduled for Planck 1.3):

1. Retrain a `planck11_noablob` variant with `--freeze-base` so the
   ablation only isolates the blob head.
2. Re-run Gate 1 against that checkpoint.

Until 1.3 ships, Gate 1 as scored is not diagnostic. The operative signal
that "blobs help and base is not broken" is Gate 3 pass + Gate 4a pass.

## 4. Gate 4 reframe

The first run reported 417 4-gram repeats for Planck 1.1 vs 258 for Planck
1.0 on 50 samples per model. Reading that as a regression conflates two
different behaviours:

- **Intra-sample** repetition: the same 4-gram appearing multiple times
  *within a single generation*. This is looping / copy-paste pathology,
  and a real fail mode.
- **Cross-sample** overlap: 4-grams shared *between different samples
  of the same prompt*. This is consistency, and it is the main upside
  of blob-style retrieval for factual, search, and code-generation use
  cases. Repetition-across-runs becomes a feature, not a bug, because
  users want deterministic-ish answers to "what is X" rather than
  creative variety.

Gate 4 was split accordingly:

- **Gate 4a** (hard): per-sample mean of within-sample repeated 4-grams,
  Planck 1.1 must be `≤` Planck 1.0.
- **Gate 4b** (informational): unique-4-gram ratio + pairwise Jaccard
  across the 50 samples of the same prompt. Reported for both models;
  lower diversity / higher Jaccard means the model commits harder to the
  same answer. No pass/fail; reported only so we understand the shape
  of the change.

Rerun results:

- Gate 4a FAIL. Mean intra-sample 4-gram repeats: Planck 1.0 = 5.16,
  Planck 1.1 = 8.34. That is a 62% relative increase in within-sample
  looping, well above the gate threshold of 0. This is the real
  regression: the blob pathway, in the current 1.1 configuration, pulls
  generations toward repeating themselves.
- Gate 4b informational. unique-4-gram ratio 0.864 → 0.825, pairwise
  Jaccard mean 0.0133 → 0.0122. The direction is toward more
  consistency, which is the property we wanted, but the magnitude is
  small (unique ratio down 4.5%, Jaccard down 8.8%). The cross-sample
  consistency gain is an order of magnitude smaller than the intra-
  sample looping cost.

Net read. Blobs lower perplexity and are used at inference, which
validates the mechanism. But the 1.1 fine-tune biases decoding toward
the same completions *inside* a sample far more than it tightens
completions *across* samples. The retrieval-for-consistency story
requires a configuration where the second effect dominates, which 1.1
does not achieve. The knobs to move are covered in §6.

## 5. Conclusions

- The mechanism works. Gate 3 is a real, non-trivial 14% ppl
  improvement. Gate 2 confirms the blob pathway is active at inference,
  not silenced. Those two are the minimum conditions for the
  architecture to be worth continuing.
- The 1.1 configuration biases decoding. Gate 4a fails by 62% on
  intra-sample 4-gram repetition, and Gate 4b shows only a marginal
  move toward cross-sample consistency. In plain terms: the blobs
  make single generations loop more than they make different runs of
  the same prompt agree more. The retrieval upside is there in the
  right direction but small.
- Gate 1 is still a false fail as specified. The fine-tune unfroze the
  base, so the `t_max=0` ablation is not measuring what the gate
  claims. This needs a base-frozen retrain to be diagnostic, independent
  of the Gate 4a result.
- Decision on Hertz 1.2. We do **not** carry H-SGS into Hertz 1.2 as
  currently configured. Gate 3 passing alone is not sufficient when
  Gate 4a fails this hard. Blobs go into the Planck 1.3 track, and the
  question of whether they make Hertz 1.2 is answered after 1.3.

## 6. Planck 1.3 proposal

Gate 4a failed hard, so Planck 1.3 is explicitly a "fix intra-sample
looping without losing the Gate 3 gain" track. Five items, ordered by
cost and by how directly they attack the failure.

1. **Frozen-base retrain for Gate 1.** Train a variant with
   `--freeze-base` so the `t_max=0` ablation actually isolates the
   blob head. Cheap, independent, required before Gate 1 says anything
   meaningful. Lands first.
2. **Blob count sweep: 50k → 200k → 500k.** Primary hypothesis for the
   Gate 4a failure: blob collision. When the store is small, similar
   contexts land on the same top-k entries, and the retrieval pathway
   pulls the decoder toward a small set of continuations that then
   repeat. Scaling the store spreads the pull across more attractors.
   Prediction: intra-sample repetition drops toward Planck 1.0 levels
   while Gate 2 and Gate 3 are preserved or improved. This is the
   decisive experiment for 1.3.
3. **Top-k and `t_max` grid on Planck 1.1.** Sweep `top-k ∈ {4, 8, 16}`
   and `t_max ∈ {0.1, 0.3, 0.5}` at inference on the existing 1.1
   checkpoint. This tells us, before retraining, whether the looping
   is an inference-time compositing issue we can dial out, or whether
   it is baked into the fine-tuned base weights. Cheapest of the five.
4. **Live blob addition (in-model RAG).** Protocol:
   - Freeze the model and the blob projection.
   - Append new blobs built from a held-out slice of TinyStories.
   - Measure: do new prompts route to new blobs? Does Gate 3 on the
     held-out slice improve? Do Gate 2 and Gate 4a stay stable on the
     original slice?
   - This is the architectural question that matters most for the
     "built-in RAG" framing: if adding blobs at inference time keeps
     the base intact, H-SGS becomes a retrieval layer we can extend
     over time without retraining. Only worth running once Gate 4a is
     under control via (2) or (3).
5. **Inference-time transmittance dial, exposed as a CLI flag.** One
   checkpoint, two modes: high cap for factual / code / search, low
   cap for creative. Trivial plumbing on top of (3).

Ordering. (1) and (3) are cheap and independent, run both first. (2)
is the main Planck 1.3 experiment. (4) follows (2). (5) is a small
follow-on to (3).

Per-domain blob shards (separate stores for code / factual / creative,
selected by a classifier or prompt marker) move to Hertz 2.x: the idea
needs the larger model and more diverse data to be testable.

## 7. Decision and next action

Decision: H-SGS does **not** enter Hertz 1.2 in the current
configuration. Gate 3 passed, Gate 2 passed, but Gate 4a failed by a
margin that a 62% regression in intra-sample looping cannot be offset
by the marginal cross-sample consistency gain shown in Gate 4b. Hertz
1.2 ships without blobs on the recipe already in `SETUP.md` §6.5.

Next action. Open the Planck 1.3 track with items (1), (3), and (2) in
that order. Once (2) lands, rerun the four-gate validator; a pass on
4a with Gate 2 and Gate 3 preserved is the condition for revisiting
the Hertz decision (Hertz 2.0 territory, not 1.2).
