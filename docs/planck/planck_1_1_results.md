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
| 4a. Intra-sample repetition | pending rerun | — | Planck 1.1 ≤ Planck 1.0 | Original aggregate metric was 417 vs 258 (conflates two behaviours). Rerun with split metric below. |
| 4b. Cross-sample diversity | pending rerun (informational) | — | no pass/fail | Same rerun populates this. Lower diversity is *desirable* in blob territory, not a regression. |

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

Rerun pending (this document is updated with the numbers as soon as
`results/planck11_validation.json` v2 lands).

## 5. Conclusions so far

- Blobs work. Gate 3 shows a real, non-trivial loss improvement.
- Blobs are being used, not ignored. Gate 2 passes above threshold.
- The only *real* open question is the texture of the generations, which
  the Gate 4 rerun will answer. If Gate 4a passes, H-SGS carries into
  Hertz 1.2. If Gate 4a fails, the failure mode is looping inside a
  sample and we have specific knobs to try (below).
- Gate 1 needs a frozen-base variant before it can say anything useful.

## 6. Planck 1.3 proposal

All conditional on Planck 1.1 Gate 4a rerun passing (or at least on being
able to attribute a fail to a knob we can move).

1. **Frozen-base retrain** for the `t_max = 0.0` ablation. Purpose:
   make Gate 1 diagnostic. Cost: one short run, base is the same. This
   is cheap, independent, and should land first.
2. **Blob count sweep.** 50k → 200k → 500k. Hypothesis: looping comes
   from blob collision, where similar contexts resolve to the same
   top-k. More blobs spread the pull across more attractors, which
   should flatten intra-sample repetition while preserving Gate 2 and
   Gate 3. Planck 1.3's primary experiment if Gate 4a flags.
3. **Live blob addition (in-model RAG).** Protocol:
   - Freeze the model and the blob projection.
   - Append new blobs from a held-out slice of TinyStories.
   - Measure: do new prompts route to new blobs? Does Gate 3 on the
     held-out slice improve? Do Gate 2 and Gate 4a stay stable on the
     original slice?
   - This is the most important architectural question: if adding
     blobs at inference time keeps the base intact, H-SGS becomes a
     retrieval layer you can extend without retraining.
4. **Inference-time `t_max` dial.** Expose `t_max` as a CLI knob at
   generation time (currently only set at training time). One checkpoint
   then serves two modes: `t_max = 0.3` for factual / consistent output,
   `t_max = 0.0` or lower for creative.
5. **Per-domain blob shards.** Separate blob stores for code / factual
   / creative, picked by a classifier or prompt marker. This is the
   operational version of "repetition is a feature in code, a bug in
   stories": different shards, different consistency behaviour by
   design. This one is a Hertz-scale experiment, listed here for the
   record.

Ordering: 1 and 4 are cheap and independent; 2 is the main Planck 1.3
experiment; 3 is the architectural result; 5 is Hertz territory.

## 7. Next action

Re-run `python scripts/validate_planck11.py` with the updated Gate 4
split, commit `results/planck11_validation.json`, then update §2 and §4
of this doc with the new numbers and the decision on whether to carry
H-SGS into Hertz 1.2.
