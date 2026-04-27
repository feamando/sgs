# Planck 1.2 — Acceleration recipe implementation plan

*Status: draft plan. Written 2026-04-27. Depends on Raum 0.0 shipping
(done). Blocks Hertz 1.2. Source paper:
`docs/papers/sgs_training_acceleration.md`.*

Planck 1.2 is the gate between the current baseline (Planck 1.0/1.1 on
TinyStories, Hertz 1.0 infeasible at wall-clock) and the 1B Hertz 1.2 run.
The paper proposes five accelerations; §2.5 (curriculum) is deferred to
Hertz 2.x. This doc covers the other four (§2.1, §2.2, §2.3, §2.4),
implemented behind CLI flags, with a shared A/B harness so each can be
validated in isolation and all four can be validated in composition.

The six-step flow for this track:

1. **Plan** — this doc.
2. **Implement** — one big commit across `src/sgs_lm.py` +
   `scripts/train_lm.py`, all four proposals gated behind flags.
3. **Validation plan** — ablation matrix: baseline + each proposal alone
   + all-four composed. Single seed, FineWeb-Edu subset.
4. **Runbook** — concrete commands for §6.4 in SETUP.md.
5. **Push results** — `results/planck_12/` + `git add`.
6. **Update SETUP.md §6.4** — replace the current strategy text with
   runnable instructions.

---

## Design decisions (confirmed 2026-04-27)

- **Scope**: implement all four at once, validate individually.
- **§2.3 tier**: B (real sparse representation + segmented ops for
  actual speedup, not a correctness-only mask).
- **Validation**: compound number is enough; per-proposal ablations
  serve as isolation check, not as the gate.
- **Dataset switch**: move from TinyStories to **FineWeb-Edu** for
  Planck 1.2 onward. Three reasons: (a) brings Planck closer to Hertz's
  distribution so accel numbers transfer, (b) Planck 1.3 will need a
  real-world corpus anyway, (c) avoids overfitting accel tuning to a
  children's-story distribution.
- **Seeds**: single seed per config. Accept that wins <5% are noise.
- **Flags**: every change gated on a CLI flag, default off. Planck 1.1
  reproduction must continue to work with `git checkout main + train_lm.py`.
- **Commit cadence**: one big commit covering all four implementations.

---

## 1. Proposals to implement

Summary of what each proposal changes, with the existing-code attach
points identified from `src/sgs_lm.py`.

### §2.1 Transmittance-weighted loss

**What changes.** Replace `F.cross_entropy(logits, y)` with a per-token
weighted version:

```
loss_t = (1 - T[t,t])^γ · (-log p(x_{t+1}))
       + λ · max(0, T[t,t] - T_max)²
```

`T[t,t]` is the transmittance *at* the predicted position after the final
rendering pass. High `(1-T)^γ` means the model absorbed a lot of context
and should be confident — we upweight its loss. The floor regularizer
prevents `T → 1` collapse (where the model learns to ignore everything).

**Attach points.**
- `_causal_render` at `src/sgs_lm.py:207-216` computes `T` but discards
  it (only `weights = eff_a * T` escapes). Need to return `T` alongside
  meaning.
- `_render_pass` needs to thread `T` through so the outer forward loop
  can keep the final-pass `T`.
- `forward` returns `(logits, T_diag)` when the flag is on; `None` when
  off.
- `scripts/train_lm.py` around line 216 (the `F.cross_entropy` call)
  picks the weighted vs. plain path.

**Hyperparameters.**
- `--transmittance-loss` (bool): enable.
- `--tl-gamma` (float, default 1.5): focal exponent. Paper suggests
  warmup 0 → 1.5 over first 500M tokens; start with constant 1.5 and
  add warmup only if needed.
- `--tl-lambda` (float, default 0.01): regularizer strength.
- `--tl-tmax` (float, default 0.3): transmittance cap before penalty.

**Dependencies.** Orthogonal. Pure loss reshaping, no compute change.

### §2.2 Adaptive pass count (batch-level early exit)

**What changes.** The multi-pass loop in
`src/sgs_lm.py:289-307` currently runs `n_passes = 3` unconditionally.
Adaptive exit compares transmittance after pass 1 vs. pass 2; if
`||T_pass2 - T_pass1||_∞ < ε`, skip pass 3 for that **whole batch**
(not per-sample, because per-sample would break the graph).

**Attach points.**
- `_render_pass` returns `(meaning, T)` instead of just `meaning`.
- The `forward` pass loop stores `T_prev` and compares against
  `T_current` at each step.
- Adaptive decision is a boolean gate that short-circuits the remaining
  passes when the flag is on.

**Hyperparameters.**
- `--adaptive-passes` (bool): enable.
- `--ap-eps` (float, default 0.02): stopping threshold. Range 0.01-0.05.
- `--ap-min-step` (int, default 2000): don't adaptive-exit before
  step N (early training T is noisy; exiting early locks us into a
  1-pass regime).

**Dependencies.** Needs §2.1's infrastructure (`T` exposed). Orthogonal
in intent to §2.3/§2.4.

### §2.3 Kernel top-k sparsity (Tier B)

**What changes.** In `_pairwise_kernel` at `src/sgs_lm.py:148-177`, after
computing Mahalanobis distances, select the top-k nearest keys per query
and set the rest to zero *before* alpha-compositing. The gain comes from
(a) the kernel materialization stays dense (we still compute the
distance matrix), but (b) the subsequent log-cumsum + bmm over weights
are done on sparse tensors via segmented operations.

Tier B means a real sparse path, not a dense mask. That requires:
- A segmented cumsum over per-query variable-length sparse indices. In
  PyTorch, the cleanest approach is:
  1. Gather top-k indices per query: `idx = topk(-mahal, k).indices`
     → `[B, L, k]`.
  2. Gather corresponding `α·K` values: `[B, L, k]`.
  3. Sort along last axis by the original key-position (so transmittance
     accumulates in causal order).
  4. Dense log-cumsum on the `[B, L, k]` tensor.
  5. Dense bmm-equivalent via `torch.gather` on features, then reduce.

This is one custom function in `src/sgs_lm.py` (call it
`_causal_render_sparse`), selected by the flag inside `_render_pass`.

**Attach points.**
- New `_causal_render_sparse(features, alpha, mahal, causal_mask, k)`
  alongside `_causal_render`.
- `_render_pass` dispatches based on `self.sparse_k` attribute.
- `__init__` takes `sparse_k: int | None = None`.

**Hyperparameters.**
- `--sparse-k` (int, 0 = disabled): top-k key cap. Paper suggests k=64
  at L=512, k=128 at L=1024.
- `--sparse-warmup-steps` (int, default 5000): keep sparsity off for
  the first N steps so kernels aren't random.
- `--sparse-tau-gate` (float, default 30.0): if current `τ > gate`,
  skip sparsity (kernels too flat, top-k meaningless).

**Dependencies.** Orthogonal. Compatible with all other proposals.

### §2.4 Shared kernel across passes

**What changes.** The multi-pass loop currently recomputes the kernel
in every pass (via `_pairwise_kernel` inside `_render_pass`). Proposal:
compute `K` once during pass 1, cache it, and reuse it in passes 2 and
3 despite `μ` being updated between passes. The `μ_update` still runs
(it affects `features` and `alpha` through `pass_ffn` / `alpha_gate`),
but no re-render of the distance geometry.

**Attach points.**
- `_render_pass` gains an optional `K_cached` argument.
- `forward` loop caches `K` from pass 0 when the flag is on and passes
  it to subsequent calls.

**Hyperparameters.**
- `--shared-kernel` (bool): enable.

No numeric hyperparameters. Pure structural.

**Dependencies.** Orthogonal. Stacks cleanly with §2.3 (shared kernel
can itself be a sparse top-k kernel — they multiply rather than
conflict).

---

## 2. Dataset switch: TinyStories → FineWeb-Edu

Hertz 1.0 already uses FineWeb-Edu (`data/fineweb/train.bin`, ~9B
tokens, per SETUP §8). Planck 1.2 will consume a ~500M-token subset
for the A/B harness, and full 1-2B tokens for the final compound run.

Action items:
- `src/tinystories.py` loads `.bin` format — verify it works against
  `data/fineweb/` directly, or add a shim.
- Validation split: carve out last 10M tokens of FineWeb-Edu train.bin
  as `val.bin` if not already present.
- Expect val-loss numbers not comparable to Planck 1.0/1.1 baselines
  since those were TinyStories. Re-run a plain-CE Planck 1.2 on
  FineWeb-Edu subset as the new baseline.

---

## 3. Implementation surface summary

| File | Change | Scope |
|---|---|---|
| `src/sgs_lm.py` | `_causal_render` returns `T`. `_render_pass` returns `(meaning, T)` and accepts `K_cached`. New `_causal_render_sparse`. `forward` returns `(logits, T_final, passes_run)` when flags active. Accepts new `__init__` flags. | ~150 LOC |
| `scripts/train_lm.py` | New CLI flags for all four proposals. Weighted-loss branch. Data source switch to FineWeb-Edu. Per-step logging of `T_mean`, `passes_run`, `sparsity_active`. | ~80 LOC |
| `scripts/validate_planck12.py` (new) | A/B harness: loads N checkpoints, runs same val set, emits `results/planck_12/ablation.json`. | ~120 LOC |
| `data/fineweb/` | Verify val split exists. Add README note if `val.bin` needed to be generated. | small |
| `docs/papers/sgs_training_acceleration.md` | No changes (this is the source). |

Backwards compat: every flag defaults to the pre-1.2 behaviour. Running
`train_lm.py` with no new flags must produce numerically identical
output to current main (fuzz test: compare logits on a fixed
token seed).

---

## 4. Validation plan (preview — full doc in step 3)

Six runs, same seed, same 500M-token FineWeb-Edu subset, same arch:

1. **Baseline** — no flags
2. **+§2.1** — `--transmittance-loss`
3. **+§2.2** — `--adaptive-passes`
4. **+§2.3** — `--sparse-k 64`
5. **+§2.4** — `--shared-kernel`
6. **+all four** — all flags on

Metrics logged per run:
- Wall-clock to X validation-loss target
- Tokens consumed to X validation-loss target
- Final val perplexity after fixed 500M tokens
- tok/s throughput (mean over steps 1000-N)

Gate to Hertz 1.2: compound run (6) reaches Planck 1.0's val loss on
≤70% of the tokens (1.43x sample efficiency) *and* >1.8x wall-clock
speedup vs. baseline. Either fail → revisit individual proposals
using the per-proposal numbers.

---

## 5. Rollout order

1. Implement all four proposals behind flags (one commit).
2. Verify backward compat on plain-CE run (5k steps, compare loss
   curve to last Planck 1.1 run).
3. Write `scripts/validate_planck12.py`.
4. Run six-config ablation matrix. Push results to
   `results/planck_12/`.
5. Update `SETUP.md §6.4` with runnable commands.
6. If compound gate passes, update `roadmap.md` (Planck 1.2 → done)
   and unblock Hertz 1.2 prep.

---

## Open questions — resolved

- ~~Scope of "implement"~~ → all four at once.
- ~~§2.3 tier~~ → Tier B.
- ~~Seeds~~ → single.
- ~~Dataset~~ → FineWeb-Edu.
- ~~Flag-gated vs. default-on~~ → flag-gated, default off.
- ~~Commit cadence~~ → one big commit.

## Resolved — ready to proceed

- **`train.bin` exists** (~17 GB). **`val.bin` exists** (~174 MB).
  Dataset files are ready. Loader compatibility with `src/tinystories.py`
  will be verified during step 2 (implement): first action of the
  implementation commit is a 100-batch dry-run on FineWeb-Edu bytes to
  confirm the existing loader eats the new `.bin` format. If it doesn't,
  the shim is small (header/byte-order differences at worst).
- **§2.3 top-k profiling strategy**: start with Tier B as planned. To
  catch the "top-k itself is the bottleneck" case without a full profiling
  pass, the training loop will log **per-substep wall times** (kernel
  compute, top-k selection, render, mu/alpha updates) behind a
  `--log-profile` flag, printed every N steps to stdout. If top-k ends
  up >25% of step time, we flag it in the validation doc and consider
  a fused replacement. Stdout line format:

  ```
  step 1000 | t/step 142ms | kernel 38ms | topk 9ms | render 71ms | update 24ms | tok/s 11840
  ```

  For one-shot deep profiling, add a `--profile-step N` flag that wraps
  step N in `torch.profiler.profile` and dumps the top 20 ops by
  cumulative CUDA time. Cheap insurance; usually only needed once.
