# Planck 1.2.2 — Fix the three 1.2.1 bugs and retry the gate

*Status: active. Written 2026-04-29. Follows the failed Planck 1.2.1
matrix (see `results/planck_12_1/ablation.json`). Blocks Hertz 1.2.*

## What failed in 1.2.1

The 1.2.1 eight-run matrix produced three distinct failure modes and
two honest completions. Only `baseline` and `ap` were adopted; of the
six fresh runs, two finished and four crashed.

| run | outcome | honest takeaway |
|---|---|---|
| `tl` | finished, val loss 4.259 (+0.16 nats vs baseline), tok/s ≈ baseline | reweight regresses quality, does nothing for throughput |
| `shk` | finished, val loss 4.261 (+0.16 nats), tok/s 1.02× | `mix` schedule still costs 0.16 nats |
| `sk` | OOM at step 9,900 | gather rewrite did **not** kill the `[B, L, L, d_f]` intermediate |
| `all` | val loss collapsed to 0.066 train / 5.49 val, then OOM at step 29,900 | tl is still degenerate (T pinned to 1.0), and the compound hit the same sk allocator wall |
| `muon` | crashed at step 0 | `TypeError: MuonWithAuxAdam is not an Optimizer` (scheduler refused the wrapper) |
| `all_plus` | crashed at step 0 | same scheduler bug |

1.2.2 is not another full matrix. It is a targeted 3-bug fix pass,
then the smallest re-run that reproves the gate.

---

## The three bugs

### Bug 1 — `MuonWithAuxAdam` is not a `torch.optim.Optimizer`

**Failure.** `LinearLR(optimizer, ...)` inside `scripts/train_lm.py`
calls `super().__init__(optimizer, last_epoch, verbose)` on
`torch.optim.lr_scheduler.LRScheduler`, which type-checks with
`isinstance(optimizer, Optimizer)`. `MuonWithAuxAdam` is a duck-typed
wrapper holding two real `Optimizer` instances but is itself a plain
`object`. Both `muon` and `all_plus` died at model-init time.

**Fix.** Subclass `torch.optim.Optimizer` directly. The wrapper
already exposes `param_groups`, `state_dict`, `load_state_dict`,
`step`, `zero_grad`. What's missing is the `Optimizer` protocol
surface: the `defaults` dict, the `_step_count` mechanics, and
registration of the combined param groups. Two implementation options,
pick the simpler:

- **Option A (preferred).** Make `MuonWithAuxAdam` subclass
  `Optimizer` with an empty `defaults={}`, then in `__init__` call
  `super().__init__([], {})` and manually extend `self.param_groups`
  with the inner optimizers' groups. `add_param_group` is the right
  API here. This is the canonical PyTorch idiom for wrapper
  optimizers.
- **Option B.** Don't wrap; expose Muon and AdamW as two separate
  optimizers, and create two schedulers. Adds bookkeeping everywhere
  (`optimizer.step()` → `muon.step(); adam.step()` in the training
  loop, same for `zero_grad`, save/load, grad-clip). Rejected because
  it scatters the change across the training loop.

Implement Option A. Add a 500-step smoke test:

```
python scripts/train_lm.py --optimizer muon --max-steps 500 \
  --save-dir checkpoints/smoke/muon_bug1
```

must: (a) finish without the TypeError, (b) log monotonically
decreasing loss.

### Bug 2 — `sk` still OOMs on the `[B, L, L, d_f]` allocation

**Failure.** `torch.OutOfMemoryError: Tried to allocate 15.62 GiB`
exactly — that's `32 × 512 × 512 × 1000 × 2` bytes (bf16), i.e. the
full pairwise `[B, L, L, d_f]` tensor we claimed we had killed in the
1.2.1 gather rewrite.

**Root cause.** `features[batch_idx, top_idx]` with
`features: [B, L, d_f]`, `batch_idx: [B, 1, 1]`, `top_idx: [B, L, k]`
*ought to* materialise `[B, L, k, d_f]` directly. But PyTorch's
advanced indexing backward pass for this shape class broadcasts
`batch_idx` against `top_idx`, and the gradient-of-indexing path
constructs an intermediate the size of the expanded index. The
forward is fine (~4 GiB); the backward inflates to 16 GiB. The 1.2.1
OOM is in `backward()`, not `forward()` — look at the `all` traceback:
it's inside `loss.backward()`.

**Fix.** Replace advanced indexing with a flat `torch.gather`:

```python
# features: [B, L, d_f], top_idx: [B, L, k]
B, L, d_f = features.shape
k = top_idx.size(-1)
# Flatten the batch+key axes: [B, L, d_f] → [B*L, d_f].
flat = features.reshape(B * L, d_f)
# Shift top_idx into global index space: [B, L, k] → [B*L*k] indices
# into [B*L]. Add B*L offsets per batch so b's queries index into b's
# own keys, not some other batch's.
offsets = (torch.arange(B, device=features.device) * L).view(B, 1, 1)  # [B, 1, 1]
global_idx = (top_idx + offsets).reshape(-1)                            # [B*L*k]
top_feats = flat.index_select(0, global_idx).view(B, L, k, d_f)
```

`index_select`'s backward is a `index_add` / sparse-scatter, which
touches only the selected rows. No `[B, L, L, d_f]` intermediate in
either direction.

**Per-run sanity.** Smoke test at `B=32, L=512, d_f=1000, sparse-k=64`
for 500 steps. Peak VRAM from `torch.cuda.max_memory_allocated()` must
stay under 18 GB (1.2.1's `sk` ran past step 9,000 before OOM, so peak
was ~24 GB at step 9,900 — fragmentation compounds; aim for headroom).
Also a **parity check**: same seed, 500 steps, `baseline` vs `sk`; loss
must agree within 0.01 nats.

### Bug 3 — `tl` keeps T pinned at 1.0

**Failure.** In both `tl` (finished) and `all` (collapsed), the final
`T_mean_final = 1.0`. The 1.2.1 warmup + floor fixed the
dead-zero-gradient failure of 1.2, but T never falls below 1 in
practice. The `all` run's train loss crashed to 0.066 while val loss
rose to 5.49 — the model memorised the `max(0, T - T_max)²` penalty
(trivially satisfied at `T=1` because `T_max=0.3 < 1` means the
penalty is maximised, not minimised — **this is the bug**).

Wait, re-read: the penalty is `F.relu(T - T_max).pow(2)`. At T=1,
T_max=0.3, that's `(0.7)² = 0.49`. The model should be minimising it,
not pinning at 1. So the penalty *wants* T ≤ 0.3, yet T lands at 1.0.
That means the reweight's other term — the CE component weighted by
`((1-T)(1-eps)+eps)^gamma` — dominates in the wrong direction: at T=1,
CE is scaled by `eps^gamma = 0.05^1.5 ≈ 0.011`. So the model is
getting a near-free pass on CE, and minimises by pushing tokens to
"fully absorbed" early, killing the prediction signal. The compound
collapses to a trivial fixed point where it learns the regulariser's
penalty gradient but not the language.

**Fix.** Two changes, applied together:

1. **Invert the weight sign.** The original paper intent was *more
   weight on under-absorbed tokens* (low T) because those are the
   tokens the model is failing to integrate. The current formula
   `weight = (1 - T)^gamma` gives that. But the floor breaks the
   monotonicity: a token with T=1 (worst absorption) still gets
   floor weight, so the gradient on those tokens is near zero and
   the model never fixes them. Replace the weight with

   ```
   weight = (1 - T_clamp).pow(gamma) + floor
   ```

   (additive floor, not multiplicative). Now weight at T=1 is exactly
   `floor` (non-zero, keeps CE gradient) and weight at T=0 is
   `1 + floor` (upweighted, as intended). This is the signature the
   paper actually described; 1.2.1 implemented the multiplicative
   floor by mistake.

2. **Renormalise the weight.** Divide the per-token weight by its
   batch mean so the *effective* CE magnitude is preserved across
   T distributions:

   ```
   weight = weight / weight.mean().detach().clamp_min(1e-6)
   ```

   Without this, changing `gamma` or `floor` silently rescales the CE
   loss, which is why 1.2's `tl_gamma=1.5` and 1.2.1's settings both
   looked reasonable in isolation but interacted badly with the
   optimiser's LR.

**Per-run sanity.** After these fixes, `tl` at step 20,000 must show
`T_mean < 0.5` and val loss within 0.05 nats of baseline at step
20,000. If T still sticks at 1.0 the reweight math is still wrong and
we escalate (probably the T_diag signal itself is broken — it's
computed inside `_render_pass` and averaged across heads; worth
verifying with a toy sanity run that T actually varies).

---

## Six-step flow

Mirrors 1.2 / 1.2.1:

1. **Plan** — this doc.
2. **Implement** — one commit across:
   - `src/optim/muon.py` — make `MuonWithAuxAdam` subclass
     `Optimizer` (bug 1).
   - `src/sgs_lm.py` — replace advanced-indexing gather with flat
     `index_select` in `_causal_render_sparse` (bug 2).
   - `scripts/train_lm.py` — additive-floor + renormalised weight
     in `_compute_loss` tl branch (bug 3).
3. **Validation plan** — `docs/plans/planck_12_2_validation.md`:
   3-run minimum-viable re-matrix + gate.
4. **Runbook** — `docs/plans/planck_12_2_runbook.md`.
5. **Push results** — `results/planck_12_2/` sibling directory.
6. **Update SETUP.md §6.4c** — append a 1.2.2 section below 1.2.1.

---

## Scope: minimum-viable re-matrix

Not another 8 runs. Only the runs the gate actually needs.

| run_id | purpose | budget |
|---|---|---|
| `baseline` | adopted from 1.2 | 0h |
| `sk_fix` | prove bug 2 is fixed (finishes + val loss ≤ baseline + 0.05) | ~2.5h |
| `muon_fix` | prove bug 1 is fixed + land the Muon sample-eff claim | ~3.0h |
| `tl_fix` | prove bug 3 is fixed (T drops below 0.5 by step 20k) | ~3.0h |
| `all_fix` | SGS-native compound with all three fixes | ~1.7h |
| `all_plus_fix` | SGS-native + Muon, all fixes | ~1.7h |

Five fresh runs, ~12h GPU on RTX 4090. Baseline adopted, `ap` not
re-run (diagnostic only; status unchanged from 1.2.1).

If any of `sk_fix` / `muon_fix` / `tl_fix` still fails, fix in place
and re-run only that run before proceeding to the compounds.

---

## Gate (unchanged from 1.2.1)

Same either-path gate — only the run IDs change. Any one of:

| compound | sample eff. | wall-clock |
|---|---:|---:|
| `all_fix` (SGS-native) | ≥1.43× | ≥1.8× |
| `muon_fix` | ≥1.35× | ≥1.0× (no regression) |
| `all_plus_fix` | ≥1.55× | ≥1.8× |

If at least one passes: flip Planck 1.2.1 + 1.2.2 to `done`, Hertz
1.2 unblocks.

If none pass: stop the SGS-native track. The 1.2.1 data already
suggests each proposal individually costs 0.16 nats at ~baseline
throughput. If the compound doesn't multiplicatively recover those
losses with Muon stacked on top, the recipe is not the accelerator
Hertz 1.2 needs, and we either (a) pivot Hertz 1.2 to plain Muon
alone (if `muon_fix` clears its own bar), or (b) keep Hertz 1.0
paused and open a different accelerator investigation.

---

## What stays out of scope

- No new accel proposals (ap stays dropped; no FP8, no FA-2).
- No Liger (still Windows-incompat; flag stays wired for Linux).
- No hyperparam sweeps on `tl_gamma` / `tl_floor` / `shk-schedule` —
  we're proving the bug fixes work, not tuning.
- No new matrix columns. 6 runs, same seed, same token budget.

---

## Risk register

- **Bug 3 fix doesn't land T below 1.** Means the problem is upstream
  of the reweight: T_diag computation, head aggregation, or the
  render geometry itself. Escalate: instrument a toy 100-step run
  that logs the raw T_all distribution (not just the mean after
  head-averaging). This is 1 extra hour of investigation, worth
  doing before burning the full `tl_fix` budget.
- **Bug 2 fix is slower.** `index_select` has a materialise-then-copy
  cost; it's O(B·L·k·d_f) writes. Forward should match the forward
  path of advanced indexing, backward should be cheaper (that's the
  point). If forward tok/s drops below 85% of baseline, the win
  isn't big enough to justify sk and we demote to `--sparse-k 32` or
  drop from the compound.
- **Muon + transmittance-loss + sparse-k interact.** The compound
  `all_plus_fix` stacks all three. If it NaNs where the individual
  runs are stable, reduce `--muon-lr 0.01` once then bail.
