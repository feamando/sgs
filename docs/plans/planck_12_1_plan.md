# Planck 1.2.1 — Accel remediation + industry-standard hedge plan

*Status: draft plan. Written 2026-04-28. Follows the failed Planck 1.2
compound ablation (see `results/planck_12/README.md`). Blocks Hertz 1.2.*

Planck 1.2's six-run ablation (2026-04-28) failed both sides of the
compound gate: val loss 8.55 vs baseline 4.10, wall-clock 1.07× vs 1.8×
target. Per-proposal diagnoses made the failure modes tractable rather
than existential, but stacking the fix attempts exclusively on the
SGS-native thesis leaves the Hertz 1.2 GPU-week decision hostage to one
more experimental ablation. Planck 1.2.1 therefore runs two remediation
tracks in parallel:

- **(A) SGS-native remediation** — targeted fixes to the three
  proposals that landed but underperformed (tl, sk, shk); drop the one
  that never fired (ap).
- **(B) Industry-standard accel** — Muon optimizer + Liger Kernel, both
  drop-in and known to compound cleanly, as an independent path to the
  Hertz 1.2 unblock gate.

The six-step flow for this track mirrors 1.2:

1. **Plan** — this doc.
2. **Implement** — one big commit across `src/sgs_lm.py`,
   `scripts/train_lm.py`, and new `src/optim/muon.py`; all changes
   gated on CLI flags, default off.
3. **Validation plan** — `docs/plans/planck_12_1_validation.md`: the
   nine-run ablation matrix + gate criteria.
4. **Runbook** — `docs/plans/planck_12_1_runbook.md`.
5. **Push results** — `results/planck_12_1/` sibling directory so the
   1.2 and 1.2.1 tables stay separable.
6. **Update SETUP.md §6.4b** — refresh with the actual implementation
   SHA once the matrix completes.

---

## Design decisions

- **Two parallel tracks.** Either-path gate on the compound. If both
  pass, `all_plus` (SGS-native + Muon + Liger) becomes the Hertz 1.2
  recipe; if only one passes, that one carries.
- **Baseline stays adopted.** The 66,750-step Planck 1.2 baseline
  (`results/planck_12/ablation.json:baseline`) is the reference point.
  No re-training the baseline.
- **Results live in a new directory.** `results/planck_12_1/` rather
  than overwriting `results/planck_12/`, so post-mortems remain
  comparable.
- **Same token budget.** 66,750 steps, batch 32, L=512, seed 1337,
  bf16, RTX 4090. Identical to 1.2.
- **Drop `ap` from the compound.** It did not fire in 66k steps of 1.2
  and val loss was within seed noise. It rejoins only if a standalone
  probe (Planck 1.2.2) fixes its exit gate.
- **Vendor Muon.** Pure-PyTorch reference (~120 LoC), zero external
  deps. The official implementations all assume multi-GPU DDP layouts
  we don't need.
- **Liger as an external dep.** `pip install liger-kernel`, pinned.
  Smoke-test the import + `FusedLinearCrossEntropy` forward-parity
  against bare `F.cross_entropy` as part of the implementation commit.
- **One big commit.** Same cadence as 1.2; easier to revert if the
  matrix reveals a regression in shared infrastructure.

---

## 1. SGS-native remediation (Track A)

### §2.1 tl — fix the degenerate weighting

**Root cause.** On the 1.2 run `T_mean_final = 1.0` across 66k steps.
`(1 - T)^γ` therefore multiplies the CE term by ~0, and the effective
training signal degrades to the `max(0, T - T_max)²` penalty alone. The
penalty minimises trivially by pinning `T = T_max = 0.3`, so the
learning dynamic decouples from next-token prediction entirely.

**Fix.** Two levers, apply both, tuned conservatively:

1. **Plain-CE warmup.** Add `--tl-warmup-steps` (default 5000). Before
   this step the loss is the baseline `F.cross_entropy` call, so T has
   time to stabilise under honest gradients. After warmup, the tl
   reweighting engages.
2. **Weight floor.** Inside the reweighting formula, replace
   `(1 - T)^γ` with `((1 - T) * (1 - eps) + eps)^γ`, parameterised as
   `--tl-floor-eps` (default 0.05). At T=1 this yields `eps^γ ≈ 0.01`
   instead of 0, preserving a non-zero CE gradient. At T≪1 the floor
   is negligible, so the reweighting still peaks at under-absorbed
   tokens.

**Implementation attach point.** `scripts/train_lm.py:286-294` inside
`_compute_loss`. Warmup gate via `global_step < args.tl_warmup_steps`;
floor via one scalar multiply on the weight tensor.

**Per-proposal sanity (diagnostic, not a gate).**
- `T_mean` must drift below `T_max = 0.3` by step 10k.
- Val loss at step 20k must be within 0.05 nats of the baseline's
  step-20k val loss (parity check — the reweighting should not
  regress quality).

### §2.3 sk — fix the OOM in the gather

**Root cause.** `src/sgs_lm.py:324` does
`feat_exp = features.unsqueeze(1).expand(B, L, L, d_f)` before calling
`torch.gather(feat_exp, 2, idx_exp)`. At `B=32, L=512, d_f=1000` bf16
that is exactly the 15.62 GiB the traceback reported. The expand is
needless — we want to index `features[b, top_idx[b, l, :], :]` to get
`[B, L, k, d_f]`, which is only ~4 GiB.

**Fix.** Replace the `expand + gather` pair with advanced indexing:

```python
# Before (lines 322-325):
idx_exp = top_idx.unsqueeze(-1).expand(B, L, k, d_f)
feat_exp = features.unsqueeze(1).expand(B, L, L, d_f)
top_feats = torch.gather(feat_exp, 2, idx_exp)

# After:
batch_idx = torch.arange(B, device=features.device).view(B, 1, 1)
top_feats = features[batch_idx, top_idx]  # [B, L, k, d_f]
```

`features` is `[B, L, d_f]`; `top_idx` is `[B, L, k]`; advanced
indexing produces `[B, L, k, d_f]` directly with no `[B, L, L, d_f]`
intermediate. Memory drops from 16 GiB to ~4 GiB at the same shape.

**Per-proposal sanity.**
- Matches baseline val loss within 0.05 nats over 66k steps (sparse-k
  is a compute trick, not a quality change).
- Throughput ≥1.25× baseline past the sparse warmup at step 5000. If
  it lands below 1.15× the sparse machinery isn't paying for itself
  and we demote to `--sparse-k 32` or drop the proposal.

### §2.4 shk — sharing-schedule instead of always-on

**Root cause.** 1.2's implementation shares one kernel across all 3
passes throughout training. Throughput gain (+9.9%) falls short of the
20% target, and the quality cost (+0.23 nats) exceeds the 0.05 nats
tolerance. Most likely at d_s=128 the passes learn genuinely different
kernels early on, and forcing a shared one blocks that discovery.

**Fix.** Introduce `--shk-schedule`, values `{always, mix, late}`:

- `always` — current behaviour (kept for comparability).
- `mix` — first 2 passes use per-pass kernels, last pass uses shared.
  Default for the re-run.
- `late` — first N steps use per-pass kernels on all passes; after
  `--shk-switch-step` (default 20000), switch the last pass to
  shared. Captures the "learn fine-grained kernels first, then
  consolidate" intuition.

**Implementation attach point.** The shared-kernel codepath in
`_render_pass` (currently a branch on `self.shared_kernel`). Replace
the flag with a schedule state passed in from `train_lm.py` per step.

**Per-proposal sanity.**
- `mix` throughput should land ~1.10-1.15× baseline (down from `always`
  at ~1.10× but quality recovers).
- Val loss within 0.10 nats of baseline. If >0.10 nats, this proposal
  is a quality regression and we drop it from `all`.

### §2.2 ap — dropped from compound

Not fixed in 1.2.1. `passes_ema_final = 3.0` across 66k steps of 1.2
indicates the exit gate never fires, so no quality/wall-clock signal
exists. Fixing it would require instrumenting the gate to log
`T > 1 - eps` frequency and tuning `--ap-eps` / `--ap-min-step`
interactively, which is worth doing as a standalone probe (Planck
1.2.2) but not worth blocking Hertz 1.2 on. Excluded from the `all`
compound in 1.2.1.

---

## 2. Industry-standard accel (Track B)

### Muon optimizer

**What it is.** Matrix-aware optimizer that replaces AdamW on 2D
weight matrices via Newton-Schulz iteration to orthogonalise gradient
updates. Consistently reports 1.3-1.5× sample efficiency over AdamW
across 2025 LM work (Kimi K2 and follow-ons). Pure PyTorch, no CUDA
extension needed. For 1D parameters (embeddings, norms, scalar
gates like `log_tau`), falls back to AdamW.

**Implementation.** New file `src/optim/muon.py`, ~120 LoC. Vendored
from the reference implementation at the canonical Moonshot/KellerJordan
variant; pure PyTorch, Newton-Schulz with T=5 iterations (Moonshot
default). Wrapper class that holds two inner optimizers (Muon for 2D,
AdamW for the rest) and dispatches `step()` / `zero_grad()` to both.

In `scripts/train_lm.py`:

- New arg `--optimizer {adamw,muon}`, default `adamw`.
- On `muon`, partition params by shape: 2D weights go to Muon, all
  else to AdamW. Use a simple `if p.dim() == 2` predicate; special-case
  the `lm_head.weight` which is 2D but behaves like an embedding (send
  it to AdamW alongside `tok_features`, per Muon's own guidance).

**Muon hyperparameters.** Defaults taken from the reference:
- `lr = 0.02` on 2D params (much higher than AdamW's 3e-4; Muon
  normalises update magnitude).
- `momentum = 0.95`.
- AdamW sub-optimizer unchanged: `lr=3e-4, betas=(0.9, 0.95), wd=0.1`.
- Warmup unchanged.

**Per-proposal sanity.**
- Val loss at step 20k ≤ baseline's val loss at step 30k (the 1.5×
  sample-efficiency claim, diagnosed mid-run rather than at the end).
- No divergence; if it NaNs, fall back to `lr = 0.01` once before
  flagging a bug.

### Liger Kernel (FusedLinearCrossEntropy)

**What it is.** Triton-backed fused kernel that computes `logits =
x @ W.T; loss = F.cross_entropy(logits, y)` in one pass without
materialising the intermediate `[B * L, vocab]` logits tensor. At
vocab=32k, d_f=1000, B*L = 16,384, the logits tensor alone is ~2 GiB
bf16 per step; Liger saves that plus the backward-pass equivalent.
Public claims: ~20% step-time reduction, ~60% memory reduction at our
shape class.

**Implementation.**

- Dependency: `pip install liger-kernel` (pin to a known-working
  version in `requirements.txt`).
- In `src/sgs_lm.py`, expose the tied `lm_head.weight` as an attribute
  so `train_lm.py` can hand it to Liger directly.
- In `scripts/train_lm.py`, gate on `--liger`:
  - When off: existing `logits = lm_head(x); ce = F.cross_entropy(...)`.
  - When on: replace with
    `loss = FusedLinearCrossEntropy()(x, lm_head.weight, y)`. Liger
    returns the loss directly; the `--transmittance-loss` reweighting
    path needs a logits tensor for `F.cross_entropy(reduction='none')`
    so it is incompatible with Liger. When both are set,
    `--liger` wins and we log a one-line warning (Liger + `tl` cannot
    co-exist without writing a custom Triton kernel, which is out of
    scope for 1.2.1).

**Per-proposal sanity.**
- Forward-parity smoke: same input, same weights — `Liger` loss must
  match `F.cross_entropy` within 1e-3 absolute. If not, bail on
  `--liger` and open an issue.
- Throughput ≥1.15× baseline (conservative; public claims are 1.20×).

---

## 3. Harness additions

Four new entries appended to `scripts/validate_planck12.py`'s `RUNS`
list, plus the existing six untouched for comparability:

| run_id | label | new flags |
|---|---|---|
| `muon` | Muon only | `--optimizer muon` |
| `liger` | Liger only | `--liger` |
| `muon_liger` | Muon + Liger | `--optimizer muon --liger` |
| `all_plus` | SGS-native `all` (no `ap`) + Muon + Liger | `--transmittance-loss --sparse-k 64 --shared-kernel --shk-schedule mix --optimizer muon --liger` |

Also modify the existing `all` entry to drop `--adaptive-passes` (ap is
no longer in the compound) and use `--shk-schedule mix` by default.

Results land in a new `results/planck_12_1/ablation.json`; per-run
stdout in `results/planck_12_1/<run_id>/train_log.txt`. The harness
accepts `--results-dir` so the 1.2 JSON stays read-only.

---

## 4. Risk register

- **Muon stability.** Higher effective LR. If it diverges, we fall
  back to `lr=0.01` once; a second divergence means bail and file a
  bug.
- **Liger + tl incompatibility.** Code-enforced mutual exclusion with a
  clear error message. `all_plus` uses both, so we either (a) drop tl
  from `all_plus` if a Liger+tl merger is not written, or (b) write a
  tiny wrapper that computes the fused CE then re-weights — TBD at
  implementation time. Default for now: **drop tl from `all_plus`**
  unless the Liger+tl merger is trivial. This is acceptable because
  tl's remediation is orthogonal to Liger's throughput gain.
- **sk gather rewrite changes numerics.** Advanced indexing vs.
  `gather(expand)` should be bitwise identical for the forward pass;
  verify with a 500-step parity run before the full matrix.
- **Disk.** Nine runs × default checkpoints = non-trivial disk use on
  the Windows box. Set `--save-interval 10000` (instead of the default
  2000) in all 1.2.1 runs. Seven runs × 3 checkpoints at ~400 MB each
  is ~8 GB.
- **Wall-clock budget.** 7 new 66,750-step runs at ~3h each = ~21h.
  Budget is tight but acceptable.

---

## 5. What's out of scope

- **FP8 training.** 4090 supports SM 8.9 + `transformer_engine`, but
  stacking FP8 on top of the tl reweight path (which already has
  numeric issues) is a bad sequencing decision. Revisit at Planck
  1.3+.
- **FlashAttention-2.** Cited as a design reference for the sk
  gather rewrite (avoid materialising `[B, L, L, d_f]` — FA-2's tiled
  softmax is exactly this pattern). Not a drop-in: SGS uses Gaussian
  compositing, not QKV attention, so the kernel itself does not apply.
- **Speculative rollout.** RL/post-training only; not relevant to
  pretraining.
- **Tuning `ap`.** Planck 1.2.2 if it becomes a priority. For 1.2.1
  `ap` is dropped entirely.

---

## 6. Gate for Hertz 1.2 unblock

Any of the following compounds passes:

| compound | sample eff. | wall-clock | notes |
|---|---:|---:|---|
| `all` (SGS-native, no ap) | ≥1.43× | ≥1.8× | original thesis |
| `muon_liger` | ≥1.30× | ≥1.7× | industry-standard hedge (Muon ~1.5× sample-eff; Liger ~1.2× wall-clock) |
| `all_plus` | ≥1.50× | ≥1.9× | stacked path (only if both above pass) |

The `muon_liger` thresholds are slightly lower than `all`'s because
the known-good estimates are narrower and we don't need to hit the
full original 1.43×/1.8× to justify Hertz 1.2.

If at least one passes, flip Planck 1.2.1 to `done`, use the winning
recipe for Hertz 1.2, and flip Hertz 1.2 from `blocked` to `open`.

If none pass, demote the worst proposal in each compound and run a
Planck 1.2.2. Do not take Hertz 1.2 off block until a compound passes.
