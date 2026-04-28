# Klang 1.3 — Scale Klang 1.2 architecture to beat Variant A

*Status: stub plan. Written 2026-04-28. Depends on Klang 1.2
shipped 2026-04-28 (`results/klang_validation.json`). This is the
absolute-quality pass for the Klang track.*

## Why

Klang 1.2 passed both gates (MSE 0.00156, log-MAE 1.378 vs Variant B
2.057). The architecture — complex-valued Gaussians, transmittance
compositing, MRSTFT loss — is clearly the right direction: it fixed
Variant B's sub-200 Hz dropout and near-Nyquist whine. But at 20
Gaussians it's still behind Klang 1.1's brute-force Variant A
configuration on every metric:

| | MSE | log-MAE | MCD-13 |
|---|---:|---:|---:|
| Klang 1.2 iSTFT (20g) | 0.00156 | 1.378 | 685.9 |
| Klang 1.1 stft_3000g (Variant A) | **0.00064** | **1.255** | **308.9** |

Variant A used 3000 Gaussians with a simpler summation path. 1.3 closes
the gap by running 1.2's architecture at Variant-A capacity.

## Hypothesis

Klang 1.2 at 20g is capacity-starved, not architecturally limited. The
error spectrogram (`klang/klang_1_2/trajectories.png` panel 3) shows
residual mass above freq-bin 300 — high-frequency content that 20
Gaussians cannot cover. Scale the same model to 1000-3000 Gaussians
and the fundamentals should beat Variant A:

- Complex amplitudes + transmittance handle phase correctly (Variant A
  uses magnitude STFT only).
- MRSTFT targets three resolutions, so the scaled model shouldn't
  over-fit a single window.
- Mel-scaled init places Gaussians where the spectral energy is.

## Gates (Klang track ships if both pass)

1. **Gate A** — log-MAE < Klang 1.1 Variant A's 1.255.
2. **Gate B** — MCD-13 < Klang 1.1 Variant A's 308.9.

(MSE is not a gate because Variant A already has very low MSE; log-MAE
and MCD-13 are the perceptually meaningful metrics.)

If **only one passes**, ship as "Klang 1.3, wins on X but not Y" and
decide then whether to push to 1.4 or call the track done. If
**neither passes**, the architecture does not scale and we stop at
1.2 + document the ceiling.

## Scope

**In scope:**
- Re-run `klang/klang_1_2_experiment.py` with `--n-layers` sweep:
  500, 1000, 2000, 3000.
- Single canonical clip (same as 1.2: `klang/test_clip.wav`).
- Step budget scaled with layer count (heuristic: `n_steps = 3000 *
  (n_layers / 20) ** 0.5` — diminishing returns, but capacity-starved
  runs need more iterations).
- Re-run validator against **Klang 1.1 Variant A** as the
  reference-for-gates (not Variant B — that was the 1.2 bar).
- Decode via iSTFT path only (Griffin-Lim underperformed in 1.2 and
  the extra decode is diagnostic-only at 1.3 scale).

**Out of scope:**
- No new architecture changes. 1.3 is a pure scaling run.
- No HiFi-GAN bridge (never carried its weight across 1.2's variants).
- No perceptual losses (VGGish/CLAP stub). Revisit if gates fail.
- Hyperparameter tuning of σ/f₀ bounds, MRSTFT resolutions, α floor.
  Those are 1.4 work if 1.3 doesn't clear the gates by capacity alone.

## Risk register

- **Memory.** 3000 complex Gaussians × MRSTFT three-resolution
  forward is >10× the 1.2 activation footprint. May need gradient
  checkpointing or batch-of-windows chunking. Smoke-test at 1000g
  first; abort and chunk if OOM.
- **Runtime.** 1.2 at 20g × 3000 steps was ~10 min on 4090. Scaled,
  3000g × 8000 steps could be 8-12h. Budget the sweep accordingly.
- **Optimisation stability.** Complex Gaussians at high count may
  need a smaller LR or longer warmup. Start at 1.2's defaults; adjust
  only if loss curve is non-monotonic.
- **Plateau.** If loss plateaus above Variant A's equivalent loss,
  1.3 has hit an optimisation wall, not a capacity wall, and further
  scaling won't help. Document + stop.

## Out-of-session decision

If 1.3 passes both gates and ships: Klang track is **done**. Post-ship,
Klang becomes a product concern (Klang 0.x product swimlane) rather
than a model concern.

If 1.3 passes one gate: open a focused 1.4 for the missing axis (log-MAE
or MCD-13 tuning) and treat 1.3 as an intermediate checkpoint.

If 1.3 fails both: stop the model track. The architecture does not
scale within reach. Revisit if/when a new theorem or decoder lands.

## Milestones (rough)

1. Smoke-test `--n-layers 1000 --n-steps 5000`. ~1h. Confirm no OOM
   and loss curve shape.
2. Full `--n-layers 3000` run. ~8-12h.
3. Intermediate `--n-layers 500` + `--n-layers 2000` for a scaling
   curve. ~2h + ~4h.
4. Re-run validator with `--reference-for-gates klang/stft_3000g.wav`
   (Variant A).
5. Write `results/klang_13_validation.json` + `klang/klang_1_3/README.md`
   summary.
6. Roadmap.md + SETUP.md updates.

## Runbook (preview)

```powershell
# Smoke
python klang/klang_1_2_experiment.py ^
  --audio klang/test_clip.wav --n-layers 1000 --n-steps 5000 ^
  --device cuda --out-dir klang/klang_1_3_smoke

# Full sweep
python klang/klang_1_2_experiment.py --audio klang/test_clip.wav --n-layers 500  --n-steps 5000  --device cuda --out-dir klang/klang_1_3_500g
python klang/klang_1_2_experiment.py --audio klang/test_clip.wav --n-layers 1000 --n-steps 6000  --device cuda --out-dir klang/klang_1_3_1000g
python klang/klang_1_2_experiment.py --audio klang/test_clip.wav --n-layers 2000 --n-steps 7000  --device cuda --out-dir klang/klang_1_3_2000g
python klang/klang_1_2_experiment.py --audio klang/test_clip.wav --n-layers 3000 --n-steps 8000  --device cuda --out-dir klang/klang_1_3_3000g

# Gate
python scripts/validate_klang.py ^
  --ref klang/original.wav ^
  --reference-for-gates klang/stft_3000g.wav ^
  --results-out results/klang_13_validation.json
```

(The validator's gate thresholds may need updating to log-MAE and MCD
comparison rather than MSE — check before running.)
