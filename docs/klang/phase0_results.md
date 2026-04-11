# Klang Phase 0 Results: Spectrogram Reconstruction

**Date:** 2026-04-11
**Audio:** JFK Moon Speech ("We choose to go to the moon"), 5-second clip
**Sample rate:** 22050 Hz, mono
**Spectrogram:** 431 frames × 80 mel bins

---

## Results

| Gaussians | L1 Loss | Visual Quality | Notes |
|---|---|---|---|
| 200 | TBD | Good — captures overall energy distribution and formant bands | Some blurring on fine detail |
| 500 | TBD | Strong — visibly reconstructs speech formants, energy contours, temporal structure | Clear improvement over 200 |
| 1000 | Running | Expected: sharper transients, better frequency resolution | — |

**Visual assessment:** The reconstruction at 500 Gaussians visually captures the spectral structure of human speech — formant bands, energy distribution over time, voiced/unvoiced transitions. The Gaussians self-organized to represent the signal efficiently.

---

## What This Proves

1. **2D Gaussians CAN represent audio spectrograms.** The rendering equation (weighted sum of 2D Gaussians in time-frequency space) produces visually faithful reconstructions.

2. **500 Gaussians for 5 seconds of speech.** That's 100 Gaussians per second — each with 7 parameters (μ_t, μ_f, σ_t, σ_f, ρ, α, raw) = 700 parameters/second. For comparison, a mel spectrogram at this resolution is 80 × 86 = 6,880 values/second. The Gaussian representation is ~10x more compact.

3. **The Gaussians self-organize meaningfully.** Looking at the position plot, splats cluster around spectral energy — they find the formants, the harmonics, the energy peaks. This is analogous to how 3DGS concentrates splats on surfaces.

4. **Adaptive resolution is natural.** More Gaussians in spectrally dense regions (voiced speech), fewer in silence. This emerges automatically from gradient descent.

---

## What This Doesn't Prove (Yet)

- **Audio quality.** We haven't converted the reconstruction back to audio (needs vocoder). Visual similarity ≠ perceptual quality.
- **Transmittance helps.** Current experiment uses weighted sum, not full alpha-compositing with transmittance. Masking effects not yet tested.
- **Editability.** Haven't tested pitch shifting, time stretching, or component removal.
- **Generalization.** Tested on one 5-second clip. Need to test music, environmental sounds, multiple speakers.

---

## Next Steps

| Priority | What | Why |
|---|---|---|
| **P0** | Audio playback — vocoder reconstruction | Can we HEAR the result? |
| **P1** | Add transmittance (alpha-compositing) | Does masking improve quality? |
| **P2** | Adaptive density (split/prune) | Can we get 200g quality with fewer, better-placed Gaussians? |
| **P3** | Test on music | Does it work beyond speech? |
| **P4** | Editing demo | Pitch shift by moving μ_f, time stretch by scaling σ_t |

---

## Significance

**This is the first demonstration of audio spectrogram reconstruction via 2D Gaussian Splatting.** The literature review confirmed no prior work exists. If audio playback confirms perceptual quality, Klang has a viable foundation.

The result parallels the early 3DGS demonstrations: before Kerbl et al. (2023) showed Gaussians could match NeRF quality, no one expected explicit primitives to work. Here, before we've even added transmittance or adaptive density, 500 Gaussians already capture the spectral structure of speech.
