# Klang LinkedIn Post Images

Image copies for the three-part LinkedIn series. Post text lives in
`../linkedin_posts.md`. Naming: `post-{post-number}-{carousel-slot}_{description}.png`.

## Post 1 — Variant A
- `post-1-1_variant-a_reconstruction.png`     — target vs rendered spectrogram at 1000 Gaussians
- `post-1-2_variant-a_gaussian-positions.png` — where the Gaussians land in time-frequency
- `post-1-3_variant-a_loss-curve.png`         — convergence curve

## Post 2 — Variant B
- `post-2-1_variant-b_20L_trajectories.png`               — layer frequency trajectories (headline shot)
- `post-2-2_variant-b_20L_reconstruction.png`             — target vs rendered vs error
- `post-2-3_variant-b_20L_opacity.png`                    — per-layer opacity envelopes
- `post-2-4_variant-b_10L_trajectories_optional.png`      — optional: contrast with 10 layers
- `post-2-5_variant-b_40L_trajectories_optional.png`      — optional: contrast with 40 layers

Use 4 and 5 only if you want to drive home that the failure mode is identical across depths.

## Post 3 — Klang 1.2
No images yet. After running `klang/klang_1_2_experiment.py` on Windows, copy:
- `klang/klang_1_2/reconstruction.png` → `post-3-1_klang-1-2_reconstruction.png`
- `klang/klang_1_2/trajectories.png`   → `post-3-2_klang-1-2_trajectories.png`
- `klang/klang_1_2/loss.png`           → `post-3-3_klang-1-2_loss.png`
