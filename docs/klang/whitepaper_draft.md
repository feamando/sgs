# Klang: Gaussian Splatting for Audio — Radiance Fields of Sound

**Radiance Labs — Research Proposal**
**Date:** 2026-04-11
**Status:** Draft — Feasibility Analysis

---

## 1. Core Idea

A spectrogram is a 2D field: **time × frequency → amplitude**. Just as 3D Gaussian Splatting represents visual scenes as collections of Gaussians in (x, y, z) → (color, opacity), we propose representing audio as collections of Gaussians in **(time, frequency) → (amplitude, phase)**.

Each "audio Gaussian" (or "audio splat") represents a localized sound component:
- **Position (μ):** center in time-frequency space — *when* and at *what pitch* this sound occurs
- **Covariance (Σ):** spread in time and frequency — how long the sound lasts and how wide its frequency band is
- **Amplitude (α):** how loud this component is
- **Phase/features (f):** timbral characteristics, harmonic content

A complete sound is the **alpha-compositing** of all its Gaussian components — just as an image is the compositing of visual Gaussians.

```
Audio(t, f) = Σᵢ aᵢ · K((t,f), μᵢ, Σᵢ) · Tᵢ · φᵢ
```

Where K is the Gaussian kernel in time-frequency space, T is transmittance (acoustic "masking" — louder sounds mask quieter ones behind them, exactly like visual occlusion), and φ captures phase/timbre.

---

## 2. Why This Might Work

### 2.1 Sound IS a Radiance Field

In 3DGS, a radiance field answers: "What color and density exist at point (x,y,z)?"

An **acoustic radiance field** answers: "What amplitude and timbre exist at point (time, frequency)?"

Both are continuous functions over a coordinate space. Both have:
- **Locality:** sounds are localized in time and frequency (a piano note occupies a specific time window and fundamental + harmonics)
- **Occlusion/masking:** louder sounds mask quieter ones at similar frequencies (psychoacoustic masking = transmittance)
- **Multi-scale structure:** a single note is a small Gaussian; a chord is overlapping Gaussians; an orchestra is thousands of Gaussians across the frequency spectrum

### 2.2 Spectrograms Already Look Like Gaussian Scenes

Look at any mel spectrogram. What do you see? Blobs of energy — localized in time and frequency, with soft boundaries, overlapping, varying in intensity. These blobs ARE Gaussians. The spectrogram IS a 2D radiance field.

Current audio ML systems discretize this into grids (mel bins × time frames) and process it through convolutions or transformers. But the underlying signal is continuous — discretization is a lossy approximation.

Gaussian splatting offers a continuous, adaptive representation: more Gaussians for complex passages (many overlapping sounds), fewer for simple passages (a single tone). This is analogous to how 3DGS uses more Gaussians for detailed textures and fewer for flat surfaces.

### 2.3 Psychoacoustic Masking = Transmittance

One of the most elegant correspondences: **psychoacoustic masking** (the phenomenon where louder sounds make nearby quieter sounds inaudible) is exactly **transmittance** in the rendering equation.

In 3DGS: opaque objects block light from objects behind them.
In audio: loud frequencies mask perception of quiet frequencies nearby.

The alpha-compositing equation naturally models this:
- A loud Gaussian at 440 Hz absorbs transmittance at nearby frequencies
- A quiet Gaussian at 445 Hz has reduced contribution because the transmittance is depleted
- This is frequency masking — built into the rendering equation for free

### 2.4 Adaptive Density Control = Adaptive Audio Resolution

3DGS splits Gaussians in areas of high reconstruction error and prunes them where they're redundant.

For audio:
- **Split:** A single Gaussian representing a chord splits into individual note Gaussians during training
- **Prune:** Silent regions or redundant Gaussians are removed
- **Clone:** Under-represented transients (drum hits, consonants) get additional Gaussians

The vocabulary of Gaussians self-organizes to represent the audio efficiently — more resolution for complex passages, less for simple ones.

---

## 3. Three Application Directions

### 3.1 Sound Synthesis (Text-to-Speech, Music Generation)

**Input:** Text (or symbolic music representation)
**Process:** SGS language model produces a "semantic meaning" → mapped to audio Gaussians → rendered into spectrogram → vocoder to waveform

The SGS text encoder (from our existing work) produces a meaning vector. A decoder maps this to a collection of audio Gaussians: positions in time-frequency space, spreads, amplitudes. The rendering equation composites them into a mel spectrogram. A standard vocoder (HiFi-GAN, BigVGAN) converts to waveform.

**Why Gaussians help TTS:** Current TTS (VALL-E, XTTS) uses discrete audio tokens. But speech is fundamentally continuous — formants slide, pitch varies smoothly, prosody is a continuous function. Gaussians naturally represent these continuous phenomena.

### 3.2 Audio Understanding (Speech-to-Text, Sound Classification)

**Input:** Audio waveform
**Process:** Convert to spectrogram → fit Gaussians (like 3DGS fits to images) → the Gaussian parameters become the representation → downstream tasks

Instead of mel spectrograms + transformers, represent the audio as a set of learned Gaussians. Each Gaussian captures a spectral component. The set of Gaussians IS the audio representation — position, shape, amplitude, features per Gaussian.

This representation is:
- **Interpretable:** each Gaussian corresponds to a physical sound component
- **Compact:** a few hundred Gaussians can represent a complex sound (vs. thousands of spectrogram bins)
- **Continuous:** can query at any time/frequency resolution

### 3.3 Sound Manipulation and Transformation

**The killer app:** Because each Gaussian is a named, editable primitive, you can directly manipulate sound:

- **Pitch shift:** Move all Gaussians up/down in frequency
- **Time stretch:** Spread all Gaussians in time (change Σ_time without changing Σ_freq)
- **Isolate voice:** Keep only Gaussians in the vocal frequency range (85-3000 Hz)
- **Remove noise:** Prune low-amplitude Gaussians
- **Voice cloning:** Replace the Gaussian parameters of one voice with another's while keeping the content Gaussians
- **Style transfer:** Keep positions, replace timbre features
- **Audio inpainting:** Add Gaussians in a gap region, render to fill silence

All of these are trivial operations on the Gaussian set — no complex neural processing needed. This is analogous to how 3DGS enables easy 3D scene editing by moving/deleting individual splats.

---

## 4. The "Klang" Architecture

**Klang** (German: resonance, tone, sound) — Gaussian Splatting for Audio.

### 4.1 Audio Gaussian Primitive

```
G_audio = (μ_t, μ_f, σ_t, σ_f, ρ, α, φ)
```

| Parameter | Meaning | Range |
|---|---|---|
| μ_t | Time center (seconds) | [0, duration] |
| μ_f | Frequency center (Hz or mel) | [0, max_freq] |
| σ_t | Time spread (duration of this component) | > 0 |
| σ_f | Frequency spread (bandwidth) | > 0 |
| ρ | Time-frequency correlation (chirps, glides) | [-1, 1] |
| α | Amplitude (volume) | [0, 1] |
| φ | Phase + timbre features | R^d |

The covariance matrix in 2D (time × frequency):
```
Σ = [[σ_t², ρ·σ_t·σ_f],
     [ρ·σ_t·σ_f, σ_f²]]
```

The correlation parameter ρ captures **chirps** — sounds that sweep in frequency over time (sirens, bird calls, vocal formant transitions). When ρ = 0, the Gaussian is axis-aligned (fixed pitch over time). When ρ ≠ 0, it's tilted — representing a frequency sweep.

### 4.2 Audio Rendering Equation

```
Spectrogram(t, f) = Σᵢ αᵢ · K((t,f), μᵢ, Σᵢ) · Tᵢ · φᵢ
```

This produces a **continuous spectrogram** — queryable at any time and frequency resolution. To produce a discrete mel spectrogram for vocoder input, simply evaluate at the standard mel grid.

### 4.3 Training: Fit Gaussians to Audio

Given a target audio file:
1. Initialize with N random Gaussians in time-frequency space
2. Render to spectrogram via the rendering equation
3. Compare to ground-truth spectrogram (L1 + spectral loss)
4. Backpropagate to Gaussian parameters
5. Apply adaptive density control (split/prune/clone)
6. Repeat until convergence

This is exactly the 3DGS training loop, applied to audio instead of images.

---

## 5. Feasibility Analysis

### 5.1 What Makes This Plausible

| Factor | Evidence |
|---|---|
| **Spectrograms look like Gaussian scenes** | Visual inspection of any spectrogram shows localized blobs |
| **Gaussians already model audio** | GMMs are the foundation of classical speech recognition (HMM-GMMs) |
| **SIREN proves neural fields work for audio** | Sitzmann et al. (2020) represented audio as continuous neural fields |
| **DynFOA uses 3DGS for spatial audio** | Luo et al. (2026) already bridge Gaussians and audio (different angle) |
| **Psychoacoustic masking = transmittance** | Deep structural correspondence, not just analogy |
| **Adaptive density = adaptive audio resolution** | More Gaussians for complex sounds, fewer for simple — proven in vision |

### 5.2 What Could Go Wrong

| Risk | Severity | Mitigation |
|---|---|---|
| **Phase is hard** | High | Spectrograms are magnitude-only; phase reconstruction via Griffin-Lim or learned vocoder. Or: learn phase directly in φ |
| **Audio is higher resolution than images** | High | 1 second of audio = 16,000+ samples. A spectrogram at 22kHz with 1024 FFT = ~86 frames/sec × 512 freq bins. May need many Gaussians |
| **Frequency is log-scale, not linear** | Medium | Use mel scale for Gaussian positions — well established |
| **Transients (drum hits) are very short** | Medium | Gaussians with very small σ_t — adaptive density should handle this |
| **No existing pipeline** | High | Must build from scratch — spectrogram rendering, Gaussian optimizer, vocoder integration |

### 5.3 Alternative Modeling Approaches

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **Full 2D Gaussian Splatting** (our proposal) | Gaussians in time-frequency plane, render to spectrogram | Most direct 3DGS analog; continuous; interpretable | Novel, unproven; phase is hard |
| **1D Gaussian Splatting on waveform** | Gaussians directly on the time axis, render to waveform | Simpler; no spectrogram step | Very high resolution needed; less interpretable |
| **3D Gaussian Splatting for spatial audio** | Gaussians in 3D space with audio features (DynFOA approach) | Already demonstrated | Different goal (spatialization, not synthesis/understanding) |
| **Gaussian mixture spectrogram** | GMM fitted to spectrogram energy as a density | Classical; well-understood math | Static (no rendering equation, no transmittance) |
| **Neural audio codec + Gaussians** | Use Encodec-style tokenization, represent tokens as Gaussians | Leverages existing codecs | Loses the continuous advantage |
| **Implicit neural audio field** | SIREN-style MLP: (t,f) → amplitude | Proven for audio (SIREN) | Implicit = not editable; slow |

**Recommendation:** Start with **Full 2D Gaussian Splatting** on spectrograms. It's the most novel, most directly analogous to 3DGS, and offers the unique editing/manipulation capabilities. Fall back to **Gaussian mixture spectrogram** if the rendering equation adds insufficient value.

---

## 6. Proof of Concept: Experiment Plan

### Phase 0: Spectrogram Reconstruction (1 week)

**Can Gaussians reconstruct a spectrogram?**

1. Take a 3-second audio clip (speech or music)
2. Compute mel spectrogram (ground truth)
3. Initialize 500 Gaussians randomly in time-frequency space
4. Optimize via gradient descent: render → compare to ground truth → update Gaussians
5. Apply adaptive density control (split/prune)
6. Measure reconstruction quality (L1 error, perceptual loss)
7. Convert reconstructed spectrogram back to audio via vocoder

**Success:** Reconstructed audio is recognizable. L1 error within 2x of a JPEG-equivalent compression.

**This is the "Phase 0 feasibility" equivalent of our SGS language experiments.**

### Phase 1: Audio Compression (2 weeks)

If Phase 0 works, test compression:
- How many Gaussians needed for "good" quality speech? Music?
- Compare bits-per-second to Opus, Encodec
- Test editing: can we pitch-shift by moving Gaussians? Time-stretch?

### Phase 2: Audio Generation (3-4 weeks)

Train a model to GENERATE audio Gaussians from text:
- Text → SGS encoder → meaning vector → Gaussian decoder → audio Gaussians → render → vocoder
- Train on LJSpeech (single-speaker TTS dataset)
- Compare to Tacotron/VITS baselines

---

## 7. Connection to Existing SGS Work

Klang reuses the core SGS primitives:
- **Gaussian kernel** (src/kernel.py) — same math, 2D instead of 64D
- **Rendering equation** (src/rendering.py) — same alpha-compositing with transmittance
- **Adaptive density control** — same split/prune/clone logic

The proven theorem (Softmax ⊂ Alpha-Compositing) applies to audio composition too — if we compose audio features via the rendering equation, it's provably at least as expressive as attention-based audio transformers.

---

## 8. Product Vision: Radiance Klang

| Product | What | Market |
|---|---|---|
| **Klang Studio** | Edit audio by manipulating Gaussians directly. Move splats to change pitch, stretch them to change duration, delete to remove sounds. | Music production, podcast editing, audio post-production |
| **Klang Voice** | Voice cloning via Gaussian transfer — extract your voice as Gaussians, apply to new text | TTS, accessibility, content creation |
| **Klang Field** | Spatial audio via 3D Gaussian scenes with sound — every Gaussian has both visual AND audio features | VR/AR, gaming, immersive media |

---

---

## 9. Literature Review: What Exists and What Doesn't

### 9.1 Gaussian Splatting + Audio (Existing)

All existing work uses 3DGS geometry to *inform* audio, not to *represent* audio:

- **DynFOA** (Luo et al., 2026): 3DGS scene geometry conditions a diffusion model for spatial ambisonics. Gaussians encode the room, not the sound.
- **AV-GS** (Bhosale et al., 2024): Audio-Visual Gaussian Splatting learns material-aware scene representations for sound propagation.
- **NeRAF** (Brunetto et al., 2025): Joint acoustic + radiance fields for room impulse responses.
- **Audio-Plane** (Shen et al., 2025): Gaussian planes for talking head synthesis — closest to "Gaussians encoding audio" but for facial motion, not audio signals.

**None of these represent audio signals directly as Gaussian primitives.**

### 9.2 Neural Audio Fields (Closest Conceptual Work)

- **SIREN** (Sitzmann et al., 2020): Proved audio waveforms CAN be represented as continuous neural fields.
- **"Representing Sounds as Neural Amplitude Fields"** (Li et al., 2026): Benchmark for coordinate-MLPs in audio. Most directly relevant — treats audio as a queryable field.
- **HyperSound** (Szatkowski et al., 2022): Meta-learned INRs that generalize across audio signals.

These prove audio-as-continuous-field works — but use **implicit** MLPs, not **explicit** Gaussian primitives. Klang replaces the MLP with Gaussians, gaining the same advantages 3DGS gained over NeRF: speed, editability, interpretability.

### 9.3 Audio Tokenization (What We're Competing With)

Current SOTA audio systems (VALL-E, Voicebox, Step-Audio) ALL use discrete tokens via neural codecs:

- **EnCodec** (Defossez et al., 2022): Residual vector quantization, 1.5-24 kbps.
- **DAC** (Kumar et al., 2023): ~90x compression of 44.1 kHz audio to 8 kbps tokens.

Pipeline: waveform → encoder → continuous latent → VQ discretization → discrete tokens.

**Klang proposes:** waveform → Gaussian fitting → continuous Gaussian set (no discretization). The Gaussians ARE the representation — queryable at any resolution, editable, interpretable.

### 9.4 The Confirmed Gap

**No one has proposed using Gaussian primitives to represent audio signals directly.** Specifically:

1. **Gaussian Audio Fields** — spectrograms as 2D Gaussian mixtures with rendering: unexplored
2. **Gaussian Audio Tokenization** — replacing VQ codebooks with Gaussian sets: unexplored
3. **Adaptive audio resolution via split/prune** — from 3DGS: unexplored

The closest work (Neural Amplitude Fields) uses MLPs for the same idea. Klang is to Neural Audio Fields what 3DGS was to NeRF.

---

*This is a research proposal. Phase 0 (spectrogram reconstruction) is the feasibility gate. If Gaussians can reconstruct audio spectrograms with reasonable quality, the direction is validated.*
