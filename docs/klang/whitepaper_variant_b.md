# Klang Variant B: Layer-Based Audio Gaussian Splatting

**Radiance Labs — Research Proposal**
**Date:** 2026-04-12
**Status:** Draft

---

## 1. The Problem with Variant A

Variant A (point-blob Gaussians on the spectrogram) failed to scale: reconstructing 5 seconds of speech required 1000+ Gaussians that essentially became dots. No structural advantage over a raw spectrogram — just a lossy approximation.

**Root cause:** A spectrogram has ~200K values (431 frames × 513 bins). Representing it with 1000 Gaussians (7000 parameters) is 28× compression — but the Gaussians don't capture any musical/audio structure. They're just a basis function approximation, no better than PCA or a wavelet decomposition.

## 2. Variant B: Gaussians as Layers

**Core insight:** In 3DGS, each Gaussian represents an OBJECT, not a pixel. A coffee mug is a cluster of Gaussians, each describing part of the mug's surface. The representation is object-level, not pixel-level.

In audio, the equivalent of an "object" is a **sound source** — a voice, an instrument, a noise layer. Variant B represents each sound source as a single Gaussian layer with continuous behavior over time.

### 2.1 The Layer Gaussian

```
Layer = {
  μ_f:        center frequency (Hz or mel) — where this layer lives in frequency space
  Σ_f:        frequency bandwidth — how wide this layer's frequency band is
  path(t):    frequency trajectory — continuous function of time
              (pitch contour: how the frequency moves)
  α(t):       opacity trajectory — continuous function of time
              (amplitude envelope: when this layer is audible)
  timbre:     spectral shape features (harmonic content)
}
```

A human voice might be:
- μ_f = 200 Hz (fundamental frequency range)
- Σ_f = 50 Hz (bandwidth around the fundamental)
- path(t) = the pitch contour of speech (rising for questions, falling for statements)
- α(t) = loud during vowels, quiet during pauses, zero during silence
- timbre = the spectral envelope that makes it sound like a specific person

### 2.2 Continuous, Not Discrete

The path and opacity are **continuous functions**, not point samples. Parameterized as smooth curves (e.g., control points with interpolation, or basis function expansion). This means:

- Between any two moments, the sound evolves smoothly
- A pitch glide (portamento) is a smooth path
- A pitch jump (staccato) is α→0 → teleport → α→1 at new frequency
- Vibrato is a sinusoidal modulation of the path

### 2.3 Opacity Handles Discontinuities

When a voice jumps from 200Hz to 400Hz, it doesn't sweep through 300Hz. The opacity α(t) drops to zero, the path jumps, then α(t) returns. The listener hears a clean jump, not a slide.

This is exactly how visual transmittance works: an object behind another is occluded, not blended through.

### 2.4 Transmittance for Multi-Layer Composition

Multiple layers composite via the rendering equation:

```
Audio(t, f) = Σᵢ αᵢ(t) · K(f, pathᵢ(t), Σᵢ) · timbreᵢ · Tᵢ(t)

where Tᵢ(t) = ∏ⱼ<ᵢ (1 - αⱼ(t) · K(f, pathⱼ(t), Σⱼ))
```

Louder layers mask quieter ones at similar frequencies — exactly psychoacoustic masking. A loud drum hit temporarily masks a quiet violin at nearby frequencies.

### 2.5 Manipulation

Because each layer is a named, independent Gaussian:
- **Pitch shift one voice:** shift pathᵢ(t) up/down
- **Remove vocals:** delete that layer
- **Time stretch:** scale t axis of pathᵢ and αᵢ
- **Change volume:** scale αᵢ
- **Voice cloning:** replace one layer's timbre with another's
- **Add harmony:** duplicate a layer with path offset by a musical interval

---

## 3. Mathematical Formulation

### 3.1 Layer Parameters

For K layers over T time frames and F frequency bins:

```python
class AudioLayer:
    mu_f:           float           # center frequency
    log_sigma_f:    float           # log bandwidth
    path_ctrl:      [N_ctrl]        # frequency trajectory control points
    alpha_ctrl:     [N_ctrl]        # opacity trajectory control points
    timbre:         [N_harmonics]   # harmonic amplitudes (optional)
```

Path and opacity are interpolated from control points to arbitrary time resolution.

N_ctrl = T // stride (e.g., stride=4 → one control point per 4 frames ≈ 46ms at 22050Hz/256hop)

### 3.2 Rendering

At time frame t and frequency bin f:

```
# Interpolate path and alpha at time t
freq_t = mu_f + interp(path_ctrl, t)      # current frequency
alpha_t = sigmoid(interp(alpha_ctrl, t))   # current opacity [0,1]
sigma_f = exp(log_sigma_f)                 # bandwidth

# Gaussian kernel in frequency
K = exp(-0.5 * (f - freq_t)^2 / sigma_f^2)

# This layer's contribution at (t, f)
contribution = alpha_t * K
```

The full spectrogram:
```
S(t, f) = Σ_layers contribution_i(t, f) * T_i(t, f)
```

### 3.3 Parameter Count

| Component | Per Layer | K=10 layers | K=20 layers |
|---|---|---|---|
| μ_f, σ_f | 2 | 20 | 40 |
| path (108 ctrl pts) | 108 | 1,080 | 2,160 |
| alpha (108 ctrl pts) | 108 | 1,080 | 2,160 |
| timbre (16 harmonics) | 16 | 160 | 320 |
| **Total** | **234** | **2,340** | **4,680** |

**10 layers = 2,340 parameters** for 5 seconds of audio. Compare to Variant A's 7,000 parameters (1000 Gaussians × 7) that produced worse results. And these parameters are *structurally meaningful*.

---

## 4. Experiment Design

### Phase 0: Can 10-20 layers reconstruct speech?

**Setup:**
1. Take JFK 5-second clip
2. Compute STFT magnitude (target)
3. Initialize K=10 layers with random frequencies, flat paths, uniform opacity
4. Render spectrogram from layers
5. L1 loss against target
6. Optimize via gradient descent
7. Griffin-Lim to audio

**Key question:** With only 10-20 layers, can we capture the essential structure of speech?

**Expected result:** Coarse but recognizable — the layers should learn the main formant tracks and energy contours. Fine harmonic detail may be missing with few layers.

### Phase 1: Automatic source separation

**Setup:**
1. Mix two known audio sources (voice + music)
2. Fit layers to the mixture
3. Do the layers naturally separate into voice-layer and music-layer?

**Why this might work:** The Gaussians have different μ_f and Σ_f. Voice (80-3000Hz, narrow formants) and music (wide frequency range) should naturally separate because they occupy different Gaussian regions.

### Phase 2: Layer manipulation

**Setup:**
1. Fit layers to speech
2. Shift one layer's path up by a musical fifth (×1.5 frequency)
3. Render → listen
4. Does it sound like the voice pitched up cleanly?

---

## 5. Advantages Over Variant A

| Property | Variant A (blobs) | Variant B (layers) |
|---|---|---|
| Gaussians needed | 1000+ | 10-20 |
| Parameters | 7,000+ | 2,340 |
| Structural meaning | None (dots on spectrogram) | Each = a sound source |
| Editability | Move dots (meaningless) | Shift pitch, remove layer |
| Continuity | Discrete blobs | Continuous paths |
| Compression | 28× (poor) | 90× (good) |
| Musical meaning | None | Layers ≈ tracks |

---

## 6. Connection to Prior Art

**Sinusoidal Modeling (SMS, Serra & Smith 1990):** Represented audio as tracked sinusoidal partials with frequency/amplitude trajectories + noise residual. Variant B is an evolution: replaces single-frequency partials with Gaussian-bandwidth layers, adds transmittance for inter-layer masking, and uses the proven rendering equation for composition.

**WORLD Vocoder:** Decomposes speech into F0 trajectory + spectral envelope + aperiodicity. Similar decomposition spirit, but WORLD uses fixed analysis frames while Variant B uses continuous Gaussians.

**Source Separation (Demucs, etc.):** Neural networks that separate audio into stems. Variant B would do this via Gaussian fitting — a structural decomposition rather than a learned one.
