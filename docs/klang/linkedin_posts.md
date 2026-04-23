# Klang LinkedIn Posts

Three-part LinkedIn series on the Klang audio track of the SGS project. Posts
continue an earlier LinkedIn thread about Semantic Gaussian Splatting and the
Planck language model.

Tone: formal, technical narrative. No em dashes. No "not X, but Y"
contrastives. No evaluative AI adverbs.

---

## Post 1 — Klang Variant A

Continuing the SGS series.

Earlier posts in this thread introduced Semantic Gaussian Splatting, the idea
that the compositing math behind 3D Gaussian Splatting transfers cleanly to
sequence modelling, along with early experiments on Planck, our small language
model built on that foundation. Those posts made the case for SGS as an
alternative to attention for text.

One claim deserved a separate test. If SGS is a real representational
primitive, it should not be text-only. It should behave in informative ways on
other modalities, whether it succeeds or fails. So I opened a second track,
Klang, to stress the multimodality claim on audio.

Variant A was the most direct translation possible. Treat the STFT magnitude
of a short speech clip as a 2D "image," scatter several thousand tiny
Gaussians across it, and fit their positions, bandwidths, and amplitudes with
the same splatting math used in 3D. No architectural novelty. The question
was simply whether the primitive carries over.

Result: the primitive carries over. The audio does not follow completely.
Reconstructions at 1500 and 3000 Gaussians fit magnitude well. The loss
curves converge. Speech is comprehensible. What is consistently wrong is
phase. The splat represents only magnitude, so decoding the waveform requires
Griffin-Lim to invent phase from scratch, and the invented phase does not
stay coherent frame to frame. The audible signature is a faint warble on
every syllable. Adding Gaussians narrows the warble. The warble does not go
away.

That is an informative failure. It tells us where the representation needs
to grow, and it does so without casting doubt on the direction. Variant B
takes the next step.

**Images (carousel order)**
- `klang/results_1000g/reconstruction.png`
- `klang/results_1000g/gaussian_positions.png`
- `klang/results_1000g/loss_curve.png`

---

## Post 2 — Klang Variant B

Continuing the SGS multimodality series.

The previous post described Variant A of Klang, our audio track for SGS.
Variant A fit thousands of small Gaussians to a spectrogram, produced
comprehensible but warbled speech, and located the problem precisely:
magnitude is represented, phase is not.

Variant B asks a different question. What if each Gaussian represented a
whole sound layer: a voice, an instrument, a harmonic component, with a
continuous frequency trajectory and a continuous opacity envelope over time?
This brings Klang closer to the sequential compositing used on the text side
in Planck, where atoms contribute along a causal axis.

We parameterized each layer with a center frequency, a bandwidth, time-
varying frequency offsets (control points interpolated to frame resolution),
time-varying opacity (same), and a small set of harmonic amplitudes. Ten,
twenty, and forty layers were fit to the same speech clip, under the same
loss Variant A used.

Three observations.

The learned layers behave the way the theory predicted. The trajectories
plot shows the optimizer recovering pitch contours, with opacity envelopes
that roughly align with phoneme boundaries. Qualitatively, this is the most
encouraging visual artefact the project has produced.

The phase warble from Variant A is gone. That failure mode has been
eliminated.

A new failure mode appeared, and it was identical across 10, 20, and 40
layers. All three reconstructions lose frequencies below approximately
200 Hz and carry a faint high-pitched tone near Nyquist. The fact that a
four-fold depth range fails the same way means depth is not the constraint.
The bounds on the Gaussian centers and bandwidths do not cover the frequency
range the signal actually occupies, so the spare capacity migrates to the
top of the representable range and manifests as a whine.

The experiment did what it was built to do. It told us precisely what to
fix. Klang 1.2 addresses both failure modes, Variant A's phase loss and
Variant B's bounds ceiling, with one unified set of changes. Next post.

**Images**
- `klang/variant_b_20L/trajectories.png`
- `klang/variant_b_20L/reconstruction.png`
- `klang/variant_b_20L/opacity.png`
- Optional: a second `trajectories.png` from `variant_b_10L/` or
  `variant_b_40L/` to emphasize the identical-across-depth point.

---

## Post 3 — Klang 1.2

Continuing the SGS multimodality series.

The previous two posts established two distinct audio failure modes in SGS.
Variant A produced a consistent phase warble because only magnitude was
represented. Variant B lost sub-200 Hz content and gained a near-Nyquist
whine because the Gaussian parameterization did not cover the signal's
frequency range. Both are informative failures. Both point at specific
things to change.

Klang 1.2 takes the diagnosis as a specification and makes seven changes.

1. Complex-valued Gaussians. Each atom now carries both amplitude and phase.
   Phase is fitted directly, removing the need for Griffin-Lim to invent it.
   This targets Variant A's warble.

2. Mel-scaled initialization and widened bounds on center frequency and
   bandwidth. The representable range now extends from approximately 40 Hz
   to Nyquist, with a cap on bandwidth to prevent capacity drift. This
   targets Variant B's bass loss and whine.

3. Multi-resolution STFT loss. Three window sizes, log-magnitude and
   spectral convergence. A standard neural-vocoder loss that reduces
   single-window over-fit.

4. Transmittance-budgeted alpha compositing. Layers are sorted by salience
   and rendered with a transmittance cap. This is the same compositing rule
   used in Planck and Hertz, so Klang and the text tracks now share a
   formal foundation.

5. A mel and HiFi-GAN decode bridge, so vocoder quality and splat quality
   can be evaluated independently.

6. Optional perceptual loss via a pretrained audio embedder, since spectral
   error does not correlate well with perceived quality.

7. A quantitative validator. Spectral MSE, log-magnitude MAE, MCD-13, and,
   when available, PESQ and STOI. Klang now has the same gate-based
   evaluation structure we use on the text tracks.

The aim here is modest. Each fix responds to a specific artefact audible in
the earlier runs. The representation is being extended in the directions the
failures asked us to extend it.

Training and validation runs are queued after the current Planck 1.2 work.
Results and the formal pass or fail against the gates will follow.

**Images**

Klang 1.2 has not been trained yet. Two options:

- Post without images.
- Carry over a Variant B trajectories plot framed as "the representation
  we are extending," and add the Klang 1.2 outputs
  (`klang/klang_1_2/reconstruction.png`, `trajectories.png`, `loss.png`)
  in a follow-up post once the run completes on Windows.
