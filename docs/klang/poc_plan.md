# Klang Phase 0: Spectrogram Reconstruction PoC

**Question:** Can a collection of 2D Gaussians reconstruct a mel spectrogram via the rendering equation?

---

## Experiment

```
Input:  3-second audio clip → mel spectrogram (ground truth)
Model:  N Gaussians in (time, frequency) space
Output: Reconstructed spectrogram → audio via vocoder
Metric: L1 error, PESQ, human listening test
```

## Implementation

```python
# klang/audio_splat.py

class AudioGaussianScene:
    """A collection of 2D Gaussians representing audio in time-frequency space."""

    def __init__(self, n_gaussians=500, n_mel=80, duration_frames=300):
        # Gaussian parameters (all learnable)
        self.mu_t = nn.Parameter(torch.rand(n_gaussians) * duration_frames)  # time position
        self.mu_f = nn.Parameter(torch.rand(n_gaussians) * n_mel)            # freq position
        self.log_sigma_t = nn.Parameter(torch.zeros(n_gaussians))             # time spread
        self.log_sigma_f = nn.Parameter(torch.zeros(n_gaussians))             # freq spread
        self.rho = nn.Parameter(torch.zeros(n_gaussians))                     # correlation (chirps)
        self.raw_alpha = nn.Parameter(torch.zeros(n_gaussians))               # amplitude

    def render(self, t_grid, f_grid):
        """
        Render spectrogram at given time-frequency grid points.

        t_grid: [T] — time frame indices
        f_grid: [F] — mel frequency indices
        Returns: [T, F] — rendered spectrogram amplitude
        """
        T, F = len(t_grid), len(f_grid)

        # Expand grids: [T, F, 1] and Gaussians: [1, 1, N]
        t = t_grid.view(T, 1, 1)
        f = f_grid.view(1, F, 1)
        mu_t = self.mu_t.view(1, 1, -1)
        mu_f = self.mu_f.view(1, 1, -1)
        sigma_t = torch.exp(self.log_sigma_t).view(1, 1, -1)
        sigma_f = torch.exp(self.log_sigma_f).view(1, 1, -1)
        rho = torch.tanh(self.rho).view(1, 1, -1)
        alpha = torch.sigmoid(self.raw_alpha).view(1, 1, -1)

        # 2D Gaussian evaluation with correlation
        dt = (t - mu_t) / sigma_t
        df = (f - mu_f) / sigma_f
        z = dt**2 - 2*rho*dt*df + df**2
        K = torch.exp(-0.5 * z / (1 - rho**2 + 1e-6))

        # Alpha-compositing (sort by amplitude for transmittance)
        # Simplified: just weighted sum for Phase 0
        # Full transmittance in Phase 1
        weighted = alpha * K  # [T, F, N]
        spectrogram = weighted.sum(dim=-1)  # [T, F]

        return spectrogram


# Training loop
def fit_audio(audio_path, n_gaussians=500, n_steps=2000, lr=0.01):
    """Fit Gaussians to an audio file's mel spectrogram."""

    # Load audio → mel spectrogram
    waveform, sr = torchaudio.load(audio_path)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=80, n_fft=1024, hop_length=256,
    )
    mel = mel_transform(waveform).squeeze(0)  # [n_mels, T]
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
    # Normalize to [0, 1]
    target = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    target = target.T  # [T, n_mels]

    T, F = target.shape
    scene = AudioGaussianScene(n_gaussians, F, T)
    optimizer = torch.optim.Adam(scene.parameters(), lr=lr)

    t_grid = torch.arange(T, dtype=torch.float32)
    f_grid = torch.arange(F, dtype=torch.float32)

    for step in range(n_steps):
        rendered = scene.render(t_grid, f_grid)
        loss = F.l1_loss(rendered, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: L1={loss.item():.4f}, "
                  f"active={int((torch.sigmoid(scene.raw_alpha) > 0.01).sum())}/{n_gaussians}")

    return scene, target
```

## What to Measure

| Metric | How | Target |
|---|---|---|
| **L1 reconstruction error** | Mean absolute difference between rendered and target spectrogram | < 0.1 (normalized) |
| **Visual similarity** | Plot rendered vs target spectrogram side by side | Recognizable structure |
| **Audio quality** | Render → vocoder → listen | Recognizable speech/music |
| **Gaussians used** | Count active (α > 0.01) after pruning | < 1000 for 3-second clip |
| **Adaptive density** | Does splitting improve reconstruction? | Yes → validates the approach |

## What We Learn

| If reconstruction is... | Interpretation | Next step |
|---|---|---|
| **Good (L1 < 0.05)** | Gaussians can represent audio. Proceed to Phase 1. | Compression, editing, generation |
| **OK (0.05 < L1 < 0.15)** | Needs more Gaussians or better optimization. | Try full transmittance, more Gaussians, spectral loss |
| **Bad (L1 > 0.15)** | 2D Gaussians don't capture spectrogram structure well. | Try 1D on waveform, or GMM approach instead |

## Dependencies

```
torch, torchaudio
numpy, matplotlib (visualization)
Optional: HiFi-GAN or Griffin-Lim (vocoder for audio reconstruction)
```

## Data

Any short audio clip works. Good test cases:
- Single speaker saying a sentence (LJSpeech)
- Piano note (simple, should be easy)
- Bird call (chirp = tilted Gaussian, tests ρ parameter)
- Drum hit (transient = narrow σ_t)
