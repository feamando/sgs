#!/usr/bin/env python3
"""
Klang Phase 0: Can Gaussians reconstruct a mel spectrogram?

Fits a collection of 2D Gaussians in time-frequency space to a target
mel spectrogram via gradient descent. Uses the alpha-compositing
rendering equation from 3DGS adapted to 2D audio.

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 klang/phase0_experiment.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ═══════════════════════════════════════
# Audio loading (minimal, no librosa)
# ═══════════════════════════════════════

def load_wav(path):
    """Load a WAV file using soundfile."""
    import soundfile as sf
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono
    return y.astype(np.float32), sr


def compute_mel_spectrogram(y, sr, n_mels=80, n_fft=1024, hop_length=256):
    """Compute mel spectrogram using torch."""
    waveform = torch.from_numpy(y).float().unsqueeze(0)

    # Mel filterbank
    n_freqs = n_fft // 2 + 1
    mel_lo, mel_hi = 2595 * np.log10(1 + 0 / 700), 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        for j in range(bins[i], bins[i + 1]):
            filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(bins[i + 1], bins[i + 2]):
            filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

    filterbank = torch.from_numpy(filterbank).float()

    # STFT
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform.squeeze(0), n_fft, hop_length, window=window, return_complex=True)
    power = stft.abs() ** 2  # [n_freqs, T]

    # Apply mel filterbank
    mel = filterbank @ power  # [n_mels, T]

    # Convert to dB
    mel_db = 10 * torch.log10(mel.clamp(min=1e-10))

    return mel_db


# ═══════════════════════════════════════
# Audio Gaussian Scene
# ═══════════════════════════════════════

class AudioGaussianScene(nn.Module):
    """
    A collection of 2D Gaussians in time-frequency space.

    Each Gaussian has:
    - mu_t, mu_f: center in time and frequency
    - sigma_t, sigma_f: spread (via log for positivity)
    - rho: time-frequency correlation (captures chirps/glides)
    - alpha: amplitude (via raw param + sigmoid or softplus)
    """

    def __init__(self, n_gaussians, T, F):
        super().__init__()
        self.n = n_gaussians
        self.T = T
        self.F = F

        # Initialize positions uniformly across the spectrogram
        self.mu_t = nn.Parameter(torch.rand(n_gaussians) * T)
        self.mu_f = nn.Parameter(torch.rand(n_gaussians) * F)

        # Initialize spreads (log scale) — small to start
        self.log_sigma_t = nn.Parameter(torch.randn(n_gaussians) * 0.5 + 1.0)
        self.log_sigma_f = nn.Parameter(torch.randn(n_gaussians) * 0.5 + 0.5)

        # Correlation (for chirps) — start at 0
        self.raw_rho = nn.Parameter(torch.zeros(n_gaussians))

        # Amplitude — start moderate
        self.raw_alpha = nn.Parameter(torch.randn(n_gaussians) * 0.5)

    def render(self, t_grid, f_grid):
        """
        Render spectrogram at given time-frequency grid.

        t_grid: [T] time indices
        f_grid: [F] frequency indices
        Returns: [T, F] rendered amplitude
        """
        T, F = len(t_grid), len(f_grid)

        # Grids: [T, 1, 1] and [1, F, 1]
        t = t_grid.view(T, 1, 1)
        f = f_grid.view(1, F, 1)

        # Gaussian params: [1, 1, N]
        mu_t = self.mu_t.view(1, 1, -1)
        mu_f = self.mu_f.view(1, 1, -1)
        sigma_t = torch.exp(self.log_sigma_t).view(1, 1, -1)
        sigma_f = torch.exp(self.log_sigma_f).view(1, 1, -1)
        rho = torch.tanh(self.raw_rho).view(1, 1, -1)
        alpha = self.raw_alpha.view(1, 1, -1)  # unbounded, can be negative (dB scale)

        # 2D Gaussian with correlation
        dt = (t - mu_t) / sigma_t
        df = (f - mu_f) / sigma_f
        denom = 1.0 - rho ** 2 + 1e-6
        z = (dt ** 2 - 2 * rho * dt * df + df ** 2) / denom
        K = torch.exp(-0.5 * z)  # [T, F, N]

        # Weighted sum (amplitude × kernel)
        # Using raw alpha (can be negative for dB-scale spectrograms)
        weighted = alpha * K  # [T, F, N]
        spectrogram = weighted.sum(dim=-1)  # [T, F]

        return spectrogram

    def get_info(self):
        """Return summary stats."""
        with torch.no_grad():
            sigma_t = torch.exp(self.log_sigma_t)
            sigma_f = torch.exp(self.log_sigma_f)
            rho = torch.tanh(self.raw_rho)
            return {
                'n': self.n,
                'sigma_t_mean': sigma_t.mean().item(),
                'sigma_f_mean': sigma_f.mean().item(),
                'rho_mean': rho.abs().mean().item(),
                'alpha_range': (self.raw_alpha.min().item(), self.raw_alpha.max().item()),
            }


# ═══════════════════════════════════════
# Training
# ═══════════════════════════════════════

def fit_spectrogram(target, n_gaussians=500, n_steps=3000, lr=0.02, output_dir='klang/results'):
    """
    Fit Gaussians to a target spectrogram.

    target: [T, F] tensor — the mel spectrogram to reconstruct
    """
    os.makedirs(output_dir, exist_ok=True)
    T, F = target.shape
    print(f"Target spectrogram: {T} frames × {F} mel bins")
    print(f"Fitting {n_gaussians} Gaussians for {n_steps} steps...")

    scene = AudioGaussianScene(n_gaussians, T, F)
    optimizer = torch.optim.Adam(scene.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    t_grid = torch.arange(T, dtype=torch.float32)
    f_grid = torch.arange(F, dtype=torch.float32)

    losses = []
    t0 = time.time()

    for step in range(n_steps):
        rendered = scene.render(t_grid, f_grid)
        loss = TF.l1_loss(rendered, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 200 == 0 or step == n_steps - 1:
            info = scene.get_info()
            dt = time.time() - t0
            print(f"  Step {step:4d}/{n_steps} | L1={loss.item():.4f} | "
                  f"σ_t={info['sigma_t_mean']:.2f} σ_f={info['sigma_f_mean']:.2f} "
                  f"|ρ|={info['rho_mean']:.3f} | {dt:.1f}s")

    # Save results
    with torch.no_grad():
        final_rendered = scene.render(t_grid, f_grid).numpy()

    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].imshow(target.numpy().T, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Target (Ground Truth)')
    axes[0].set_ylabel('Mel bin')

    axes[1].imshow(final_rendered.T, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title(f'Reconstructed ({n_gaussians} Gaussians, L1={losses[-1]:.4f})')
    axes[1].set_ylabel('Mel bin')

    diff = np.abs(target.numpy() - final_rendered)
    axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='hot')
    axes[2].set_title(f'Error (mean={diff.mean():.4f}, max={diff.max():.4f})')
    axes[2].set_ylabel('Mel bin')
    axes[2].set_xlabel('Time frame')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction.png'), dpi=150)
    plt.close()
    print(f"\nSaved reconstruction.png")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color='#F4A300')
    ax.set_xlabel('Step')
    ax.set_ylabel('L1 Loss')
    ax.set_title('Reconstruction Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"Saved loss_curve.png")

    # Plot Gaussian positions
    fig, ax = plt.subplots(figsize=(14, 5))
    with torch.no_grad():
        sigma_t = torch.exp(scene.log_sigma_t).numpy()
        sigma_f = torch.exp(scene.log_sigma_f).numpy()
        alphas = scene.raw_alpha.numpy()
        # Normalize alpha for color
        a_norm = (alphas - alphas.min()) / (alphas.max() - alphas.min() + 1e-8)

    ax.imshow(target.numpy().T, aspect='auto', origin='lower', cmap='magma', alpha=0.3)
    scatter = ax.scatter(
        scene.mu_t.detach().numpy(),
        scene.mu_f.detach().numpy(),
        s=sigma_t * sigma_f * 10,
        c=a_norm,
        cmap='YlOrRd',
        alpha=0.6,
        edgecolors='white',
        linewidth=0.3,
    )
    ax.set_xlabel('Time frame')
    ax.set_ylabel('Mel bin')
    ax.set_title(f'Gaussian Positions ({n_gaussians} splats)')
    ax.set_xlim(0, T)
    ax.set_ylim(0, F)
    plt.colorbar(scatter, label='Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gaussian_positions.png'), dpi=150)
    plt.close()
    print(f"Saved gaussian_positions.png")

    return scene, losses


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("KLANG Phase 0: Spectrogram Reconstruction via Gaussians")
    print("=" * 60)

    wav_path = 'klang/test_clip.wav'
    if not os.path.exists(wav_path):
        print(f"ERROR: {wav_path} not found. Convert your audio to wav first.")
        exit(1)

    # Load audio
    print(f"\nLoading {wav_path}...")
    y, sr = load_wav(wav_path)
    print(f"  Samples: {len(y)}, Sample rate: {sr}, Duration: {len(y)/sr:.1f}s")

    # Compute mel spectrogram
    print("\nComputing mel spectrogram...")
    mel_db = compute_mel_spectrogram(y, sr, n_mels=80, n_fft=1024, hop_length=256)
    # Transpose to [T, F] and normalize to [0, 1]
    mel_db = mel_db.T  # [T, F]
    target = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    print(f"  Shape: {target.shape} (frames × mels)")
    print(f"  Range: [{mel_db.min():.1f}, {mel_db.max():.1f}] dB")

    # Run experiments with different Gaussian counts
    for n_g in [200, 500, 1000]:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {n_g} Gaussians")
        print(f"{'='*60}")
        scene, losses = fit_spectrogram(
            target, n_gaussians=n_g, n_steps=3000, lr=0.02,
            output_dir=f'klang/results_{n_g}g',
        )
        print(f"\n  Final L1: {losses[-1]:.4f}")

    print(f"\n{'='*60}")
    print("PHASE 0 COMPLETE")
    print(f"{'='*60}")
    print("\nCheck klang/results_*/reconstruction.png for visual comparison")
    print("Check klang/results_*/gaussian_positions.png to see where splats landed")
