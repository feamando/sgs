#!/usr/bin/env python3
"""
Klang: Fit Gaussians directly to STFT magnitude, reconstruct with Griffin-Lim.

Bypasses mel entirely — cleanest audio path.
Run on a powerful machine (Ryzen 9 / RTX 4090).

Usage:
    python klang/run_stft_experiment.py
    python klang/run_stft_experiment.py --n_gaussians 3000 --n_steps 3000
    python klang/run_stft_experiment.py --device cuda  # Use GPU if available
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try librosa first, fall back to manual STFT
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("librosa not found — using torch STFT")

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


class AudioGaussianScene(nn.Module):
    """2D Gaussians in time-frequency space with correlation."""

    def __init__(self, n_gaussians, T, F, device='cpu'):
        super().__init__()
        self.n = n_gaussians
        self.T = T
        self.F = F

        self.mu_t = nn.Parameter(torch.rand(n_gaussians, device=device) * T)
        self.mu_f = nn.Parameter(torch.rand(n_gaussians, device=device) * F)
        self.log_sigma_t = nn.Parameter(torch.randn(n_gaussians, device=device) * 0.5 + 1.0)
        self.log_sigma_f = nn.Parameter(torch.randn(n_gaussians, device=device) * 0.5 + 0.5)
        self.raw_rho = nn.Parameter(torch.zeros(n_gaussians, device=device))
        self.raw_alpha = nn.Parameter(torch.randn(n_gaussians, device=device) * 0.5)

    def render(self, t_grid, f_grid):
        T, F = len(t_grid), len(f_grid)
        t = t_grid.view(T, 1, 1)
        f = f_grid.view(1, F, 1)
        mu_t = self.mu_t.view(1, 1, -1)
        mu_f = self.mu_f.view(1, 1, -1)
        sigma_t = torch.exp(self.log_sigma_t).view(1, 1, -1)
        sigma_f = torch.exp(self.log_sigma_f).view(1, 1, -1)
        rho = torch.tanh(self.raw_rho).view(1, 1, -1)
        alpha = self.raw_alpha.view(1, 1, -1)

        dt = (t - mu_t) / sigma_t
        df = (f - mu_f) / sigma_f
        denom = 1.0 - rho ** 2 + 1e-6
        z = (dt ** 2 - 2 * rho * dt * df + df ** 2) / denom
        K = torch.exp(-0.5 * z)
        return (alpha * K).sum(dim=-1)


def griffin_lim_torch(magnitude, n_fft=1024, hop_length=256, n_iter=200):
    """Griffin-Lim using torch for potential GPU acceleration."""
    # Use numpy/librosa for reliability
    mag_np = magnitude.cpu().numpy() if isinstance(magnitude, torch.Tensor) else magnitude
    if HAS_LIBROSA:
        return librosa.griffinlim(mag_np, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)
    else:
        # Manual Griffin-Lim
        angles = np.exp(2j * np.pi * np.random.random(mag_np.shape))
        complex_spec = mag_np * angles
        window = np.hanning(n_fft)
        for _ in range(n_iter):
            n_frames = complex_spec.shape[1]
            sig_len = n_fft + hop_length * (n_frames - 1)
            signal = np.zeros(sig_len)
            win_sum = np.zeros(sig_len)
            for t in range(n_frames):
                s = t * hop_length
                signal[s:s+n_fft] += np.fft.irfft(complex_spec[:, t], n=n_fft) * window
                win_sum[s:s+n_fft] += window ** 2
            win_sum[win_sum < 1e-8] = 1e-8
            signal /= win_sum
            nf = (len(signal) - n_fft) // hop_length + 1
            new_spec = np.zeros((n_fft//2+1, nf), dtype=complex)
            for t in range(nf):
                s = t * hop_length
                new_spec[:, t] = np.fft.rfft(signal[s:s+n_fft] * window)
            angles = np.exp(1j * np.angle(new_spec[:, :mag_np.shape[1]]))
            complex_spec = mag_np * angles
        # Final ISTFT
        n_frames = complex_spec.shape[1]
        sig_len = n_fft + hop_length * (n_frames - 1)
        signal = np.zeros(sig_len)
        win_sum = np.zeros(sig_len)
        for t in range(n_frames):
            s = t * hop_length
            signal[s:s+n_fft] += np.fft.irfft(complex_spec[:, t], n=n_fft) * window
            win_sum[s:s+n_fft] += window ** 2
        win_sum[win_sum < 1e-8] = 1e-8
        return (signal / win_sum).astype(np.float32)


def load_audio(path, sr=22050):
    """Load audio file."""
    if HAS_LIBROSA:
        return librosa.load(path, sr=sr)
    elif HAS_SF:
        y, orig_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), orig_sr
    else:
        raise RuntimeError("Need librosa or soundfile to load audio")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    sr = 22050
    n_fft = 1024
    hop_length = 256

    # Load audio
    print(f"\nLoading {args.audio}...")
    y, _ = load_audio(args.audio, sr=sr)
    print(f"  {len(y)} samples, {len(y)/sr:.1f}s")

    # STFT magnitude
    print("Computing STFT...")
    if HAS_LIBROSA:
        S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_mag = np.abs(S_complex)
    else:
        waveform = torch.from_numpy(y).float()
        window = torch.hann_window(n_fft)
        S_complex = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
        S_mag = S_complex.abs().numpy()

    print(f"  STFT: {S_mag.shape} (freq × frames)")
    s_min, s_max = S_mag.min(), S_mag.max()
    target_np = ((S_mag - s_min) / (s_max - s_min + 1e-8)).T
    target = torch.from_numpy(target_np).float().to(device)
    T, F = target.shape
    print(f"  Target: {T} frames × {F} freq bins")

    # Fit Gaussians
    for n_g in args.n_gaussians:
        print(f"\n{'='*60}")
        print(f"  {n_g} Gaussians on STFT ({F} bins), {args.n_steps} steps")
        print(f"{'='*60}")

        scene = AudioGaussianScene(n_g, T, F, device=device).to(device)
        optimizer = torch.optim.Adam(scene.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_steps, eta_min=args.lr * 0.01,
        )

        t_grid = torch.arange(T, dtype=torch.float32, device=device)
        f_grid = torch.arange(F, dtype=torch.float32, device=device)

        t0 = time.time()
        for step in range(args.n_steps):
            rendered = scene.render(t_grid, f_grid)
            loss = torch.nn.functional.l1_loss(rendered, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
                eta = (args.n_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                print(f"  Step {step:4d}/{args.n_steps} | L1={loss.item():.4f} | "
                      f"{steps_per_sec:.1f} step/s | ETA {eta:.0f}s", flush=True)

        print(f"  Final L1: {loss.item():.4f} ({time.time()-t0:.0f}s total)")

        # Render
        with torch.no_grad():
            rendered_norm = scene.render(t_grid, f_grid).cpu().numpy()

        # Denormalize
        rendered_mag = np.maximum((rendered_norm * (s_max - s_min) + s_min).T, 0)

        # Griffin-Lim
        print("  Griffin-Lim (200 iterations)...")
        y_recon = griffin_lim_torch(rendered_mag, n_fft=n_fft, hop_length=hop_length, n_iter=200)
        y_recon = y_recon / (np.max(np.abs(y_recon)) + 1e-8) * 0.9
        y_recon = y_recon[:len(y)]

        out_path = f'klang/stft_{n_g}g.wav'
        if HAS_SF:
            sf.write(out_path, y_recon, sr)
        else:
            import scipy.io.wavfile
            scipy.io.wavfile.write(out_path, sr, (y_recon * 32767).astype(np.int16))
        print(f"  Saved: {out_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print("\nCompare with klang/diag_2_griffinlim_stft.wav (upper bound)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Klang: STFT Gaussian fitting")
    parser.add_argument('--audio', default='klang/test_clip.wav')
    parser.add_argument('--n_gaussians', type=int, nargs='+', default=[1500, 3000])
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    main(args)
