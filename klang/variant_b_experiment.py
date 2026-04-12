#!/usr/bin/env python3
"""
Klang Variant B: Layer-Based Audio Gaussian Splatting

Each Gaussian IS a sound layer (voice, instrument, track) with:
- Center frequency (μ_f) and bandwidth (Σ_f)
- Continuous frequency trajectory path(t)
- Continuous opacity trajectory α(t)
- Optional harmonic/timbre features

Fits K layers to a target STFT, where K = 10-50 (not 1000+).

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 klang/variant_b_experiment.py
    python klang/variant_b_experiment.py --device cuda --n_layers 20
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Install: pip install librosa soundfile")
    exit(1)


class AudioLayerScene(nn.Module):
    """
    K audio layers, each a Gaussian in frequency space with
    continuous path(t) and opacity(t) trajectories.
    """

    def __init__(self, n_layers, n_freqs, n_frames, ctrl_stride=4, n_harmonics=8):
        super().__init__()
        self.n_layers = n_layers
        self.n_freqs = n_freqs
        self.n_frames = n_frames
        self.ctrl_stride = ctrl_stride
        self.n_ctrl = n_frames // ctrl_stride + 1

        # Per-layer static parameters
        # Center frequency: spread across frequency range
        init_freqs = torch.linspace(0.05, 0.95, n_layers) * n_freqs
        self.mu_f = nn.Parameter(init_freqs)

        # Bandwidth (log scale): start moderate
        self.log_sigma_f = nn.Parameter(torch.ones(n_layers) * 2.0)

        # Frequency trajectory: control points (offsets from mu_f)
        # Shape: [n_layers, n_ctrl]
        self.path_ctrl = nn.Parameter(torch.zeros(n_layers, self.n_ctrl))

        # Opacity trajectory: control points (pre-sigmoid)
        # Initialize slightly positive so layers start audible
        self.alpha_ctrl = nn.Parameter(torch.ones(n_layers, self.n_ctrl) * 0.5)

        # Harmonic amplitudes: model overtones
        # harmonic_k at frequency k * center_freq
        # Shape: [n_layers, n_harmonics]
        self.harmonics = nn.Parameter(torch.zeros(n_layers, n_harmonics))
        # First harmonic (fundamental) starts strong
        with torch.no_grad():
            self.harmonics[:, 0] = 2.0

    def _interpolate_ctrl(self, ctrl, n_frames):
        """Interpolate control points to frame resolution."""
        # ctrl: [n_layers, n_ctrl]
        # Use linear interpolation
        ctrl_expanded = ctrl.unsqueeze(1)  # [n_layers, 1, n_ctrl]
        # Interpolate to n_frames
        interp = TF.interpolate(ctrl_expanded, size=n_frames, mode='linear', align_corners=True)
        return interp.squeeze(1)  # [n_layers, n_frames]

    def render(self):
        """
        Render the full spectrogram from all layers.

        Returns: [n_frames, n_freqs]
        """
        T = self.n_frames
        F = self.n_freqs

        # Interpolate trajectories to frame resolution
        path = self._interpolate_ctrl(self.path_ctrl, T)      # [K, T]
        alpha = torch.sigmoid(self._interpolate_ctrl(self.alpha_ctrl, T))  # [K, T]

        # Current frequency at each time: mu_f + path offset
        freq_t = self.mu_f.unsqueeze(1) + path  # [K, T]

        # Bandwidth
        sigma_f = torch.exp(self.log_sigma_f)  # [K]

        # Frequency grid
        f_grid = torch.arange(F, dtype=torch.float32, device=self.mu_f.device)  # [F]

        # Compute each layer's contribution
        # freq_t: [K, T] → [K, T, 1]
        # f_grid: [F] → [1, 1, F]
        freq_t_exp = freq_t.unsqueeze(2)        # [K, T, 1]
        f_exp = f_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, F]
        sigma_exp = sigma_f.unsqueeze(1).unsqueeze(2)  # [K, 1, 1]

        # Fundamental Gaussian kernel
        diff = f_exp - freq_t_exp
        K_fund = torch.exp(-0.5 * (diff / sigma_exp) ** 2)  # [K, T, F]

        # Add harmonics: each harmonic is a Gaussian at k * freq_t
        harmonic_weights = torch.softmax(self.harmonics, dim=1)  # [K, n_harm]
        K_total = K_fund * harmonic_weights[:, 0].unsqueeze(1).unsqueeze(2)

        for h in range(1, self.harmonics.shape[1]):
            harm_freq = freq_t_exp * (h + 1)  # harmonic frequency
            K_harm = torch.exp(-0.5 * ((f_exp - harm_freq) / (sigma_exp * 0.7)) ** 2)
            K_total = K_total + K_harm * harmonic_weights[:, h].unsqueeze(1).unsqueeze(2)

        # Apply opacity: [K, T, 1] * [K, T, F]
        alpha_exp = alpha.unsqueeze(2)  # [K, T, 1]
        weighted = alpha_exp * K_total   # [K, T, F]

        # Sum across layers (simple sum for now; transmittance in v2)
        spectrogram = weighted.sum(dim=0)  # [T, F]

        return spectrogram

    def get_layer_info(self):
        """Return per-layer summary for visualization."""
        with torch.no_grad():
            info = []
            path = self._interpolate_ctrl(self.path_ctrl, self.n_frames)
            alpha = torch.sigmoid(self._interpolate_ctrl(self.alpha_ctrl, self.n_frames))
            for i in range(self.n_layers):
                freq_traj = (self.mu_f[i] + path[i]).cpu().numpy()
                alpha_traj = alpha[i].cpu().numpy()
                info.append({
                    'mu_f': self.mu_f[i].item(),
                    'sigma_f': torch.exp(self.log_sigma_f[i]).item(),
                    'freq_traj': freq_traj,
                    'alpha_traj': alpha_traj,
                    'mean_alpha': alpha_traj.mean(),
                    'harmonics': torch.softmax(self.harmonics[i], dim=0).cpu().numpy(),
                })
            return info


def fit_layers(target, n_layers=10, n_steps=3000, lr=0.03, device='cpu'):
    """Fit audio layers to a target STFT magnitude."""
    T, F = target.shape
    target_t = torch.from_numpy(target).float().to(device)

    scene = AudioLayerScene(n_layers, F, T, ctrl_stride=4, n_harmonics=8).to(device)
    optimizer = torch.optim.Adam(scene.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    losses = []
    t0 = time.time()

    for step in range(n_steps):
        rendered = scene.render()
        loss = TF.l1_loss(rendered, target_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if step % 200 == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"  Step {step:4d}/{n_steps} | L1={loss.item():.4f} | "
                  f"{sps:.1f} step/s | {elapsed:.0f}s", flush=True)

    return scene, losses


def plot_results(target, scene, losses, output_dir):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        rendered = scene.render().cpu().numpy()

    # 1. Reconstruction comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].imshow(target.T, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Target STFT')
    axes[0].set_ylabel('Freq bin')

    axes[1].imshow(rendered.T, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title(f'Rendered ({scene.n_layers} layers, L1={losses[-1]:.4f})')
    axes[1].set_ylabel('Freq bin')

    diff = np.abs(target - rendered)
    axes[2].imshow(diff.T, aspect='auto', origin='lower', cmap='hot')
    axes[2].set_title(f'Error (mean={diff.mean():.4f})')
    axes[2].set_ylabel('Freq bin')
    axes[2].set_xlabel('Time frame')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction.png'), dpi=150)
    plt.close()

    # 2. Layer trajectories overlaid on spectrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(target.T, aspect='auto', origin='lower', cmap='magma', alpha=0.5)

    info = scene.get_layer_info()
    colors = plt.cm.Set1(np.linspace(0, 1, len(info)))

    for i, layer in enumerate(info):
        if layer['mean_alpha'] > 0.05:  # only show active layers
            t_axis = np.arange(len(layer['freq_traj']))
            # Plot frequency trajectory with opacity as line width
            ax.plot(t_axis, layer['freq_traj'], color=colors[i],
                    linewidth=1.5, alpha=0.8,
                    label=f'L{i}: μ={layer["mu_f"]:.0f} σ={layer["sigma_f"]:.1f} α={layer["mean_alpha"]:.2f}')

            # Show bandwidth as a shaded region
            sigma = layer['sigma_f']
            ax.fill_between(t_axis,
                           layer['freq_traj'] - sigma,
                           layer['freq_traj'] + sigma,
                           color=colors[i], alpha=0.1)

    ax.set_xlabel('Time frame')
    ax.set_ylabel('Frequency bin')
    ax.set_title(f'Layer Trajectories ({scene.n_layers} layers)')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.set_ylim(0, target.shape[1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectories.png'), dpi=150)
    plt.close()

    # 3. Opacity over time per layer
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, layer in enumerate(info):
        if layer['mean_alpha'] > 0.05:
            ax.plot(layer['alpha_traj'], color=colors[i], alpha=0.7,
                    label=f'L{i}: μ_f={layer["mu_f"]:.0f}')
    ax.set_xlabel('Time frame')
    ax.set_ylabel('Opacity α(t)')
    ax.set_title('Layer Opacity Trajectories')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'opacity.png'), dpi=150)
    plt.close()

    # 4. Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color='#F4A300')
    ax.set_xlabel('Step')
    ax.set_ylabel('L1 Loss')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=150)
    plt.close()

    print(f"  Saved plots to {output_dir}/")


def reconstruct_audio(scene, s_min, s_max, sr, n_fft, hop_length, orig_length, output_path):
    """Render spectrogram from layers, Griffin-Lim to audio."""
    with torch.no_grad():
        rendered_norm = scene.render().cpu().numpy()

    # Denormalize
    rendered_mag = np.maximum(rendered_norm * (s_max - s_min) + s_min, 0).T  # [F, T]

    # Griffin-Lim
    print("  Griffin-Lim (200 iterations)...")
    y = librosa.griffinlim(rendered_mag, n_iter=200, hop_length=hop_length, n_fft=n_fft,
                           length=orig_length)
    y = y / (np.max(np.abs(y)) + 1e-8) * 0.9

    sf.write(output_path, y, sr)
    print(f"  Saved: {output_path}")
    return y


def main(args):
    print("=" * 60)
    print("KLANG Variant B: Layer-Based Audio Gaussians")
    print("=" * 60)

    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    sr = 22050
    n_fft = 1024
    hop_length = 256

    # Load audio
    print(f"\nLoading {args.audio}...")
    y, _ = librosa.load(args.audio, sr=sr)
    print(f"  {len(y)} samples, {len(y)/sr:.1f}s")

    # STFT
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S_complex)  # [F, T]
    print(f"  STFT: {S_mag.shape}")

    # Normalize
    s_min, s_max = S_mag.min(), S_mag.max()
    target = ((S_mag - s_min) / (s_max - s_min + 1e-8)).T  # [T, F]
    T, F = target.shape
    print(f"  Target: {T} frames × {F} freq bins")

    # Run experiments with different layer counts
    for n_layers in args.n_layers:
        print(f"\n{'='*60}")
        print(f"  {n_layers} LAYERS")
        print(f"{'='*60}")

        n_params = n_layers * (2 + (T // 4 + 1) * 2 + 8)
        print(f"  ~{n_params:,} parameters")

        scene, losses = fit_layers(
            target, n_layers=n_layers, n_steps=args.n_steps,
            lr=args.lr, device=args.device,
        )

        out_dir = f'klang/variant_b_{n_layers}L'
        plot_results(target, scene, losses, out_dir)

        audio = reconstruct_audio(
            scene, s_min, s_max, sr, n_fft, hop_length, len(y),
            os.path.join(out_dir, 'audio.wav'),
        )

        # Print layer summary
        info = scene.get_layer_info()
        print(f"\n  Layer summary:")
        active = [l for l in info if l['mean_alpha'] > 0.05]
        for i, layer in enumerate(info):
            status = "ACTIVE" if layer['mean_alpha'] > 0.05 else "silent"
            print(f"    L{i:2d}: μ_f={layer['mu_f']:6.1f} σ_f={layer['sigma_f']:5.1f} "
                  f"α_mean={layer['mean_alpha']:.3f} [{status}]")
        print(f"  Active layers: {len(active)}/{n_layers}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print("\nCompare with klang/diag_2_griffinlim_stft.wav (upper bound)")
    print("Check */trajectories.png to see layer frequency paths")
    print("Check */opacity.png to see when each layer is active")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Klang Variant B")
    parser.add_argument('--audio', default='klang/test_clip.wav')
    parser.add_argument('--n_layers', type=int, nargs='+', default=[10, 20, 40])
    parser.add_argument('--n_steps', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    main(args)
