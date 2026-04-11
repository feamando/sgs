#!/usr/bin/env python3
"""
Klang: Convert Gaussian-reconstructed spectrogram back to audio.

Uses Griffin-Lim algorithm (no neural vocoder needed) to estimate
phase from magnitude spectrogram and produce a WAV file.

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python3 klang/reconstruct_audio.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

from klang.phase0_experiment import (
    load_wav, compute_mel_spectrogram, AudioGaussianScene
)


def mel_to_stft(mel_spec, sr=22050, n_fft=1024, n_mels=80):
    """
    Approximate inverse mel transform: mel spectrogram → linear STFT magnitude.
    Uses pseudoinverse of mel filterbank.
    """
    n_freqs = n_fft // 2 + 1

    # Build mel filterbank (same as in compute_mel_spectrogram)
    mel_lo = 2595 * np.log10(1 + 0 / 700)
    mel_hi = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        for j in range(bins[i], bins[i + 1]):
            filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(bins[i + 1], bins[i + 2]):
            filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])

    # Pseudoinverse to go back from mel to linear
    pinv = np.linalg.pinv(filterbank)  # [n_freqs, n_mels]
    stft_mag = pinv @ mel_spec  # [n_freqs, T]
    stft_mag = np.maximum(stft_mag, 0)  # Ensure non-negative
    return stft_mag


def griffin_lim(magnitude, n_fft=1024, hop_length=256, n_iter=60):
    """
    Griffin-Lim algorithm: estimate phase from magnitude spectrogram.
    Iteratively applies STFT → magnitude replacement → ISTFT.
    """
    # Random initial phase
    angles = np.exp(2j * np.pi * np.random.random(magnitude.shape))
    complex_spec = magnitude * angles

    window = np.hanning(n_fft)

    for i in range(n_iter):
        # ISTFT
        n_frames = complex_spec.shape[1]
        signal_len = n_fft + hop_length * (n_frames - 1)
        signal = np.zeros(signal_len)
        window_sum = np.zeros(signal_len)

        for t in range(n_frames):
            start = t * hop_length
            frame = np.fft.irfft(complex_spec[:, t], n=n_fft)
            signal[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2

        # Normalize
        window_sum[window_sum < 1e-8] = 1e-8
        signal = signal / window_sum

        # STFT again
        n_frames_new = (len(signal) - n_fft) // hop_length + 1
        complex_spec_new = np.zeros((n_fft // 2 + 1, n_frames_new), dtype=complex)
        for t in range(n_frames_new):
            start = t * hop_length
            frame = signal[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            complex_spec_new[:, t] = spectrum

        # Replace magnitude, keep estimated phase
        angles = np.exp(1j * np.angle(complex_spec_new[:, :magnitude.shape[1]]))
        complex_spec = magnitude * angles

    # Final ISTFT
    n_frames = complex_spec.shape[1]
    signal_len = n_fft + hop_length * (n_frames - 1)
    signal = np.zeros(signal_len)
    window_sum = np.zeros(signal_len)

    for t in range(n_frames):
        start = t * hop_length
        frame = np.fft.irfft(complex_spec[:, t], n=n_fft)
        signal[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2

    window_sum[window_sum < 1e-8] = 1e-8
    signal = signal / window_sum

    return signal.astype(np.float32)


def main():
    print("=" * 60)
    print("KLANG: Audio Reconstruction from Gaussians")
    print("=" * 60)

    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80

    # Load original audio for reference
    print("\nLoading original audio...")
    y_orig, _ = load_wav('klang/test_clip.wav')
    print(f"  Original: {len(y_orig)} samples, {len(y_orig)/sr:.1f}s")

    # Compute target mel spectrogram (same as in phase0)
    print("Computing mel spectrogram...")
    mel_db = compute_mel_spectrogram(y_orig, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = mel_db.T  # [T, F]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    target = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
    T, F = target.shape
    print(f"  Spectrogram: {T} frames × {F} mels, range [{mel_min:.1f}, {mel_max:.1f}] dB")

    # For each Gaussian count, fit and reconstruct audio
    for n_g in [200, 500, 1000]:
        print(f"\n{'='*60}")
        print(f"  Reconstructing with {n_g} Gaussians")
        print(f"{'='*60}")

        # Fit Gaussians
        scene = AudioGaussianScene(n_g, T, F)
        optimizer = torch.optim.Adam(scene.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.0002)

        t_grid = torch.arange(T, dtype=torch.float32)
        f_grid = torch.arange(F, dtype=torch.float32)

        print("  Fitting Gaussians...")
        for step in range(3000):
            rendered = scene.render(t_grid, f_grid)
            loss = torch.nn.functional.l1_loss(rendered, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 1000 == 0:
                print(f"    Step {step}: L1={loss.item():.4f}")

        print(f"    Final L1: {loss.item():.4f}")

        # Render final spectrogram
        with torch.no_grad():
            rendered_norm = scene.render(t_grid, f_grid).numpy()  # [T, F], normalized [0,1]

        # Denormalize back to dB
        rendered_db = rendered_norm * (mel_max.item() - mel_min.item()) + mel_min.item()

        # Convert dB back to power
        rendered_power = 10 ** (rendered_db / 10)  # [T, F]

        # Inverse mel → linear STFT magnitude
        print("  Inverting mel transform...")
        stft_magnitude = mel_to_stft(rendered_power.T, sr=sr, n_fft=n_fft, n_mels=n_mels)
        stft_magnitude = np.sqrt(np.maximum(stft_magnitude, 0))  # power → amplitude

        # Griffin-Lim phase estimation
        print("  Running Griffin-Lim (60 iterations)...")
        audio = griffin_lim(stft_magnitude, n_fft=n_fft, hop_length=hop_length, n_iter=60)

        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9

        # Trim to original length
        audio = audio[:len(y_orig)]

        # Save
        out_path = f'klang/reconstructed_{n_g}g.wav'
        sf.write(out_path, audio, sr)
        print(f"  Saved: {out_path} ({len(audio)/sr:.1f}s)")

    # Also save original as reference
    sf.write('klang/original.wav', y_orig, sr)
    print(f"\nSaved: klang/original.wav (reference)")

    print(f"\n{'='*60}")
    print("DONE — Listen and compare!")
    print(f"{'='*60}")
    print("\n  klang/original.wav          — Original JFK speech")
    print("  klang/reconstructed_200g.wav — 200 Gaussians")
    print("  klang/reconstructed_500g.wav — 500 Gaussians")
    print("  klang/reconstructed_1000g.wav — 1000 Gaussians")


if __name__ == '__main__':
    main()
