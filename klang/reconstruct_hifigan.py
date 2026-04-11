#!/usr/bin/env python3
"""
Klang: Reconstruct audio from Gaussians using HiFi-GAN vocoder.

Much higher quality than Griffin-Lim — neural vocoder estimates
both magnitude and phase coherently.

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib PYTHONPATH=. python3 klang/reconstruct_hifigan.py
"""

import os
import numpy as np
import torch
import soundfile as sf
from speechbrain.inference.vocoders import HIFIGAN

from klang.phase0_experiment import (
    load_wav, compute_mel_spectrogram, AudioGaussianScene
)


def main():
    print("=" * 60)
    print("KLANG: HiFi-GAN Audio Reconstruction from Gaussians")
    print("=" * 60)

    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80

    # Load HiFi-GAN
    print("\nLoading HiFi-GAN vocoder...")
    hifi = HIFIGAN.from_hparams(
        source='speechbrain/tts-hifigan-ljspeech',
        savedir='klang/hifigan_model',
    )
    print("  Loaded!")

    # Load original audio
    print("\nLoading original audio...")
    y_orig, _ = load_wav('klang/test_clip.wav')
    print(f"  {len(y_orig)} samples, {len(y_orig)/sr:.1f}s")

    # Compute target mel spectrogram
    print("Computing mel spectrogram...")
    mel_db = compute_mel_spectrogram(y_orig, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = mel_db.T  # [T, F]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    target = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
    T, F = target.shape
    print(f"  {T} frames × {F} mels")

    # First: reconstruct from the ORIGINAL spectrogram (upper bound quality)
    print("\n--- Reconstructing from original spectrogram (reference) ---")
    # HiFi-GAN expects mel in log scale, shape [batch, n_mels, T]
    mel_for_hifi = mel_db.T.unsqueeze(0)  # [1, 80, T]
    # Normalize to roughly match LJSpeech training distribution
    mel_for_hifi = (mel_for_hifi - mel_for_hifi.mean()) / (mel_for_hifi.std() + 1e-8)
    with torch.no_grad():
        waveform = hifi.decode_batch(mel_for_hifi)
    audio = waveform.squeeze().numpy()
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
    sf.write('klang/hifigan_from_original.wav', audio[:len(y_orig)], sr)
    print(f"  Saved: klang/hifigan_from_original.wav")

    # Now: reconstruct from Gaussian-rendered spectrograms
    for n_g in [200, 500, 1000]:
        print(f"\n{'='*60}")
        print(f"  {n_g} Gaussians → HiFi-GAN")
        print(f"{'='*60}")

        # Fit Gaussians
        scene = AudioGaussianScene(n_g, T, F)
        optimizer = torch.optim.Adam(scene.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=3000, eta_min=0.0002,
        )

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
            rendered_norm = scene.render(t_grid, f_grid)  # [T, F], normalized [0,1]

        # Denormalize back to dB
        rendered_db = rendered_norm * (mel_max.item() - mel_min.item()) + mel_min.item()

        # HiFi-GAN expects [batch, n_mels, T]
        mel_for_hifi = rendered_db.T.unsqueeze(0)  # [1, F, T]
        mel_for_hifi = (mel_for_hifi - mel_for_hifi.mean()) / (mel_for_hifi.std() + 1e-8)

        print("  Running HiFi-GAN vocoder...")
        with torch.no_grad():
            waveform = hifi.decode_batch(mel_for_hifi)

        audio = waveform.squeeze().numpy()
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        audio = audio[:len(y_orig)]

        out_path = f'klang/hifigan_{n_g}g.wav'
        sf.write(out_path, audio, sr)
        print(f"  Saved: {out_path} ({len(audio)/sr:.1f}s)")

    print(f"\n{'='*60}")
    print("DONE — Listen and compare!")
    print(f"{'='*60}")
    print("\n  klang/original.wav              — Original JFK")
    print("  klang/hifigan_from_original.wav  — HiFi-GAN from original spectrogram (upper bound)")
    print("  klang/hifigan_200g.wav           — HiFi-GAN from 200 Gaussians")
    print("  klang/hifigan_500g.wav           — HiFi-GAN from 500 Gaussians")
    print("  klang/hifigan_1000g.wav          — HiFi-GAN from 1000 Gaussians")


if __name__ == '__main__':
    main()
