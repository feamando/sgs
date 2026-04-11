#!/usr/bin/env python3
"""
Klang: HiFi-GAN reconstruction v2 — fixed mel normalization.

The key issue: HiFi-GAN from SpeechBrain expects mel spectrograms
in a specific format matching its LJSpeech training. We need to
match that format exactly.

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib PYTHONPATH=. python3 klang/reconstruct_hifigan_v2.py
"""

import os
import numpy as np
import torch
import soundfile as sf

from klang.phase0_experiment import AudioGaussianScene


def compute_mel_for_hifigan(y, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    """
    Compute mel spectrogram in the format HiFi-GAN expects.

    SpeechBrain's HiFi-GAN was trained with:
    - log mel spectrogram (natural log, not dB)
    - n_mels=80, n_fft=1024, hop_length=256
    - sr=22050
    - No additional normalization
    """
    waveform = torch.from_numpy(y).float()

    # STFT
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    magnitude = stft.abs()  # [n_freqs, T]

    # Mel filterbank
    n_freqs = n_fft // 2 + 1
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

    filterbank_t = torch.from_numpy(filterbank).float()

    # Apply mel filterbank to magnitude (not power)
    mel = filterbank_t @ magnitude  # [n_mels, T]

    # Log mel (natural log, clamped)
    log_mel = torch.log(mel.clamp(min=1e-5))

    return log_mel, mel, filterbank_t


def mel_to_stft_magnitude(mel_linear, filterbank):
    """Pseudoinverse of mel filterbank to get back STFT magnitude."""
    pinv = torch.linalg.pinv(filterbank)  # [n_freqs, n_mels]
    stft_mag = pinv @ mel_linear  # [n_freqs, T]
    stft_mag = stft_mag.clamp(min=0)
    return stft_mag


def griffin_lim(magnitude_np, n_fft=1024, hop_length=256, n_iter=100):
    """Griffin-Lim with more iterations for better quality."""
    angles = np.exp(2j * np.pi * np.random.random(magnitude_np.shape))
    complex_spec = magnitude_np * angles
    window = np.hanning(n_fft)

    for _ in range(n_iter):
        # ISTFT
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

        # STFT
        nf = (len(signal) - n_fft) // hop_length + 1
        new_spec = np.zeros((n_fft//2+1, nf), dtype=complex)
        for t in range(nf):
            s = t * hop_length
            new_spec[:, t] = np.fft.rfft(signal[s:s+n_fft] * window)
        angles = np.exp(1j * np.angle(new_spec[:, :magnitude_np.shape[1]]))
        complex_spec = magnitude_np * angles

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


def main():
    print("=" * 60)
    print("KLANG: HiFi-GAN Reconstruction v2 (fixed normalization)")
    print("=" * 60)

    sr = 22050
    n_fft = 1024
    hop_length = 256
    n_mels = 80

    # Load original
    print("\nLoading audio...")
    y_orig, _ = sf.read('klang/test_clip.wav')
    y_orig = y_orig.astype(np.float32)
    if y_orig.ndim > 1:
        y_orig = y_orig.mean(axis=1)
    print(f"  {len(y_orig)} samples, {len(y_orig)/sr:.1f}s")

    # Compute mel in HiFi-GAN format
    print("Computing mel spectrogram (log scale for HiFi-GAN)...")
    log_mel, mel_linear, filterbank = compute_mel_for_hifigan(y_orig, sr, n_fft, hop_length, n_mels)
    print(f"  Log mel: {log_mel.shape}, range [{log_mel.min():.2f}, {log_mel.max():.2f}]")

    # Also compute normalized version for Gaussian fitting
    # Normalize log mel to [0, 1] for fitting, denormalize after
    mel_min = log_mel.min()
    mel_max = log_mel.max()
    target_norm = (log_mel.T - mel_min) / (mel_max - mel_min + 1e-8)  # [T, n_mels]
    T, F = target_norm.shape
    print(f"  Target for fitting: {T} × {F}")

    # ─── Step 1: HiFi-GAN from original spectrogram ───
    print("\n--- HiFi-GAN from ORIGINAL spectrogram ---")
    try:
        from speechbrain.inference.vocoders import HIFIGAN
        hifi = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech', savedir='klang/hifigan_model')

        # Feed log mel directly [batch, n_mels, T]
        mel_input = log_mel.unsqueeze(0)  # [1, 80, T]
        with torch.no_grad():
            waveform = hifi.decode_batch(mel_input)
        audio = waveform.squeeze().numpy()
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
        sf.write('klang/hifigan_v2_original.wav', audio[:len(y_orig)], sr)
        print(f"  Saved: klang/hifigan_v2_original.wav")
        use_hifigan = True
    except Exception as e:
        print(f"  HiFi-GAN failed: {e}")
        print("  Falling back to Griffin-Lim only")
        use_hifigan = False

    # ─── Step 2: Griffin-Lim from original (improved, 100 iter) ───
    print("\n--- Griffin-Lim from ORIGINAL (100 iterations) ---")
    stft_mag = mel_to_stft_magnitude(mel_linear, filterbank).numpy()
    audio_gl = griffin_lim(stft_mag, n_fft=n_fft, hop_length=hop_length, n_iter=100)
    audio_gl = audio_gl / (np.max(np.abs(audio_gl)) + 1e-8) * 0.9
    sf.write('klang/griffinlim_v2_original.wav', audio_gl[:len(y_orig)], sr)
    print(f"  Saved: klang/griffinlim_v2_original.wav")

    # ─── Step 3: Gaussian reconstructions ───
    for n_g in [500, 1000]:
        print(f"\n{'='*60}")
        print(f"  {n_g} Gaussians")
        print(f"{'='*60}")

        scene = AudioGaussianScene(n_g, T, F)
        optimizer = torch.optim.Adam(scene.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.0002)

        t_grid = torch.arange(T, dtype=torch.float32)
        f_grid = torch.arange(F, dtype=torch.float32)

        print("  Fitting...")
        for step in range(3000):
            rendered = scene.render(t_grid, f_grid)
            loss = torch.nn.functional.l1_loss(rendered, target_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 1000 == 0:
                print(f"    Step {step}: L1={loss.item():.4f}")
        print(f"    Final L1: {loss.item():.4f}")

        with torch.no_grad():
            rendered_norm = scene.render(t_grid, f_grid)  # [T, F]

        # Denormalize to log mel scale
        rendered_log_mel = rendered_norm * (mel_max - mel_min) + mel_min  # [T, F]
        rendered_log_mel_t = rendered_log_mel.T  # [F, T] = [n_mels, T]

        # --- HiFi-GAN ---
        if use_hifigan:
            print("  HiFi-GAN vocoding...")
            mel_input = rendered_log_mel_t.unsqueeze(0)  # [1, 80, T]
            with torch.no_grad():
                waveform = hifi.decode_batch(mel_input)
            audio = waveform.squeeze().numpy()
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
            sf.write(f'klang/hifigan_v2_{n_g}g.wav', audio[:len(y_orig)], sr)
            print(f"  Saved: klang/hifigan_v2_{n_g}g.wav")

        # --- Griffin-Lim ---
        print("  Griffin-Lim vocoding (100 iter)...")
        rendered_mel_linear = torch.exp(rendered_log_mel_t)  # back to linear
        stft_mag = mel_to_stft_magnitude(rendered_mel_linear, filterbank).numpy()
        audio_gl = griffin_lim(stft_mag, n_fft=n_fft, hop_length=hop_length, n_iter=100)
        audio_gl = audio_gl / (np.max(np.abs(audio_gl)) + 1e-8) * 0.9
        sf.write(f'klang/griffinlim_v2_{n_g}g.wav', audio_gl[:len(y_orig)], sr)
        print(f"  Saved: klang/griffinlim_v2_{n_g}g.wav")

    print(f"\n{'='*60}")
    print("DONE — Compare these files:")
    print(f"{'='*60}")
    print("\n  Original:                    klang/original.wav")
    print("  HiFi-GAN from original:     klang/hifigan_v2_original.wav")
    print("  Griffin-Lim from original:   klang/griffinlim_v2_original.wav")
    print("  HiFi-GAN 500 Gaussians:     klang/hifigan_v2_500g.wav")
    print("  HiFi-GAN 1000 Gaussians:    klang/hifigan_v2_1000g.wav")
    print("  Griffin-Lim 500 Gaussians:   klang/griffinlim_v2_500g.wav")
    print("  Griffin-Lim 1000 Gaussians:  klang/griffinlim_v2_1000g.wav")


if __name__ == '__main__':
    main()
