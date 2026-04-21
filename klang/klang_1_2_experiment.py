#!/usr/bin/env python3
"""
Klang 1.2 experiment driver.

Builds on Variant B with:
  #1 Complex-valued Gaussians (phase is fitted, not hallucinated)
  #2 Mel-scaled init + widened σ/f₀ bounds (fixes sub-200 Hz dropout)
  #3 Multi-resolution STFT loss (time-domain — kills single-window warble)
  #4 Transmittance-budgeted alpha compositing (SGS theorem)
  #5 Mel + HiFi-GAN decode bridge  (A/B against Griffin-Lim)
  #6 Optional perceptual loss (VGGish/CLAP)
  #7 Companion `scripts/validate_klang.py` emits quantitative metrics

Usage (one-shot):
    python klang/klang_1_2_experiment.py --audio klang/test_clip.wav \\
        --n-layers 20 --n-steps 3000 --device cuda

Ablation flags:
    --no-complex            disable #1 (back to magnitude-only)
    --compositing sum       disable #4 (additive sum like Variant B)
    --no-mrstft             disable #3 (train on STFT L1 only)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.klang import ComplexAudioLayerScene, SceneConfig, MRSTFTLoss, MelBridge

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Install: pip install librosa soundfile")
    raise


def _stft_to_waveform(stft_complex: torch.Tensor, n_fft: int, hop_length: int,
                       win_length: int, length: int) -> torch.Tensor:
    """iSTFT of a complex [T, F] tensor (note: torch expects [F, T])."""
    s = stft_complex.transpose(0, 1).unsqueeze(0)  # [1, F, T]
    window = torch.hann_window(win_length, device=stft_complex.device)
    y = torch.istft(
        s, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, length=length, center=True,
    )
    return y.squeeze(0)


def _denormalize_complex(S_hat_norm: torch.Tensor, s_min: float, s_max: float,
                          eps: float = 1e-8) -> torch.Tensor:
    """Invert min-max normalization on magnitude while preserving phase.

    S_hat_norm has |S| ∈ [0, 1]. We want |S_denorm| ≈ |S|·(s_max-s_min) + s_min
    and arg(S_denorm) = arg(S_hat_norm). Compute phase as S / |S| and multiply
    by the denormalized magnitude.
    """
    mag_norm = S_hat_norm.abs()
    mag_denorm = mag_norm * (s_max - s_min) + s_min
    phase = S_hat_norm / (mag_norm + eps)  # complex unit
    return phase * mag_denorm.to(phase.dtype)


def fit_scene(target_complex: torch.Tensor, target_mag_norm: torch.Tensor,
              s_min: float, s_max: float, y_target: torch.Tensor,
              args, device) -> tuple[ComplexAudioLayerScene, list[float]]:
    """Run the fit loop. target_complex is the ground-truth STFT (complex,
    [T, F]); target_mag_norm is the min-max normalized magnitude for
    Variant B–style L1; y_target is the raw waveform for MRSTFT."""
    T, F_ = target_mag_norm.shape
    cfg = SceneConfig(
        n_layers=args.n_layers,
        n_freqs=F_,
        n_frames=T,
        sr=args.sr,
        n_fft=args.n_fft,
        n_harmonics=args.n_harmonics,
        ctrl_stride=args.ctrl_stride,
        complex=not args.no_complex,
        compositing=args.compositing,
        t_max=args.t_max,
    )
    scene = ComplexAudioLayerScene(cfg).to(device)

    optimizer = torch.optim.Adam(scene.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=args.lr * 0.01,
    )

    mrstft = MRSTFTLoss(
        fft_sizes=args.mrstft_ffts,
        log_mag_weight=args.mrstft_logmag_weight,
    ).to(device) if not args.no_mrstft else None

    losses = []
    t0 = time.time()
    for step in range(args.n_steps):
        # The scene's magnitude output is in [0, 1] by construction — train
        # it to match `target_mag_norm` directly. For the time-domain loss
        # we denormalize the magnitude while preserving the learned phase
        # before iSTFT, so the waveform has realistic energy.
        if cfg.complex:
            S_hat = scene.render_complex()                       # complex, |S|∈[0,1]
            mag_hat_norm = S_hat.abs()
            loss_l1 = TF.l1_loss(mag_hat_norm, target_mag_norm)
            if mrstft is not None:
                S_denorm = _denormalize_complex(S_hat, s_min, s_max)
                y_hat = _stft_to_waveform(
                    S_denorm, args.n_fft, args.hop_length, args.n_fft,
                    length=y_target.shape[-1],
                )
                loss_time = mrstft(y_hat, y_target)
            else:
                loss_time = torch.zeros((), device=device)
        else:
            mag_hat_norm = scene.render_magnitude()
            loss_l1 = TF.l1_loss(mag_hat_norm, target_mag_norm)
            loss_time = torch.zeros((), device=device)

        loss = args.l1_weight * loss_l1 + args.time_weight * loss_time

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if step % 200 == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t0
            sps = (step + 1) / max(elapsed, 1e-6)
            print(
                f"  Step {step:4d}/{args.n_steps} | "
                f"total={loss.item():.4f} L1={loss_l1.item():.4f} "
                f"time={loss_time.item():.4f} | "
                f"{sps:.1f} step/s | {elapsed:.0f}s",
                flush=True,
            )

    return scene, losses


def save_outputs(scene, losses, target_mag: np.ndarray, s_min: float,
                 s_max: float, orig_len: int, args, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save scene
    torch.save({
        "cfg": scene.cfg.__dict__,
        "state_dict": scene.state_dict(),
        "losses": losses,
    }, out_dir / "scene.pt")

    # Spectrograms
    with torch.no_grad():
        if scene.cfg.complex:
            S_hat = scene.render_complex()                       # |S|∈[0,1]
            S_denorm = _denormalize_complex(S_hat, s_min, s_max)
            mag_hat = S_denorm.abs().cpu().numpy()               # [T, F] real magnitudes
            y_hat = _stft_to_waveform(
                S_denorm, args.n_fft, args.hop_length, args.n_fft, orig_len,
            ).detach().cpu().numpy()
            y_hat = y_hat / (np.max(np.abs(y_hat)) + 1e-8) * 0.9
            sf.write(out_dir / "decode_istft.wav", y_hat, args.sr)
        else:
            mag_hat_norm = scene.render_magnitude().cpu().numpy()
            mag_hat = mag_hat_norm * (s_max - s_min) + s_min

    # Griffin-Lim decode from (denormalized) magnitude, always produced
    # as a baseline comparison.
    mag_denorm = np.maximum(mag_hat, 0).T                         # [F, T]
    y_gl = librosa.griffinlim(
        mag_denorm, n_iter=200, hop_length=args.hop_length,
        n_fft=args.n_fft, length=orig_len,
    )
    y_gl = y_gl / (np.max(np.abs(y_gl)) + 1e-8) * 0.9
    sf.write(out_dir / "decode_griffinlim.wav", y_gl, args.sr)

    # Mel + HiFi-GAN bridge if the model path was supplied
    if args.hifigan_weights:
        try:
            from src.klang.mel_bridge import load_hifigan
            bridge = MelBridge(sr=args.sr, n_fft=args.n_fft,
                               hop_length=args.hop_length, n_mels=args.n_mels)
            mel = bridge.stft_to_mel(mag_denorm)
            gen = load_hifigan(args.hifigan_weights, args.hifigan_config,
                               device=args.device)
            y_hifi = bridge.hifigan_decode(mel, gen)
            sf.write(out_dir / "decode_hifigan.wav", y_hifi, args.sr)
        except Exception as e:
            print(f"  HiFi-GAN decode skipped: {e}")

    # Plots
    _plot_reconstruction(target_mag, mag_hat.T, losses[-1], out_dir)
    _plot_loss(losses, out_dir)
    _plot_layers(scene, target_mag, out_dir)


def _plot_reconstruction(target: np.ndarray, rendered: np.ndarray,
                         final_loss: float, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].imshow(target, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Target STFT")
    axes[1].imshow(rendered, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title(f"Rendered (loss={final_loss:.4f})")
    diff = np.abs(target - rendered)
    axes[2].imshow(diff, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title(f"Error (mean={diff.mean():.4f})")
    for a in axes:
        a.set_ylabel("Freq bin")
    axes[-1].set_xlabel("Time frame")
    plt.tight_layout()
    plt.savefig(out_dir / "reconstruction.png", dpi=150)
    plt.close()


def _plot_loss(losses, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#F4A300")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total loss")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()


def _plot_layers(scene, target_mag: np.ndarray, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    info = scene.layer_summary()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(target_mag.T, aspect="auto", origin="lower", cmap="magma", alpha=0.5)
    colors = plt.cm.Set1(np.linspace(0, 1, len(info)))
    for i, layer in enumerate(info):
        if layer["alpha_mean"] > 0.05:
            t_axis = np.arange(len(layer["freq_traj"]))
            ax.plot(t_axis, layer["freq_traj"], color=colors[i], linewidth=1.5,
                    alpha=0.8,
                    label=f"L{i}: {layer['mu_f_hz']:.0f}Hz α={layer['alpha_mean']:.2f}")
    ax.set_ylim(0, scene.cfg.n_freqs)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Freq bin")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Klang 1.2 experiment")
    p.add_argument("--audio", default="klang/test_clip.wav")
    p.add_argument("--out-dir", default="klang/klang_1_2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # STFT
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)

    # Model
    p.add_argument("--n-layers", type=int, default=20)
    p.add_argument("--n-harmonics", type=int, default=8)
    p.add_argument("--ctrl-stride", type=int, default=4)
    p.add_argument("--compositing", choices=["sum", "alpha"], default="alpha")
    p.add_argument("--t-max", type=float, default=0.9)
    p.add_argument("--no-complex", action="store_true",
                   help="Disable complex-valued layers (ablation).")

    # Losses
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--time-weight", type=float, default=0.5)
    p.add_argument("--no-mrstft", action="store_true")
    p.add_argument("--mrstft-ffts", type=int, nargs="+",
                   default=[512, 1024, 2048])
    p.add_argument("--mrstft-logmag-weight", type=float, default=1.0)

    # Training
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.03)

    # HiFi-GAN (optional)
    p.add_argument("--hifigan-weights", default=None)
    p.add_argument("--hifigan-config", default=None)
    p.add_argument("--n-mels", type=int, default=80)

    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("KLANG 1.2 — Complex SGS + MRSTFT + α-compositing")
    print("=" * 60)

    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"\nLoading {args.audio}...")
    y, _ = librosa.load(args.audio, sr=args.sr)
    print(f"  {len(y)} samples, {len(y)/args.sr:.2f}s")

    S_complex = librosa.stft(y, n_fft=args.n_fft, hop_length=args.hop_length)
    S_mag = np.abs(S_complex)                                    # [F, T]
    s_min, s_max = float(S_mag.min()), float(S_mag.max())
    target_mag_norm = (S_mag - s_min) / max(s_max - s_min, 1e-8)
    target_mag_norm_T = torch.from_numpy(target_mag_norm.T).float().to(device)
    target_complex_T = torch.from_numpy(S_complex.T).to(device)   # [T, F]
    y_target = torch.from_numpy(y).float().to(device)

    scene, losses = fit_scene(
        target_complex_T, target_mag_norm_T, s_min, s_max, y_target, args, device,
    )

    out_dir = Path(args.out_dir)
    save_outputs(scene, losses, S_mag, s_min, s_max, len(y), args, out_dir)

    # Quick layer summary at the end
    info = scene.layer_summary()
    active = [l for l in info if l["alpha_mean"] > 0.05]
    print(f"\n  Active layers: {len(active)}/{len(info)}")
    for l in info[:5]:
        print(f"    μ={l['mu_f_hz']:7.1f}Hz σ={l['sigma_f_bin']:5.2f}bin "
              f"α̅={l['alpha_mean']:.3f}")
    print(f"\nOutputs in {out_dir}/")


if __name__ == "__main__":
    main()
