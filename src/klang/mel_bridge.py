"""Klang 1.2 mel-intermediate + HiFi-GAN decode bridge (#5).

The HiFi-GAN vocoder runs on mel spectrograms. Klang's native output is
STFT; this bridge converts STFT → mel → waveform with a HiFi-GAN model
so we can A/B decode quality against Griffin-Lim.

Importantly: a BAD decode is not necessarily a BAD splat. diag_4
(mel_griffinlim, no splat) is the ceiling of what any mel-GL decode can
sound like; if splat-mel-GL is close to diag_4 then the splat is fine
and vocoder choice is the next lever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MelBridge:
    """STFT ↔ mel ↔ waveform helper. All numpy I/O for plug-in use."""

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        import librosa
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr / 2
        self._mel_fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=self.f_max,
        )  # [n_mels, n_freqs]

    def stft_to_mel(self, mag: np.ndarray) -> np.ndarray:
        """mag: [F, T] → mel: [n_mels, T]."""
        return self._mel_fb @ mag

    def griffinlim_from_mel(self, mel: np.ndarray, n_iter: int = 200,
                            orig_length: Optional[int] = None) -> np.ndarray:
        import librosa
        mag = librosa.feature.inverse.mel_to_stft(
            mel, sr=self.sr, n_fft=self.n_fft, power=1.0, fmin=self.f_min,
            fmax=self.f_max,
        )
        y = librosa.griffinlim(
            mag, n_iter=n_iter, hop_length=self.hop_length, n_fft=self.n_fft,
            length=orig_length,
        )
        return y / (np.max(np.abs(y)) + 1e-8) * 0.9

    def hifigan_decode(self, mel: np.ndarray, hifigan_model) -> np.ndarray:
        """Decode with a HiFi-GAN model (PyTorch). Expects a module whose
        `.forward(mel_tensor)` returns waveform. mel is log-mel by
        convention; convert if needed."""
        with torch.no_grad():
            x = torch.from_numpy(np.log(mel + 1e-9)).float().unsqueeze(0)
            x = x.to(next(hifigan_model.parameters()).device)
            y = hifigan_model(x).squeeze().cpu().numpy()
        return y / (np.max(np.abs(y)) + 1e-8) * 0.9


def load_hifigan(weights_path: str, config_path: Optional[str] = None,
                 device: str = "cuda") -> nn.Module:
    """Thin loader. HiFi-GAN implementations vary — this assumes the
    jik876/hifi-gan layout (Generator + config). Adjust on Windows if
    the actual weights came from a different repo."""
    import json
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"HiFi-GAN weights not found: {p}")
    if config_path is None:
        cfg_candidates = [p.parent / "config.json", p.with_suffix(".json")]
        config_path = next((str(c) for c in cfg_candidates if c.exists()), None)
    if config_path is None:
        raise FileNotFoundError("Pass --hifigan-config; no config.json beside weights.")
    with open(config_path) as f:
        cfg = json.load(f)

    # jik876/hifi-gan Generator lives in their models.py. We don't vendor
    # it; rely on user to have it on sys.path. This matches the existing
    # klang/reconstruct_hifigan_v2.py approach.
    try:
        from hifi_gan.models import Generator
    except ImportError as e:
        raise ImportError(
            "HiFi-GAN code not on sys.path. Add the hifi-gan repo or adjust "
            "MelBridge.hifigan_decode to call your vendored module directly."
        ) from e

    from types import SimpleNamespace
    ns = SimpleNamespace(**cfg)
    gen = Generator(ns).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    gen.load_state_dict(state["generator"] if "generator" in state else state)
    gen.eval()
    gen.remove_weight_norm() if hasattr(gen, "remove_weight_norm") else None
    return gen
