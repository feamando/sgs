"""Klang 1.2 losses.

MRSTFTLoss (#3):
    Multi-resolution STFT loss — sum of log-magnitude L1 and spectral-
    convergence terms across several window sizes. Standard in neural
    vocoders; handles phase-warble by not over-fitting to any single
    window's magnitude.

PerceptualLoss (#6):
    Wrap a pretrained embedding (VGGish / CLAP) as a feature-space loss.
    Stub by default — returns 0 if no backbone is provided. Wire in
    whichever is available on Windows at experiment time.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_mag(y: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=y.device)
    spec = torch.stft(
        y, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True, center=True,
    )
    return spec.abs()


class MRSTFTLoss(nn.Module):
    """Multi-resolution STFT loss in the waveform domain.

    L = Σ_r [ SC_r + α · logmag_r ]
    where SC_r = ||mag - mag_hat||_F / ||mag||_F  (spectral convergence)
          logmag_r = L1(log(mag + eps), log(mag_hat + eps)).
    """

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Optional[Sequence[int]] = None,
        win_sizes: Optional[Sequence[int]] = None,
        log_mag_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes or [n // 4 for n in fft_sizes])
        self.win_sizes = tuple(win_sizes or fft_sizes)
        self.log_mag_weight = log_mag_weight
        self.eps = eps

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = y_hat.new_zeros(())
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            mag = _stft_mag(y, n_fft, hop, win)
            mag_hat = _stft_mag(y_hat, n_fft, hop, win)
            sc = torch.linalg.norm(mag - mag_hat, "fro") / (
                torch.linalg.norm(mag, "fro") + self.eps
            )
            log_l1 = F.l1_loss(
                torch.log(mag_hat + self.eps), torch.log(mag + self.eps)
            )
            total = total + sc + self.log_mag_weight * log_l1
        return total / len(self.fft_sizes)


class PerceptualLoss(nn.Module):
    """Optional feature-space loss. No-op unless an embedding model is
    provided. Kept pluggable because VGGish/CLAP availability varies
    across platforms."""

    def __init__(self, embedder: Optional[nn.Module] = None,
                 sample_rate: int = 22050, weight: float = 1.0):
        super().__init__()
        self.embedder = embedder
        self.sample_rate = sample_rate
        self.weight = weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.embedder is None:
            return y_hat.new_zeros(())
        f_hat = self.embedder(y_hat)
        f = self.embedder(y)
        return self.weight * F.l1_loss(f_hat, f)
