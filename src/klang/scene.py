"""Klang 1.2 — Complex-valued layered audio scene.

Changes vs Variant B:
  (#1) Complex amplitude A·e^(iφ) per layer-frame — fits phase directly,
       so Griffin-Lim / HiFi-GAN no longer have to hallucinate it.
  (#2) Mel-scaled center init + widened σ/f₀ bounds: covers sub-200 Hz.
  (#4) Transmittance-budgeted alpha compositing: sort layers by salience,
       render with T_max cap (same theorem as Planck/Hertz).

Back-compat: set `complex=False, compositing="sum"` to recover the
Variant B additive-magnitude model for ablation.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SceneConfig:
    n_layers: int = 40
    n_freqs: int = 513           # n_fft // 2 + 1 for n_fft=1024
    n_frames: int = 256
    sr: int = 22050
    n_fft: int = 1024
    n_harmonics: int = 8
    ctrl_stride: int = 4

    # #2 widened bounds — defaults tuned for 22050 Hz STFT with n_fft=1024.
    # bin_i ↔ hz_i = i * sr / n_fft; bin 2 ≈ 43 Hz, bin 25 ≈ 537 Hz.
    f_min_hz: float = 40.0       # was effectively ~540 Hz in Variant B
    f_max_hz_frac: float = 0.95  # fraction of Nyquist
    sigma_f_min_bins: float = 0.5   # was ~0.6 effective
    sigma_f_max_bins: float = 60.0  # cap to kill near-Nyquist whine
    init_sigma_log: float = 1.2     # σ ≈ 3.3 bins initial

    # #1 complex mode
    complex: bool = True
    init_phase_std: float = 0.1

    # #4 compositing
    compositing: Literal["sum", "alpha"] = "alpha"
    t_max: float = 0.9           # transmittance budget (1.0 == no cap)


def _mel_hz(m: torch.Tensor) -> torch.Tensor:
    """Mel → Hz (HTK)."""
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _hz_mel(hz: torch.Tensor) -> torch.Tensor:
    """Hz → Mel (HTK)."""
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_spaced_bins(n_layers: int, f_min_hz: float, f_max_hz: float,
                    sr: int, n_fft: int) -> torch.Tensor:
    """Return `n_layers` center frequencies in STFT-bin units, mel-spaced."""
    m_min = _hz_mel(torch.tensor(f_min_hz))
    m_max = _hz_mel(torch.tensor(f_max_hz))
    mels = torch.linspace(m_min.item(), m_max.item(), n_layers)
    hz = _mel_hz(mels)
    return hz * n_fft / sr  # bins


class ComplexAudioLayerScene(nn.Module):
    """K audio layers, each a Gaussian in frequency with complex amplitude.

    Output: complex STFT of shape [T, F] (if complex=True) or magnitude
    STFT of shape [T, F] (if complex=False). Both render through the
    same Gaussian machinery so ablations swap only the final step.
    """

    def __init__(self, cfg: SceneConfig):
        super().__init__()
        self.cfg = cfg
        K = cfg.n_layers
        T = cfg.n_frames
        n_ctrl = T // cfg.ctrl_stride + 1
        self.n_ctrl = n_ctrl

        f_max_hz = cfg.f_max_hz_frac * cfg.sr * 0.5
        init_bins = mel_spaced_bins(K, cfg.f_min_hz, f_max_hz, cfg.sr, cfg.n_fft)
        self.mu_f = nn.Parameter(init_bins)

        # σ stored in log-space; bounds enforced via clamp at render time.
        self.log_sigma_f = nn.Parameter(torch.full((K,), cfg.init_sigma_log))

        # Frequency trajectory offsets (control points, linearly interpolated)
        self.path_ctrl = nn.Parameter(torch.zeros(K, n_ctrl))

        # Pre-sigmoid opacity control points. Small positive init so all
        # layers start quietly audible and the optimizer has gradient.
        self.alpha_ctrl = nn.Parameter(torch.full((K, n_ctrl), 0.0))

        # Harmonic amplitude logits (softmax-normalized at render time)
        self.harmonic_logits = nn.Parameter(torch.zeros(K, cfg.n_harmonics))
        with torch.no_grad():
            self.harmonic_logits[:, 0] = 2.0  # fundamental dominant

        # #1 complex: per layer per control-point phase offset (radians).
        # Interpolated to per-frame phase along with path.
        if cfg.complex:
            self.phase_ctrl = nn.Parameter(
                torch.randn(K, n_ctrl) * cfg.init_phase_std
            )
        else:
            self.register_parameter("phase_ctrl", None)

    # ── helpers ──────────────────────────────────────────────────────────
    def _interp(self, ctrl: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Linearly interpolate [K, n_ctrl] → [K, n_frames]."""
        x = ctrl.unsqueeze(1)  # [K, 1, n_ctrl]
        x = F.interpolate(x, size=n_frames, mode="linear", align_corners=True)
        return x.squeeze(1)

    def _sigma_f(self) -> torch.Tensor:
        s = torch.exp(self.log_sigma_f)
        return s.clamp(self.cfg.sigma_f_min_bins, self.cfg.sigma_f_max_bins)

    def _center_bins(self, path: torch.Tensor) -> torch.Tensor:
        """Per-frame center frequency (bins), clamped to [f_min_bin, Nyquist]."""
        f_min_bin = self.cfg.f_min_hz * self.cfg.n_fft / self.cfg.sr
        f_max_bin = self.cfg.n_freqs - 1
        return (self.mu_f.unsqueeze(1) + path).clamp(f_min_bin, f_max_bin)

    # ── forward ──────────────────────────────────────────────────────────
    def render_magnitude(self) -> torch.Tensor:
        """Returns [T, F] real magnitude; no phase."""
        K, T, F_ = self.cfg.n_layers, self.cfg.n_frames, self.cfg.n_freqs

        path = self._interp(self.path_ctrl, T)                   # [K, T]
        alpha = torch.sigmoid(self._interp(self.alpha_ctrl, T))  # [K, T]
        sigma = self._sigma_f()                                  # [K]
        freq_t = self._center_bins(path)                         # [K, T]

        f_grid = torch.arange(F_, device=self.mu_f.device, dtype=torch.float32)
        f_exp = f_grid.view(1, 1, F_)
        freq_exp = freq_t.unsqueeze(2)                           # [K, T, 1]
        sigma_exp = sigma.view(K, 1, 1)

        # Fundamental
        K_fund = torch.exp(-0.5 * ((f_exp - freq_exp) / sigma_exp) ** 2)

        # Harmonics (k·f₀), narrower bandwidth per harmonic
        harm = torch.softmax(self.harmonic_logits, dim=1)        # [K, H]
        mag = K_fund * harm[:, 0].view(K, 1, 1)
        for h in range(1, harm.shape[1]):
            harm_freq = freq_exp * (h + 1)
            K_h = torch.exp(-0.5 * ((f_exp - harm_freq) / (sigma_exp * 0.7)) ** 2)
            mag = mag + K_h * harm[:, h].view(K, 1, 1)

        # Compositing
        if self.cfg.compositing == "sum":
            return (alpha.unsqueeze(2) * mag).sum(dim=0)  # [T, F]
        return self._alpha_composite(alpha, mag)

    def render_complex(self) -> torch.Tensor:
        """Returns [T, F] complex STFT. Raises if `complex=False`."""
        if not self.cfg.complex:
            raise RuntimeError(
                "Scene was built with complex=False; call render_magnitude()."
            )

        K, T, F_ = self.cfg.n_layers, self.cfg.n_frames, self.cfg.n_freqs

        path = self._interp(self.path_ctrl, T)                   # [K, T]
        alpha = torch.sigmoid(self._interp(self.alpha_ctrl, T))  # [K, T]
        phase = self._interp(self.phase_ctrl, T)                 # [K, T]
        sigma = self._sigma_f()
        freq_t = self._center_bins(path)

        f_grid = torch.arange(F_, device=self.mu_f.device, dtype=torch.float32)
        f_exp = f_grid.view(1, 1, F_)
        freq_exp = freq_t.unsqueeze(2)
        sigma_exp = sigma.view(K, 1, 1)

        # Gaussian magnitude kernel (as before)
        K_fund = torch.exp(-0.5 * ((f_exp - freq_exp) / sigma_exp) ** 2)
        harm = torch.softmax(self.harmonic_logits, dim=1)
        mag = K_fund * harm[:, 0].view(K, 1, 1)
        for h in range(1, harm.shape[1]):
            harm_freq = freq_exp * (h + 1)
            K_h = torch.exp(-0.5 * ((f_exp - harm_freq) / (sigma_exp * 0.7)) ** 2)
            mag = mag + K_h * harm[:, h].view(K, 1, 1)

        # Complex layer = magnitude * e^(iφ_t)
        phase_exp = phase.unsqueeze(2)                           # [K, T, 1]
        cos_p = torch.cos(phase_exp)
        sin_p = torch.sin(phase_exp)
        # Broadcast over freq bins: shared phase across a narrow Gaussian
        # is a reasonable first-order model (phase varies slowly within one
        # spectral atom). Per-bin phase would require K·T·F params.
        real = mag * cos_p
        imag = mag * sin_p

        if self.cfg.compositing == "sum":
            real_s = (alpha.unsqueeze(2) * real).sum(dim=0)
            imag_s = (alpha.unsqueeze(2) * imag).sum(dim=0)
        else:
            real_s, imag_s = self._alpha_composite_complex(alpha, real, imag)

        return torch.complex(real_s, imag_s)

    # ── compositing ──────────────────────────────────────────────────────
    def _sort_by_salience(self, alpha: torch.Tensor, mag: torch.Tensor):
        """Order layers by total energy so small-contribution atoms are
        rendered last (beneath the transmittance cap)."""
        salience = (alpha * mag.sum(dim=-1)).sum(dim=-1)         # [K]
        _, order = salience.sort(descending=True)
        return order

    def _alpha_composite(self, alpha: torch.Tensor, mag: torch.Tensor
                         ) -> torch.Tensor:
        """Alpha compositing with transmittance budget. [T, F] out."""
        order = self._sort_by_salience(alpha, mag)
        a_eff = (alpha.unsqueeze(2) * mag).clamp(0, 1)[order]    # [K, T, F]
        out = torch.zeros_like(a_eff[0])
        T_trans = torch.ones_like(a_eff[0])
        for k in range(a_eff.shape[0]):
            out = out + T_trans * a_eff[k]
            T_trans = T_trans * (1 - a_eff[k])
            if self.cfg.t_max < 1.0:
                T_trans = T_trans.clamp(min=1.0 - self.cfg.t_max)
        return out

    def _alpha_composite_complex(self, alpha, real, imag):
        """Same compositing on real/imag channels (alpha drives both)."""
        # Use magnitude for salience + alpha envelope for transmittance.
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        order = self._sort_by_salience(alpha, mag)
        a_eff = (alpha.unsqueeze(2) * mag).clamp(0, 1)[order]
        real_o, imag_o = real[order], imag[order]
        out_r = torch.zeros_like(a_eff[0])
        out_i = torch.zeros_like(a_eff[0])
        T_trans = torch.ones_like(a_eff[0])
        for k in range(a_eff.shape[0]):
            # Weight complex contribution by transmittance but by alpha_k
            # envelope, not the magnitude (magnitude already in real/imag).
            env = alpha[order[k]].unsqueeze(1)                   # [T, 1]
            out_r = out_r + T_trans * env * real_o[k]
            out_i = out_i + T_trans * env * imag_o[k]
            T_trans = T_trans * (1 - a_eff[k])
            if self.cfg.t_max < 1.0:
                T_trans = T_trans.clamp(min=1.0 - self.cfg.t_max)
        return out_r, out_i

    # ── diagnostics ──────────────────────────────────────────────────────
    @torch.no_grad()
    def layer_summary(self) -> list[dict]:
        path = self._interp(self.path_ctrl, self.cfg.n_frames)
        alpha = torch.sigmoid(self._interp(self.alpha_ctrl, self.cfg.n_frames))
        sigma = self._sigma_f()
        out = []
        for i in range(self.cfg.n_layers):
            freq_traj = (self.mu_f[i] + path[i]).cpu().numpy()
            a = alpha[i].cpu().numpy()
            out.append({
                "mu_f_bin": float(self.mu_f[i].item()),
                "mu_f_hz": float(self.mu_f[i].item() * self.cfg.sr / self.cfg.n_fft),
                "sigma_f_bin": float(sigma[i].item()),
                "alpha_mean": float(a.mean()),
                "freq_traj": freq_traj,
                "alpha_traj": a,
                "harmonics": torch.softmax(self.harmonic_logits[i], dim=0).cpu().numpy(),
            })
        return out
