"""Klang 1.2 — SGS for audio.

Modules
-------
scene:      ComplexAudioLayerScene — complex-valued layers with mel-scaled
            centers, widened bounds, and alpha compositing (transmittance).
losses:     MRSTFT loss (multi-resolution), perceptual stubs.
mel_bridge: 80-bin mel path + HiFi-GAN decode hook.
"""

from .scene import ComplexAudioLayerScene, SceneConfig
from .losses import MRSTFTLoss, PerceptualLoss
from .mel_bridge import MelBridge

__all__ = [
    "ComplexAudioLayerScene",
    "SceneConfig",
    "MRSTFTLoss",
    "PerceptualLoss",
    "MelBridge",
]
