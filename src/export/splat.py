"""
Write compressed .splat format for web delivery.

The .splat format stores 32 bytes per Gaussian:
  - position: float32 x 3 (12 bytes)
  - scale: float32 x 3 (12 bytes, exp of log-scale)
  - color+opacity: uint8 x 4 (4 bytes, RGBA)
  - rotation: uint8 x 4 (4 bytes, compressed quaternion)

This format is used by web viewers like antimatter15/splat and
GaussianSplats3D. It drops SH (view-dependent color) in favor of
a single baked RGB per Gaussian.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch


def write_splat(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    """
    Write a compressed .splat file.

    Args:
        tensors: dict with keys:
            means: [N, 3] positions
            scales_log: [N, 3] log-scales
            rotations: [N, 4] quaternions (w, x, y, z)
            opacities: [N] logit opacities (pre-sigmoid)
            colors: [N, 3] RGB in [0, 1]
        path: output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means = tensors["means"].detach().cpu().numpy().astype(np.float32)
    scales_log = tensors["scales_log"].detach().cpu().numpy().astype(np.float32)
    rotations = tensors["rotations"].detach().cpu().numpy().astype(np.float32)
    opacities = tensors["opacities"].detach().cpu().numpy().astype(np.float32)
    colors = tensors["colors"].detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    if n == 0:
        path.write_bytes(b"")
        return

    # Convert log-scale to actual scale
    scales = np.exp(scales_log)

    # Sigmoid on opacity logit -> [0, 1] -> uint8
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities))
    opacity_u8 = (opacity_sigmoid * 255).clip(0, 255).astype(np.uint8)

    # RGB [0,1] -> uint8
    colors_u8 = (colors * 255).clip(0, 255).astype(np.uint8)

    # Normalize quaternions
    rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rot_norms = np.maximum(rot_norms, 1e-10)
    rotations = rotations / rot_norms

    # Quaternion [w,x,y,z] in [-1,1] -> uint8 [0,255]
    # Map: -1 -> 0, 0 -> 128, 1 -> 255
    rot_u8 = ((rotations * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    with open(path, "wb") as f:
        for i in range(n):
            # Position (12 bytes)
            f.write(struct.pack("<3f", *means[i]))
            # Scale (12 bytes)
            f.write(struct.pack("<3f", *scales[i]))
            # RGBA (4 bytes)
            f.write(struct.pack("<4B", colors_u8[i, 0], colors_u8[i, 1],
                                colors_u8[i, 2], opacity_u8[i]))
            # Rotation quaternion (4 bytes)
            f.write(struct.pack("<4B", rot_u8[i, 0], rot_u8[i, 1],
                                rot_u8[i, 2], rot_u8[i, 3]))
