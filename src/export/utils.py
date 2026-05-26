"""Export utility functions: quaternion math, SH evaluation."""

from __future__ import annotations

import math

import torch


def euler_to_quaternion(rx: float, ry: float, rz: float) -> list[float]:
    """Convert Euler angles (radians) to quaternion [w, x, y, z]."""
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return [w, x, y, z]


def normalize_quaternion(q: list[float]) -> list[float]:
    """Normalize a quaternion to unit length."""
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / norm for v in q]


def sh_dc_from_rgb(r: float, g: float, b: float) -> list[float]:
    """
    Convert linear RGB [0,1] to SH degree-0 (DC) coefficients.

    The 3DGS convention: color = SH_C0 * f_dc + 0.5
    where SH_C0 = 0.28209479177387814
    So f_dc = (color - 0.5) / SH_C0
    """
    SH_C0 = 0.28209479177387814
    return [
        (r - 0.5) / SH_C0,
        (g - 0.5) / SH_C0,
        (b - 0.5) / SH_C0,
    ]


def rgb_from_sh_dc(f_dc: list[float]) -> list[float]:
    """Convert SH degree-0 coefficients back to linear RGB [0,1]."""
    SH_C0 = 0.28209479177387814
    return [
        max(0.0, min(1.0, SH_C0 * f_dc[0] + 0.5)),
        max(0.0, min(1.0, SH_C0 * f_dc[1] + 0.5)),
        max(0.0, min(1.0, SH_C0 * f_dc[2] + 0.5)),
    ]
