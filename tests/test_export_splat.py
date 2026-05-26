"""Tests for src/export/splat.py"""

import tempfile
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.splat import write_splat


def _make_tensors(n: int = 50) -> dict[str, torch.Tensor]:
    return {
        "means": torch.randn(n, 3),
        "scales_log": torch.randn(n, 3) - 2,
        "rotations": torch.randn(n, 4),
        "opacities": torch.randn(n),
        "colors": torch.rand(n, 3),
    }


def test_splat_file_size():
    n = 100
    tensors = _make_tensors(n)
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name
    write_splat(tensors, path)
    # 32 bytes per Gaussian
    assert Path(path).stat().st_size == n * 32
    Path(path).unlink()


def test_splat_empty():
    tensors = {
        "means": torch.zeros(0, 3),
        "scales_log": torch.zeros(0, 3),
        "rotations": torch.zeros(0, 4),
        "opacities": torch.zeros(0),
        "colors": torch.zeros(0, 3),
    }
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name
    write_splat(tensors, path)
    assert Path(path).stat().st_size == 0
    Path(path).unlink()


def test_splat_byte_layout():
    """Verify the byte layout: 3f pos + 3f scale + 4B rgba + 4B quat = 32."""
    tensors = _make_tensors(1)
    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name
    write_splat(tensors, path)

    import struct
    with open(path, "rb") as f:
        data = f.read()

    assert len(data) == 32
    # First 12 bytes: position (3 floats)
    pos = struct.unpack("<3f", data[0:12])
    assert len(pos) == 3
    # Next 12 bytes: scale (3 floats)
    scale = struct.unpack("<3f", data[12:24])
    assert len(scale) == 3
    # Next 4 bytes: RGBA
    rgba = struct.unpack("<4B", data[24:28])
    assert all(0 <= v <= 255 for v in rgba)
    # Last 4 bytes: quaternion
    quat = struct.unpack("<4B", data[28:32])
    assert all(0 <= v <= 255 for v in quat)
    Path(path).unlink()
