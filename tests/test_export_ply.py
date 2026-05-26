"""Tests for src/export/ply.py"""

import tempfile
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.export.ply import write_ply, read_ply


def _make_tensors(n: int = 50) -> dict[str, torch.Tensor]:
    return {
        "means": torch.randn(n, 3),
        "scales_log": torch.randn(n, 3) - 2,
        "rotations": torch.randn(n, 4),
        "opacities": torch.randn(n),
        "colors": torch.rand(n, 3),
    }


def test_write_creates_file():
    tensors = _make_tensors(10)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
    Path(path).unlink()


def test_ply_header_format():
    tensors = _make_tensors(5)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path)

    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

    header_str = header.decode("ascii")
    assert "ply" in header_str
    assert "binary_little_endian" in header_str
    assert "element vertex 5" in header_str
    assert "property float x" in header_str
    assert "property float rot_3" in header_str
    Path(path).unlink()


def test_ply_byte_count():
    n = 20
    tensors = _make_tensors(n)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path, sh_degree=0)

    # SH degree 0: no f_rest properties
    # Properties: x,y,z,nx,ny,nz,f_dc_0,f_dc_1,f_dc_2,opacity,scale_0,scale_1,scale_2,rot_0,rot_1,rot_2,rot_3 = 17 floats
    n_floats_per_vertex = 17
    expected_data_bytes = n * n_floats_per_vertex * 4

    file_size = Path(path).stat().st_size
    # Header size varies, but data portion should be exact
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if b"end_header" in line:
                break
        data_bytes = f.read()

    assert len(data_bytes) == expected_data_bytes
    Path(path).unlink()


def test_ply_round_trip_positions():
    tensors = _make_tensors(30)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path)
    loaded = read_ply(path)

    assert loaded["means"].shape == tensors["means"].shape
    diff = (loaded["means"] - tensors["means"]).abs().max().item()
    assert diff < 1e-5, f"Position diff too large: {diff}"
    Path(path).unlink()


def test_ply_round_trip_rotations():
    tensors = _make_tensors(30)
    # Normalize rotations before writing (write_ply normalizes internally)
    norms = tensors["rotations"].norm(dim=1, keepdim=True)
    tensors["rotations"] = tensors["rotations"] / norms

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path)
    loaded = read_ply(path)

    diff = (loaded["rotations"] - tensors["rotations"]).abs().max().item()
    assert diff < 1e-5, f"Rotation diff too large: {diff}"
    Path(path).unlink()


def test_ply_round_trip_colors():
    tensors = _make_tensors(30)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path)
    loaded = read_ply(path)

    # Colors go through SH DC conversion, should round-trip with minimal loss
    diff = (loaded["colors"] - tensors["colors"]).abs().max().item()
    assert diff < 0.001, f"Color diff too large: {diff}"
    Path(path).unlink()


def test_ply_empty_scene():
    tensors = {
        "means": torch.zeros(0, 3),
        "scales_log": torch.zeros(0, 3),
        "rotations": torch.zeros(0, 4),
        "opacities": torch.zeros(0),
        "colors": torch.zeros(0, 3),
    }
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    write_ply(tensors, path)
    # Empty file for empty scene
    assert Path(path).stat().st_size == 0
    Path(path).unlink()
