"""
Write and read standard 3DGS .ply files (binary_little_endian).

Format matches the original Kerbl et al. 3D Gaussian Splatting implementation
so that exported files can be loaded in XV3DGS (UE5), UnityGaussianSplatting,
gsplat viewers, and other standard tools.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from src.export.utils import sh_dc_from_rgb, normalize_quaternion


def write_ply(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    sh_degree: int = 0,
) -> None:
    """
    Write a 3DGS-compatible .ply file.

    Args:
        tensors: dict with keys:
            means: [N, 3] positions
            scales_log: [N, 3] log-scales
            rotations: [N, 4] quaternions (w, x, y, z)
            opacities: [N] logit opacities
            colors: [N, 3] RGB in [0, 1]
        path: output file path
        sh_degree: SH band count (0 = DC only from colors)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means = tensors["means"].detach().cpu().numpy().astype(np.float32)
    scales = tensors["scales_log"].detach().cpu().numpy().astype(np.float32)
    rotations = tensors["rotations"].detach().cpu().numpy().astype(np.float32)
    opacities = tensors["opacities"].detach().cpu().numpy().astype(np.float32)
    colors = tensors["colors"].detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    if n == 0:
        path.write_bytes(b"")
        return

    # Normalize quaternions
    rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rot_norms = np.maximum(rot_norms, 1e-10)
    rotations = rotations / rot_norms

    # Convert RGB to SH DC coefficients
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0  # [N, 3]

    # Number of SH rest coefficients (degree 0 has none extra)
    n_sh_rest = (sh_degree + 1) ** 2 * 3 - 3
    sh_rest = np.zeros((n, max(n_sh_rest, 0)), dtype=np.float32)

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]

    for i in range(n_sh_rest):
        header_lines.append(f"property float f_rest_{i}")

    header_lines.extend([
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ])

    header = "\n".join(header_lines) + "\n"
    header_bytes = header.encode("ascii")

    # Write binary data
    normals = np.zeros((n, 3), dtype=np.float32)

    with open(path, "wb") as f:
        f.write(header_bytes)
        for i in range(n):
            f.write(struct.pack("<3f", *means[i]))
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *f_dc[i]))
            if n_sh_rest > 0:
                f.write(struct.pack(f"<{n_sh_rest}f", *sh_rest[i]))
            f.write(struct.pack("<f", opacities[i]))
            f.write(struct.pack("<3f", *scales[i]))
            f.write(struct.pack("<4f", *rotations[i]))


def read_ply(path: str | Path) -> dict[str, torch.Tensor]:
    """
    Read a 3DGS .ply file back into tensors.

    Returns dict with: means, scales_log, rotations, opacities, colors
    """
    path = Path(path)
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                properties.append(line.split()[-1])

        if n_vertices == 0:
            return {
                "means": torch.zeros(0, 3),
                "scales_log": torch.zeros(0, 3),
                "rotations": torch.zeros(0, 4),
                "opacities": torch.zeros(0),
                "colors": torch.zeros(0, 3),
            }

        n_floats = len(properties)
        row_bytes = n_floats * 4
        data = np.frombuffer(f.read(n_vertices * row_bytes), dtype=np.float32)
        data = data.reshape(n_vertices, n_floats)

    # Map property names to column indices
    prop_idx = {name: i for i, name in enumerate(properties)}

    means = torch.tensor(data[:, [prop_idx["x"], prop_idx["y"], prop_idx["z"]]])

    # SH DC -> RGB
    SH_C0 = 0.28209479177387814
    f_dc = data[:, [prop_idx["f_dc_0"], prop_idx["f_dc_1"], prop_idx["f_dc_2"]]]
    colors = torch.tensor(np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0))

    opacities = torch.tensor(data[:, prop_idx["opacity"]])

    scales_log = torch.tensor(
        data[:, [prop_idx["scale_0"], prop_idx["scale_1"], prop_idx["scale_2"]]]
    )

    rotations = torch.tensor(
        data[:, [prop_idx["rot_0"], prop_idx["rot_1"], prop_idx["rot_2"], prop_idx["rot_3"]]]
    )

    return {
        "means": means,
        "scales_log": scales_log,
        "rotations": rotations,
        "opacities": opacities,
        "colors": colors,
    }
