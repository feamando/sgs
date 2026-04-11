"""
3D object templates as Gaussian point clouds.

Each template is a set of (means, scales, opacities) representing a shape.
Color is applied at assembly time.
"""

import math
import torch
from dataclasses import dataclass


@dataclass
class GaussianTemplate:
    """A 3D object represented as a set of Gaussians."""
    means: torch.Tensor       # [N, 3] — positions
    scales: torch.Tensor      # [N, 3] — log-scales
    opacities: torch.Tensor   # [N]    — logit (pre-sigmoid)

    @property
    def n_gaussians(self) -> int:
        return self.means.shape[0]

    def to(self, device):
        return GaussianTemplate(
            means=self.means.to(device),
            scales=self.scales.to(device),
            opacities=self.opacities.to(device),
        )


def _fibonacci_sphere(n: int) -> torch.Tensor:
    """Uniformly distributed points on unit sphere via Fibonacci spiral."""
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    i = torch.arange(n, dtype=torch.float32)
    theta = 2.0 * math.pi * i / golden
    phi = torch.acos(1.0 - 2.0 * (i + 0.5) / n)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


def _uniform_disk(n: int) -> torch.Tensor:
    """Uniformly distributed points on unit disk (z=0)."""
    r = torch.sqrt(torch.rand(n))
    theta = 2.0 * math.pi * torch.rand(n)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = torch.zeros(n)
    return torch.stack([x, y, z], dim=-1)


def make_sphere(n: int = 200, radius: float = 0.5) -> GaussianTemplate:
    """Sphere: points on surface."""
    means = _fibonacci_sphere(n) * radius
    scale_val = math.log(0.06 * radius)
    return GaussianTemplate(
        means=means,
        scales=torch.full((n, 3), scale_val),
        opacities=torch.full((n,), 2.0),  # sigmoid(2) ≈ 0.88
    )


def make_cube(n: int = 200, size: float = 0.5) -> GaussianTemplate:
    """Cube: points on 6 faces."""
    per_face = n // 6
    faces = []
    half = size / 2.0
    for axis in range(3):
        for sign in [-1.0, 1.0]:
            pts = torch.rand(per_face, 3) * size - half
            pts[:, axis] = sign * half
            faces.append(pts)
    means = torch.cat(faces, dim=0)
    scale_val = math.log(0.06 * size)
    return GaussianTemplate(
        means=means,
        scales=torch.full((means.shape[0], 3), scale_val),
        opacities=torch.full((means.shape[0],), 2.0),
    )


def make_cylinder(n: int = 200, radius: float = 0.3, height: float = 1.0) -> GaussianTemplate:
    """Cylinder: points on side + top/bottom caps."""
    n_side = n * 2 // 3
    n_cap = (n - n_side) // 2
    # Side
    theta = 2.0 * math.pi * torch.rand(n_side)
    h = torch.rand(n_side) * height - height / 2.0
    side = torch.stack([radius * torch.cos(theta), h, radius * torch.sin(theta)], dim=-1)
    # Caps
    top = _uniform_disk(n_cap) * radius
    top[:, 1] = height / 2.0
    # Swap y/z for disk (disk generates on z=0, we want y=const)
    top = top[:, [0, 2, 1]]
    top[:, 1] = height / 2.0
    bot = _uniform_disk(n_cap) * radius
    bot = bot[:, [0, 2, 1]]
    bot[:, 1] = -height / 2.0
    means = torch.cat([side, top, bot], dim=0)
    scale_val = math.log(0.05)
    return GaussianTemplate(
        means=means,
        scales=torch.full((means.shape[0], 3), scale_val),
        opacities=torch.full((means.shape[0],), 2.0),
    )


def make_cone(n: int = 200, radius: float = 0.4, height: float = 1.0) -> GaussianTemplate:
    """Cone: points on side surface + base."""
    n_side = n * 3 // 4
    n_base = n - n_side
    # Side: radius shrinks linearly with height
    t = torch.rand(n_side)  # 0=base, 1=tip
    r = radius * (1.0 - t)
    theta = 2.0 * math.pi * torch.rand(n_side)
    h = t * height - height / 2.0
    side = torch.stack([r * torch.cos(theta), h, r * torch.sin(theta)], dim=-1)
    # Base
    base = _uniform_disk(n_base) * radius
    base = base[:, [0, 2, 1]]
    base[:, 1] = -height / 2.0
    means = torch.cat([side, base], dim=0)
    scale_val = math.log(0.05)
    return GaussianTemplate(
        means=means,
        scales=torch.full((means.shape[0], 3), scale_val),
        opacities=torch.full((means.shape[0],), 2.0),
    )


def make_plane(n: int = 100, width: float = 2.0, depth: float = 2.0) -> GaussianTemplate:
    """Flat plane at y=0."""
    side = int(math.sqrt(n))
    xs = torch.linspace(-width / 2, width / 2, side)
    zs = torch.linspace(-depth / 2, depth / 2, side)
    grid_x, grid_z = torch.meshgrid(xs, zs, indexing="ij")
    means = torch.stack([grid_x.flatten(), torch.zeros(side * side), grid_z.flatten()], dim=-1)
    scale_val = math.log(width / side * 0.6)
    return GaussianTemplate(
        means=means,
        scales=torch.full((means.shape[0], 3), scale_val),
        opacities=torch.full((means.shape[0],), 2.0),
    )


def make_torus(n: int = 300, R: float = 0.4, r: float = 0.15) -> GaussianTemplate:
    """Torus (donut): major radius R, minor radius r."""
    u = 2.0 * math.pi * torch.rand(n)
    v = 2.0 * math.pi * torch.rand(n)
    x = (R + r * torch.cos(v)) * torch.cos(u)
    y = r * torch.sin(v)
    z = (R + r * torch.cos(v)) * torch.sin(u)
    means = torch.stack([x, y, z], dim=-1)
    scale_val = math.log(0.04)
    return GaussianTemplate(
        means=means,
        scales=torch.full((n, 3), scale_val),
        opacities=torch.full((n,), 2.0),
    )


# ── Template registry ──
TEMPLATE_BUILDERS = {
    "sphere":   make_sphere,
    "cube":     make_cube,
    "cylinder": make_cylinder,
    "cone":     make_cone,
    "plane":    make_plane,
    "torus":    make_torus,
}


def build_template_library(n_gaussians: int = 200, device: str = "cpu") -> dict[str, GaussianTemplate]:
    """Build all object templates. Call once at startup."""
    lib = {}
    for name, builder in TEMPLATE_BUILDERS.items():
        lib[name] = builder(n=n_gaussians).to(device)
    return lib
