"""
Differentiable 3DGS rendering for Raum PoC-C.

Two backends:
  1. gsplat (preferred, fast, CUDA)
  2. Simple alpha-blend fallback (CPU-compatible, uses our own rendering equation)
"""

import torch
import torch.nn.functional as F
import math

# Try to import gsplat; fall back gracefully
_GSPLAT_AVAILABLE = False
try:
    import gsplat
    _GSPLAT_AVAILABLE = True
except ImportError:
    pass


def render_gaussians(
    means: torch.Tensor,        # [N, 3]
    scales: torch.Tensor,       # [N, 3] log-scale
    opacities: torch.Tensor,    # [N] logit (pre-sigmoid)
    colors: torch.Tensor,       # [N, 3] RGB
    viewmat: torch.Tensor,      # [4, 4] world-to-camera
    K: torch.Tensor,            # [3, 3] intrinsic
    width: int = 128,
    height: int = 128,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Render a Gaussian scene to an image.

    Args:
        means: Gaussian centers in world space
        scales: log-scale per axis
        opacities: pre-sigmoid opacity
        colors: RGB color per Gaussian
        viewmat: camera extrinsic
        K: camera intrinsic
        width, height: output image size
        backend: "gsplat", "simple", or "auto"

    Returns:
        image: [3, H, W] RGB tensor in [0, 1]
    """
    if backend == "auto":
        backend = "gsplat" if _GSPLAT_AVAILABLE and means.is_cuda else "simple"

    if backend == "gsplat":
        return _render_gsplat(means, scales, opacities, colors, viewmat, K, width, height)
    else:
        return _render_simple(means, scales, opacities, colors, viewmat, K, width, height)


# ═══════════════════════════════════════════════════════════
# Backend 1: gsplat (fast, CUDA)
# ═══════════════════════════════════════════════════════════

def _render_gsplat(means, scales, opacities, colors, viewmat, K, width, height):
    """Render via gsplat library."""
    N = means.shape[0]
    device = means.device

    # gsplat expects quaternion rotations — use identity (isotropic Gaussians)
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # identity quaternion

    # gsplat rasterization
    renders, alphas, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales.exp(),
        opacities=opacities.sigmoid(),
        colors=colors,
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        packed=False,
    )
    # renders: [1, H, W, 3]
    return renders[0].permute(2, 0, 1).clamp(0, 1)  # [3, H, W]


# ═══════════════════════════════════════════════════════════
# Backend 2: Simple alpha-composite (CPU-compatible)
# ═══════════════════════════════════════════════════════════

def _project_to_2d(means_3d, viewmat, K):
    """Project 3D points to 2D pixel coordinates + depth."""
    N = means_3d.shape[0]
    # World → camera
    pts_h = torch.cat([means_3d, torch.ones(N, 1, device=means_3d.device)], dim=-1)
    pts_cam = (viewmat @ pts_h.T).T[:, :3]  # [N, 3]
    depth = pts_cam[:, 2].clamp(min=0.01)

    # Camera → pixel
    pts_2d = (K @ pts_cam.T).T  # [N, 3]
    px = pts_2d[:, 0] / depth
    py = pts_2d[:, 1] / depth
    return px, py, depth


def _render_simple(means, scales, opacities, colors, viewmat, K, width, height):
    """
    Simple differentiable splatting via alpha-compositing.

    For each pixel, evaluate 2D Gaussian contributions and alpha-composite
    front-to-back. This is our rendering equation applied to image space.
    """
    N = means.shape[0]
    device = means.device

    # Project to 2D
    px, py, depth = _project_to_2d(means, viewmat, K)
    alpha = opacities.sigmoid()

    # 2D scales from 3D log-scales + depth
    # Approximate: screen-space size ≈ world size * focal / depth
    fx = K[0, 0]
    scale_world = scales.exp().mean(dim=-1)  # average 3D scale
    scale_2d = (scale_world * fx / depth).clamp(min=0.5, max=width / 2)

    # Sort by depth (front to back) for correct alpha-compositing
    order = depth.argsort()
    px, py, depth = px[order], py[order], depth[order]
    alpha = alpha[order]
    scale_2d = scale_2d[order]
    colors_sorted = colors[order]

    # Render: evaluate each Gaussian at each pixel
    # For efficiency, only consider Gaussians within 3*scale of each pixel
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # xx, yy: [H, W]

    # Initialize accumulation
    image = torch.zeros(3, height, width, device=device)
    transmittance = torch.ones(height, width, device=device)

    # Alpha-composite front to back (sorted by depth)
    for i in range(min(N, 500)):  # cap at 500 for speed
        dx = xx - px[i]
        dy = yy - py[i]
        s = scale_2d[i]
        gauss = torch.exp(-0.5 * (dx * dx + dy * dy) / (s * s + 1e-6))

        eff_alpha = alpha[i] * gauss  # [H, W]
        eff_alpha = eff_alpha.clamp(max=0.99)

        weight = eff_alpha * transmittance  # [H, W]
        c = colors_sorted[i]  # [3]
        image += weight.unsqueeze(0) * c.view(3, 1, 1)
        transmittance = transmittance * (1.0 - eff_alpha)

    return image.clamp(0, 1)


# ═══════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════

def render_scene_multiview(
    means, scales, opacities, colors, cameras, backend="auto",
) -> list[torch.Tensor]:
    """Render from multiple cameras. Returns list of [3, H, W] tensors."""
    images = []
    for cam in cameras:
        img = render_gaussians(
            means, scales, opacities, colors,
            cam.world_to_cam.to(means.device),
            cam.K.to(means.device),
            cam.width, cam.height,
            backend=backend,
        )
        images.append(img)
    return images


def check_backend():
    """Print which rendering backend is available."""
    if _GSPLAT_AVAILABLE:
        print("Rendering backend: gsplat (CUDA)")
    else:
        print("Rendering backend: simple alpha-composite (CPU)")
    return "gsplat" if _GSPLAT_AVAILABLE else "simple"
