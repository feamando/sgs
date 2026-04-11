"""
Camera utilities for multi-view rendering.

Orbit cameras around scene center at fixed elevation.
"""

import math
import torch
from dataclasses import dataclass


@dataclass
class Camera:
    """Simple pinhole camera."""
    world_to_cam: torch.Tensor   # [4, 4] extrinsic
    K: torch.Tensor              # [3, 3] intrinsic
    width: int
    height: int

    @property
    def cam_to_world(self) -> torch.Tensor:
        return torch.inverse(self.world_to_cam)

    @property
    def position(self) -> torch.Tensor:
        """Camera position in world coords."""
        return self.cam_to_world[:3, 3]

    def to(self, device):
        return Camera(
            world_to_cam=self.world_to_cam.to(device),
            K=self.K.to(device),
            width=self.width,
            height=self.height,
        )


def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Compute world-to-camera [4,4] matrix.

    OpenGL convention: camera looks down -z, y is up, x is right.
    """
    forward = target - eye
    forward = forward / forward.norm()
    right = torch.cross(forward, up)
    right = right / right.norm()
    new_up = torch.cross(right, forward)

    R = torch.stack([right, new_up, -forward], dim=0)  # [3, 3]

    t = -R @ eye  # [3]

    W = torch.eye(4)
    W[:3, :3] = R
    W[:3, 3] = t
    return W


def make_intrinsic(fov_deg: float, width: int, height: int) -> torch.Tensor:
    """Pinhole intrinsic matrix from field-of-view."""
    fov_rad = fov_deg * math.pi / 180.0
    fx = fy = 0.5 * width / math.tan(0.5 * fov_rad)
    cx, cy = width / 2.0, height / 2.0
    K = torch.tensor([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=torch.float32)
    return K


def orbit_cameras(
    n_views: int = 8,
    elevation_deg: float = 30.0,
    radius: float = 4.0,
    fov_deg: float = 50.0,
    img_size: int = 128,
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[Camera]:
    """
    Generate orbit cameras around a target point.

    Cameras are evenly spaced in azimuth at fixed elevation and radius.
    """
    up = torch.tensor([0.0, 1.0, 0.0])
    tgt = torch.tensor(target, dtype=torch.float32)
    K = make_intrinsic(fov_deg, img_size, img_size)

    elev_rad = elevation_deg * math.pi / 180.0
    cameras = []
    for i in range(n_views):
        az_rad = 2.0 * math.pi * i / n_views
        x = radius * math.cos(elev_rad) * math.cos(az_rad)
        y = radius * math.sin(elev_rad)
        z = radius * math.cos(elev_rad) * math.sin(az_rad)
        eye = torch.tensor([x, y, z], dtype=torch.float32) + tgt

        W = look_at(eye, tgt, up)
        cameras.append(Camera(world_to_cam=W, K=K, width=img_size, height=img_size))

    return cameras
