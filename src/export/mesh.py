"""
Mesh extraction from Gaussian splat scenes.

Converts a point cloud of Gaussian centers into a triangle mesh via
Poisson surface reconstruction. Optionally bakes Gaussian colors as
vertex colors.

Requires: open3d (pip install open3d)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def extract_mesh_poisson(
    tensors: dict[str, torch.Tensor],
    output_path: str | Path,
    depth: int = 8,
    density_threshold: float = 0.1,
) -> dict:
    """
    Extract a triangle mesh from Gaussian positions using Poisson reconstruction.

    Args:
        tensors: dict with means [N,3], colors [N,3], opacities [N]
        output_path: where to write the .obj file
        depth: octree depth for Poisson (higher = more detail, slower)
        density_threshold: prune low-density vertices (fraction of max)

    Returns:
        dict with stats: n_vertices, n_faces, file_size_bytes
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required for mesh export: pip install open3d")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    means = tensors["means"].detach().cpu().numpy().astype(np.float64)
    colors = tensors["colors"].detach().cpu().numpy().astype(np.float64)
    opacities = tensors["opacities"].detach().cpu().numpy()

    # Filter out low-opacity Gaussians (they contribute little to the surface)
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities))
    mask = opacity_sigmoid > 0.1
    means = means[mask]
    colors = colors[mask]

    if len(means) < 4:
        raise ValueError(f"Too few visible Gaussians ({len(means)}) for mesh extraction")

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))

    # Estimate normals from local geometry
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True
    )

    # Remove low-density vertices (mesh boundary artifacts)
    densities = np.asarray(densities)
    threshold = densities.max() * density_threshold
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Transfer vertex colors from nearest Gaussian
    mesh.compute_vertex_normals()

    # Write OBJ
    suffix = output_path.suffix.lower()
    if suffix == ".obj":
        o3d.io.write_triangle_mesh(str(output_path), mesh,
                                   write_vertex_normals=True,
                                   write_vertex_colors=True)
    elif suffix == ".ply":
        o3d.io.write_triangle_mesh(str(output_path), mesh)
    else:
        o3d.io.write_triangle_mesh(str(output_path), mesh)

    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.triangles)

    return {
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "file_size_bytes": output_path.stat().st_size,
    }
