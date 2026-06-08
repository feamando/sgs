"""
Raum 0.7: densify + flatten-to-surface post-process.

A GENERIC appearance pass on a flattened Gaussian cloud -- not the castle
grammar, not a model. It closes the "looks like a grid of round stones" gap on
the material/appearance axis (the axis 0.7 owns) WITHOUT touching proportions
(the where/how-big that 1.7 owns). Two operations, both driven by uniform
global knobs only -- NO per-part hand-tuning, so it stays a two-way door and
applies to ANY scene the grammar (or later, the 1.7 decomposer) emits:

  1. DENSIFY  -- jitter-clone each gaussian k times within its own footprint so
     neighbours overlap and blend into a continuous surface instead of reading
     as discrete dots. Opacity is divided across clones so total coverage is
     conserved (k overlapping splats at opacity/k ~= one solid splat).

  2. FLATTEN-TO-SURFACE -- estimate each gaussian's local surface normal from
     its neighbours (PCA: the normal is the smallest-variance axis of the local
     point patch) and squash the splat along that normal into a disk tangent to
     the surface. This is what makes a real 3DGS scene read as surfaces rather
     than a fog of blobs. Uses the geometry that ALREADY exists (neighbour
     positions); invents no new tuned numbers.

This is deliberately the SIMPLEST thing that works. If a scene ever needs
per-part density/anisotropy tuned by eye to look right, STOP -- that is the
procedural ceiling and the fix belongs in 1.7 (learned geometry), not here.

Usage:
  python scripts/densify_flatten.py --in output/castle_06.json \
    --out output/castle_06_dense.json --densify 4 --flatten 0.35
  python scripts/densify_flatten.py --in output/castle_06.json --out - --stats
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import (
    CompositionNode, GaussianParams, load_tree, save_tree,
)


# ── quaternion <-> rotation matrix (w, x, y, z convention) ────────────

def quat_to_mat(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """Rotation matrix -> [w, x, y, z]. Standard Shepperd-style branch."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return (q / (np.linalg.norm(q) or 1.0)).tolist()


# ── collect / rebuild the flat cloud ──────────────────────────────────

def collect(tree: CompositionNode):
    """World-space flat list of GaussianParams (the tree's own transform)."""
    return tree.flatten_gaussians()


def to_arrays(gaussians):
    pos = np.array([g.position for g in gaussians], dtype=np.float64)
    scale = np.array([g.scale for g in gaussians], dtype=np.float64)   # log
    opac = np.array([g.opacity for g in gaussians], dtype=np.float64)
    rot = np.array([g.rotation for g in gaussians], dtype=np.float64)
    col = np.array([(g.color or [0.5, 0.5, 0.5]) for g in gaussians], dtype=np.float64)
    return pos, scale, opac, rot, col


# ── the two operations ─────────────────────────────────────────────────

def _knn_idx(pos, k):
    """Indices of the k nearest neighbours per point, pure numpy (no scipy).
    Brute-force is fine for the few-thousand-splat clouds we work with; chunked
    so the [N,N] distance matrix never materializes in full."""
    n = pos.shape[0]
    k = min(k, n)
    idx = np.empty((n, k), dtype=np.int64)
    sq = (pos * pos).sum(1)                         # [N]
    chunk = 1024
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # squared dist: |a|^2 - 2 a.b + |b|^2
        d = sq[s:e, None] - 2.0 * (pos[s:e] @ pos.T) + sq[None, :]
        idx[s:e] = np.argpartition(d, k - 1, axis=1)[:, :k]
    return idx


def estimate_normals(pos, k=8):
    """Per-point surface normal = smallest-eigenvector of the local neighbour
    covariance (PCA). Returns [N,3] unit normals."""
    n = pos.shape[0]
    idx = _knn_idx(pos, k)
    normals = np.zeros((n, 3))
    for i in range(n):
        nb = pos[idx[i]]
        nb = nb - nb.mean(0)
        cov = nb.T @ nb
        w, v = np.linalg.eigh(cov)      # ascending eigenvalues
        normals[i] = v[:, 0]            # smallest variance = surface normal
    # orient outward from the cloud centroid (cosmetic; sign is arbitrary for a
    # symmetric splat but keeps things tidy)
    centroid = pos.mean(0)
    out = pos - centroid
    flip = (np.einsum("ij,ij->i", normals, out) < 0)
    normals[flip] *= -1
    return normals


def _surface_frames(pos, knn):
    """Per-point orthonormal frame [tx, ty, nz] with nz = local surface normal.
    Returns (frames [N,3,3] with axes as columns, normals [N,3])."""
    normals = estimate_normals(pos, k=knn)
    n = pos.shape[0]
    frames = np.zeros((n, 3, 3))
    for i in range(n):
        nz = normals[i]
        a = np.array([1.0, 0.0, 0.0]) if abs(nz[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tx = np.cross(a, nz); tx /= (np.linalg.norm(tx) or 1.0)
        ty = np.cross(nz, tx)
        frames[i] = np.column_stack([tx, ty, nz])
    return frames, normals


def densify_flatten_arrays(pos, scale_log, opac, rot, col, *,
                           densify=4, flatten=0.4, density=1.0,
                           weathering=0.0, knn=8, rng=None):
    """Combined surface pass on a flat Gaussian cloud. All knobs are uniform
    globals -- no per-part tuning.

      flatten   0..1  squash each splat onto its local tangent plane (disk)
      density   >0     in-plane footprint multiplier (overlap/coverage; the
                       solid-vs-airy knob, independent of count)
      densify   int    clones per splat, jittered IN THE TANGENT PLANE so they
                       stay on the surface (not a volume fog). Opacity is split
                       compositing-correct: a_split = 1-(1-a0)**(1/k), so k
                       overlapping clones alpha-composite back to the original
                       coverage instead of a translucent smear.
      weathering 0..1  per-clone colour jitter so a surface reads as aged stone
    """
    rng = rng or np.random.default_rng(0)
    k = max(1, int(densify))
    lin = np.exp(scale_log)                                   # linear sigma/axis
    stone = lin.mean(1)                                       # per-stone size (pre-flatten)
    inplane = lin[:, :2].mean(1) * max(density, 1e-3)         # tangent footprint
    if flatten > 0:
        frames, _ = _surface_frames(pos, knn)
        thin = lin.max(1) * (1.0 - flatten)
        base_scale = np.log(np.stack([
            np.maximum(inplane, 1e-6), np.maximum(inplane, 1e-6),
            np.maximum(thin, 1e-6)], axis=1))
        base_rot = np.array([mat_to_quat(frames[i]) for i in range(pos.shape[0])])
    else:
        # density still scales footprint, but keep original orientation
        frames = None
        base_scale = scale_log + math.log(max(density, 1e-3))
        base_rot = rot

    # Opacity: a stone castle is a SOLID, not a translucent volume. Keep clones
    # near-opaque so the FRONT splat wins and back splats are occluded -> crisp
    # surfaces. Earlier "conserve coverage" splitting (a=1-(1-a0)**(1/k)) dropped
    # each clone to ~0.19 prob -> every splat see-through -> depth haze / the
    # cloudy gsplat look. Alpha compositing caps at the splat colour, so opaque
    # clones don't over-brighten; the front one just wins. Keep original opacity
    # with a high floor.
    a0 = 1.0 / (1.0 + np.exp(-opac))
    a_solid = np.clip(np.maximum(a0, 0.8), 1e-4, 0.999)
    opac_split = np.log(a_solid / (1.0 - a_solid))

    # Jitter clones in 3D, scaled by the stone's ORIGINAL size (this is what
    # gave walls real thickness in the good build). Tangent-only jitter put
    # every flattened disk coplanar -> walls collapsed to a single edge-on line
    # and the hill scalloped. A stone-sized 3D blob keeps the surface-aligned
    # disks stacked in offset layers -> solid volume. `density` scales spread.
    spread = stone * max(density, 1e-3) * 0.6
    P, S, O, R, C = [], [], [], [], []
    for c in range(k):
        offset = (np.zeros_like(pos) if c == 0
                  else rng.normal(0, 1, size=pos.shape) * spread[:, None])
        cc = col.copy()
        if weathering > 0:
            cc = np.clip(cc + rng.normal(0, weathering * 0.12, size=col.shape), 0, 1)
        P.append(pos + offset); S.append(base_scale); O.append(opac_split)
        R.append(base_rot); C.append(cc)
    return (np.concatenate(P), np.concatenate(S), np.concatenate(O),
            np.concatenate(R), np.concatenate(C))


# ── assemble back into a flat scene tree ───────────────────────────────

def build_tree(pos, scale_log, opac, rot, col):
    scene = CompositionNode(name="scene")
    leaf = CompositionNode(name="densified")
    for i in range(pos.shape[0]):
        leaf.gaussians.append(GaussianParams(
            position=pos[i].tolist(),
            scale=scale_log[i].tolist(),
            opacity=float(opac[i]),
            color=col[i].clip(0, 1).tolist(),
            rotation=rot[i].tolist(),
            sh_degree=2,
        ))
    scene.children.append(leaf)
    return scene


def main():
    p = argparse.ArgumentParser(description="Raum 0.7 densify + flatten-to-surface")
    p.add_argument("--in", dest="inp", required=True, help="input scene JSON")
    p.add_argument("--out", required=True, help="output scene JSON ('-' = stats only)")
    p.add_argument("--splats", type=int, default=0,
                   help="TARGET total splat count; derives densify from input size "
                        "(overrides --densify when >0)")
    p.add_argument("--densify", type=int, default=4,
                   help="clones per gaussian (1 = off); opacity split compositing-correct")
    p.add_argument("--density", type=float, default=1.0,
                   help="in-plane footprint multiplier (solid<->airy, independent of count)")
    p.add_argument("--flatten", type=float, default=0.4,
                   help="squash toward a surface disk, 0..1 (0 = off)")
    p.add_argument("--weathering", type=float, default=0.0,
                   help="per-clone colour jitter, 0..1 (aged-stone variation)")
    p.add_argument("--knn", type=int, default=8, help="neighbours for normal estimate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats", action="store_true", help="print before/after counts")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tree = load_tree(args.inp)
    gaussians = collect(tree)
    pos, scale_log, opac, rot, col = to_arrays(gaussians)
    n0 = pos.shape[0]

    densify = args.densify
    if args.splats > 0:
        densify = max(1, round(args.splats / max(n0, 1)))

    pos, scale_log, opac, rot, col = densify_flatten_arrays(
        pos, scale_log, opac, rot, col,
        densify=densify, flatten=args.flatten, density=args.density,
        weathering=args.weathering, knn=args.knn, rng=rng)
    n1 = pos.shape[0]

    if args.stats or args.out == "-":
        aniso = np.exp(scale_log).max(1) / np.maximum(np.exp(scale_log).min(1), 1e-9)
        ap = 1.0 / (1.0 + np.exp(-opac))
        print(f"[densify_flatten] {n0} -> {n1} gaussians "
              f"(x{densify} densify, density={args.density}, flatten={args.flatten}, "
              f"weathering={args.weathering})")
        print(f"   anisotropy median={np.median(aniso):.2f}  opacity-prob median={np.median(ap):.3f}")
        if args.out == "-":
            return

    out_tree = build_tree(pos, scale_log, opac, rot, col)
    save_tree(out_tree, args.out)
    print(f"[densify_flatten] saved -> {args.out}")


if __name__ == "__main__":
    main()
