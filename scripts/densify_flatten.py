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


def flatten_to_surface(pos, scale_log, rot, amount, k=8):
    """Orient each splat's short axis to the local surface normal and squash it
    along that axis by `amount` (0 = unchanged, ->1 = fully flat disk).

    We REPLACE the rotation with a frame whose 3rd axis = normal, and scale the
    3rd (log) axis down. The in-plane axes keep the mean of the original two
    so footprint area is roughly preserved."""
    if amount <= 0:
        return scale_log, rot
    normals = estimate_normals(pos, k=k)
    new_scale = scale_log.copy()
    new_rot = rot.copy()
    lin = np.exp(scale_log)                       # linear per-axis sigma
    inplane = lin[:, :2].mean(1)                  # keep tangent footprint
    thin = lin.max(1) * (1.0 - amount)            # squash normal axis
    for i in range(pos.shape[0]):
        nz = normals[i]
        # build an orthonormal frame [tx, ty, nz]
        a = np.array([1.0, 0.0, 0.0]) if abs(nz[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tx = np.cross(a, nz); tx /= (np.linalg.norm(tx) or 1.0)
        ty = np.cross(nz, tx)
        R = np.column_stack([tx, ty, nz])         # columns = splat local axes
        new_rot[i] = mat_to_quat(R)
        new_scale[i] = np.log(np.array([
            max(inplane[i], 1e-6), max(inplane[i], 1e-6), max(thin[i], 1e-6)]))
    return new_scale, new_rot


def densify(pos, scale_log, opac, rot, col, k, rng):
    """Jitter-clone each gaussian k times inside its own footprint. Opacity is
    split across clones (logit space -> probability -> /k -> logit) so total
    coverage is conserved rather than k-fold brighter."""
    if k <= 1:
        return pos, scale_log, opac, rot, col
    sigma = np.exp(scale_log)                      # linear per-axis sigma
    p = 1.0 / (1.0 + np.exp(-opac))                # sigmoid -> coverage prob
    p_split = np.clip(p / k, 1e-4, 1 - 1e-4)
    opac_split = np.log(p_split / (1 - p_split))   # back to logit
    P, S, O, R, C = [], [], [], [], []
    for c in range(k):
        jitter = rng.normal(0, 1, size=pos.shape) * sigma * 0.6 if c > 0 else np.zeros_like(pos)
        P.append(pos + jitter)
        S.append(scale_log)
        O.append(opac_split)
        R.append(rot)
        C.append(col)
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
    p.add_argument("--densify", type=int, default=4,
                   help="clones per gaussian (1 = off); opacity split to conserve coverage")
    p.add_argument("--flatten", type=float, default=0.35,
                   help="squash toward a surface disk, 0..1 (0 = off)")
    p.add_argument("--knn", type=int, default=8, help="neighbours for normal estimate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stats", action="store_true", help="print before/after counts")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tree = load_tree(args.inp)
    gaussians = collect(tree)
    pos, scale_log, opac, rot, col = to_arrays(gaussians)
    n0 = pos.shape[0]

    # flatten FIRST (normals estimated on the clean cloud), then densify so
    # clones inherit the flattened, surface-aligned frame.
    scale_log, rot = flatten_to_surface(pos, scale_log, rot, args.flatten, k=args.knn)
    pos, scale_log, opac, rot, col = densify(pos, scale_log, opac, rot, col, args.densify, rng)
    n1 = pos.shape[0]

    if args.stats or args.out == "-":
        aniso = np.exp(scale_log).max(1) / np.maximum(np.exp(scale_log).min(1), 1e-9)
        print(f"[densify_flatten] {n0} -> {n1} gaussians "
              f"(x{args.densify} densify, flatten={args.flatten})")
        print(f"   anisotropy median={np.median(aniso):.2f} p90={np.percentile(aniso,90):.2f}")
        if args.out == "-":
            return

    out_tree = build_tree(pos, scale_log, opac, rot, col)
    save_tree(out_tree, args.out)
    print(f"[densify_flatten] saved -> {args.out}")


if __name__ == "__main__":
    main()
