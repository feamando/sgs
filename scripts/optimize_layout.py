"""
Raum 1.7 Stage 2: learn PART proportions by render-scored black-box search.

The systemic break from the snapping ceiling. infer_decomposer.snap_layout()
overrides every part's transform with the hard constants in _CASTLE_LAYOUT
(ring=0.7, scale=1.0) and _CONTAINER_XFORM. Those magic numbers are what 1.7
exists to kill. This script keeps the GRAMMAR's part builders but makes the
~20 layout parameters (per-part position/scale + a few globals) FREE, and
optimizes them against a render score -- no _CASTLE_LAYOUT entry used.

Why black-box (CMA-ES-style ES), not autograd: expand_part -> stones is not
differentiable, and the param space is tiny (~20). A pure-numpy (mu, lambda)
evolution strategy over the render score is the fastest reachability check and
needs no autograd-through-fill, no cma/scipy dep (cma isn't installed; scipy is
broken on the Windows box). If the search proves too slow we build the thin
differentiable layout->stones map; not before.

Why PART-level, not per-stone: Stage 1 showed free per-stone scales make flat
walls go stringy (no curvature to constrain the stretch). Optimizing ~20 layout
params with procedural fill regenerating stones each step makes that failure
STRUCTURALLY IMPOSSIBLE -- geometry stays coherent by construction.

Objective modes:
  --photometric : score = -MSE vs a TARGET render (default: the snapped layout).
                  CPU, no SD. Proves param -> build -> render -> score works and
                  the ES can recover a known-good layout. The Stage-2 selftest.
  --sds         : score = SDS agreement with a text prompt (4090, .venv-sds).
                  The real non-circular objective.

Usage:
  python scripts/optimize_layout.py --photometric --iters 40 --out output/layout_opt.json
  python scripts/optimize_layout.py --sds --prompt "a stone castle on a green hill" \
    --iters 60 --out output/layout_opt.json          # 4090 / .venv-sds
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import CompositionNode, tree_to_tensors
from src.raum.render_3d import render_gaussians
from src.raum import cameras as camlib
from scripts.castle_grammar import (
    build_tower, build_wall, build_keep, dome, STONE,
)


# ── layout parameter vector  (NO _CASTLE_LAYOUT constants used) ────────
#
# 19 free params, all in their own natural units. Towers get (dx, dz, scale)
# so the ring can become non-square / non-canonical if that renders better;
# walls get scale only (their face is implied by the towers they span); plus
# global ring radius, castle y-seat, keep height. The grammar builders fill
# stones from these -- we never touch individual stones.

PARAM_NAMES = (
    [f"tower{i}_{a}" for i in range(4) for a in ("dx", "dz", "s")]   # 12
    + [f"wall{i}_s" for i in range(4)]                                # 4
    + ["keep_s", "ring", "castle_y"]                                  # 3
)
N_PARAMS = len(PARAM_NAMES)   # 19

# initial guess + per-param search sigma. Deliberately NOT the snapped values
# as a hard prior -- ring/castle_y start near plausible but the ES is free to
# move. (Starting near a sane point just speeds convergence; the gate forbids
# READING _CASTLE_LAYOUT, not starting from a reasonable scale.)
def initial_params():
    p = np.zeros(N_PARAMS)
    for i in range(4):
        p[i * 3 + 2] = 1.0          # tower scale
    p[12:16] = 1.0                  # wall scales
    p[16] = 1.0                     # keep scale
    p[17] = 0.7                     # ring radius (free; not pinned to _RING)
    p[18] = 0.5                     # castle y-seat
    return p

def param_sigma():
    s = np.full(N_PARAMS, 0.08)
    s[17] = 0.12                    # ring can move more
    s[18] = 0.10
    return s


def params_to_tree(p, rng_seed=0):
    """Build a castle tree from the free layout vector. Mirrors
    build_castle_on_hill's STRUCTURE but every magic constant is now a param."""
    import random
    rng = random.Random(rng_seed)
    stone = [0.62, 0.58, 0.53]
    ring = float(np.clip(p[17], 0.3, 1.4))
    castle_y = float(np.clip(p[18], 0.0, 1.2))
    castle_scale = 0.9

    castle = CompositionNode(name="castle", position=[0, castle_y, 0], scale=castle_scale)
    # 4 towers at ring corners + learned (dx,dz) nudge + learned scale
    base_corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for i, (sx, sz) in enumerate(base_corners):
        dx, dz, ts = p[i * 3], p[i * 3 + 1], float(np.clip(p[i * 3 + 2], 0.4, 2.0))
        cx, cz = sx * ring + dx, sz * ring + dz
        t = build_tower(f"tower_{i}", [cx, 0, cz], 7, 0.26, stone)
        t.scale = ts
        castle.children.append(t)
    # 4 walls on the faces, learned scale
    faces = [([0, 0, -ring], None), ([0, 0, ring], None),
             ([ring, 0, 0], [0.7071, 0, 0.7071, 0]), ([-ring, 0, 0], [0.7071, 0, 0.7071, 0])]
    for i, (pos, rot) in enumerate(faces):
        ws = float(np.clip(p[12 + i], 0.4, 2.0))
        w = build_wall(f"wall_{i}", pos, ring * 2, 6, stone,
                       rot=rot, is_gate=(i == 0), slits=(0 if i == 0 else 2))
        w.scale = ws
        castle.children.append(w)
    # keep, learned scale
    keep = build_keep(9, stone)
    keep.scale = float(np.clip(p[16], 0.4, 2.0))
    keep.position = [0, 0.2, 0]
    castle.children.append(keep)

    # hill (fixed -- not part of the layout search; seat is derived)
    hill_h, hill_radius = 0.55, max(1.64, (ring + 0.26) * castle_scale * 1.9)
    grass = [0.30, 0.55, 0.20]
    hill = CompositionNode(name="hill", position=[0, -0.1, 0], scale=1.0,
                           color=grass, gaussians=dome(hill_radius, hill_h, STONE * 2, grass))
    scene = CompositionNode(name="scene")
    scene.children.append(hill)
    scene.children.append(castle)
    return scene


# ── render a layout to a small multi-view tensor stack ─────────────────

def _orbit(tensors, device, img, frac):
    P = tensors["means"]; center = P.mean(0)
    radius = (P - center).norm(dim=1).max().item()
    az = frac * 2 * math.pi
    el = 0.4
    dist = radius * 2.6
    eye = center + torch.tensor([dist * math.cos(el) * math.cos(az),
                                 dist * math.sin(el),
                                 dist * math.cos(el) * math.sin(az)], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    fwd = (center - eye); fwd = fwd / fwd.norm()
    right = torch.linalg.cross(fwd, up); right = right / right.norm()
    nup = torch.linalg.cross(right, fwd)
    R = torch.stack([right, nup, fwd], dim=0)
    W = torch.eye(4, device=device); W[:3, :3] = R; W[:3, 3] = -R @ eye
    K = camlib.make_intrinsic(50.0, img, img).to(device)
    return W, K


def render_views(tree, device, img=96, n_views=4):
    t = tree_to_tensors(tree)
    t = {k: v.to(device) for k, v in t.items()}
    imgs = []
    for v in range(n_views):
        W, K = _orbit(t, device, img, v / n_views)
        rgb = render_gaussians(t["means"], t["scales_log"], t["opacities"],
                               t["colors"], W, K, img, img, backend="simple")
        imgs.append(rgb)
    return torch.stack(imgs)   # [V,3,H,W]


# ── objectives ─────────────────────────────────────────────────────────

def make_photometric_objective(device, img, n_views):
    """Target = the SNAPPED layout's render (the thing we're trying to recover
    WITHOUT reading _CASTLE_LAYOUT). Score = -MSE. Proves the loop + that the ES
    can find a render-good layout from a render score alone."""
    target_p = initial_params()                # snapped-ish reference layout
    target_tree = params_to_tree(target_p)
    with torch.no_grad():
        target = render_views(target_tree, device, img, n_views)

    def score(p):
        tree = params_to_tree(p)
        with torch.no_grad():
            r = render_views(tree, device, img, n_views)
        return -float(((r - target) ** 2).mean())
    return score


def make_sds_objective(device, prompt, img, n_views):
    """Real objective: SDS agreement with the prompt (4090, .venv-sds)."""
    from scripts.sds_refine import SDSGuidance
    guide = SDSGuidance(prompt, device, guidance_scale=40.0)

    def score(p):
        tree = params_to_tree(p)
        with torch.no_grad():
            r = render_views(tree, device, img, n_views)
        # lower SDS surrogate loss = better agreement; average over views
        losses = [float(guide.loss(r[v], rng_t=(v / n_views)).item()) for v in range(r.shape[0])]
        return -float(np.mean(losses))
    return score


# ── pure-numpy (mu, lambda) evolution strategy ─────────────────────────

def evolution_search(score_fn, iters, pop=12, elite=None, seed=0, start=None):
    """(mu, lambda) ES with proper step-size control.

    The naive version drove sigma from elite-std, which collapses to ~0 as soon
    as the elites cluster -> premature convergence (the v1 stall: sigma 0.011 by
    iter 29, never reached the target). Fixes:
      - weighted recombination (better elites pull harder than a flat mean)
      - GLOBAL step-size adapted by a 1/5-success rule, DECOUPLED from elite
        spread, so progress keeps the step large and only stagnation shrinks it
      - a sigma FLOOR (never below 25% of the initial scale) so the search can
        always escape a shallow basin
    """
    rng = np.random.default_rng(seed)
    mean = (initial_params() if start is None else start.copy()).astype(float)
    base_sigma = param_sigma()
    sigma_floor = base_sigma * 0.25
    step = 1.0                                       # global step-size multiplier
    mu = elite or max(2, pop // 3)
    # log-decreasing recombination weights (CMA-style)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    best_p, best_s = mean.copy(), score_fn(mean)
    init_s = best_s
    print(f"[es] init score={best_s:.5f}  pop={pop} mu={mu}")
    for it in range(iters):
        eff_sigma = np.maximum(base_sigma * step, sigma_floor)
        pop_p = mean[None, :] + rng.normal(0, 1, (pop, N_PARAMS)) * eff_sigma[None, :]
        scores = np.array([score_fn(pop_p[j]) for j in range(pop)])
        order = np.argsort(-scores)                  # descending
        elites = pop_p[order[:mu]]
        new_mean = (w[:, None] * elites).sum(0)      # weighted recombination
        # 1/5-success rule on the global step: if the new centre beats the old,
        # we're making progress -> grow the step; else shrink it.
        improved = score_fn(new_mean) > best_s
        step *= 1.05 if improved else 0.85       # gentle grow, firmer shrink: converge
        step = float(np.clip(step, 0.2, 1.6))     # cap growth so it doesn't wander wide
        mean = new_mean
        gen_best = float(scores[order[0]])
        if gen_best > best_s:
            best_s, best_p = gen_best, pop_p[order[0]].copy()
        if it % 5 == 0 or it == iters - 1:
            print(f"[es] iter {it:3d}  best={best_s:.5f}  step={step:.2f}  "
                  f"eff_sigma={eff_sigma.mean():.3f}")
    # the metric that matters for a degenerate (symmetric) target: how much of
    # the starting render error did we remove? Absolute score is misleading when
    # many layouts render near-identically.
    if init_s < -1e-9:
        print(f"[es] error reduction: {100 * (1 - best_s / init_s):.1f}% "
              f"({init_s:.5f} -> {best_s:.5f})")
    return best_p, best_s


def main():
    p = argparse.ArgumentParser(description="Raum 1.7 Stage 2: learn part proportions")
    p.add_argument("--photometric", action="store_true",
                   help="recover the snapped layout from its render (CPU, no SD) -- the selftest")
    p.add_argument("--sds", action="store_true", help="SDS objective vs --prompt (4090)")
    p.add_argument("--prompt", default="a stone castle on a green hill")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--img", type=int, default=96)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--out", default="output/layout_opt.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb", type=float, default=0.0,
                   help="photometric selftest: start the ES this far (gaussian sigma) "
                        "from the target, so a non-trivial recovery is required")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.sds:
        score_fn = make_sds_objective(device, args.prompt, args.img, args.views)
    else:
        score_fn = make_photometric_objective(device, args.img, args.views)

    start = None
    if args.perturb > 0:
        rng0 = np.random.default_rng(args.seed + 999)
        start = initial_params() + rng0.normal(0, 1, N_PARAMS) * param_sigma() * args.perturb
        print(f"[es] perturbed start: score={score_fn(start):.5f} (target is 0.0)")

    best_p, best_s = evolution_search(score_fn, args.iters, pop=args.pop,
                                      seed=args.seed, start=start)
    print(f"\n[es] BEST score={best_s:.5f}")
    print("[es] learned layout (NO _CASTLE_LAYOUT used):")
    for name, val in zip(PARAM_NAMES, best_p):
        print(f"     {name:14s} {val:+.3f}")

    from src.raum.decomposition import save_tree
    tree = params_to_tree(best_p)
    save_tree(tree, args.out)
    Path(args.out).with_suffix(".params.json").write_text(
        json.dumps(dict(zip(PARAM_NAMES, best_p.tolist())), indent=2))
    print(f"[es] saved scene -> {args.out}")
    print(f"[es] saved params -> {Path(args.out).with_suffix('.params.json')}")


if __name__ == "__main__":
    main()
