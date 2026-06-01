"""
Appearance refinement for Raum 1.4 Phase C.

Three modes:
- 'sgs' (default): SGS-native refinement using template library as critic.
  Retrieves closest real GS reconstruction per semantic node, optimizes
  generated Gaussians to match via Chamfer distance. No external models.
- 'multiview': Multi-view consistency loss (penalizes floaters and gaps).
  No external models but weaker quality signal.
- 'sds': Score Distillation Sampling from Stable Diffusion. Strongest
  quality signal but requires diffusers/transformers/accelerate (~4-6 GB VRAM).

Usage:
    python scripts/refine_scene.py --input output/raum14_phase_b.json --mode sgs --templates data/objaverse_gs --output output/raum14_phase_c.json
    python scripts/refine_scene.py --input output/raum14_phase_b.json --mode multiview --output output/raum14_phase_c.json
    python scripts/refine_scene.py --input output/raum14_phase_b.json --mode sds --prompt "a castle" --output output/raum14_phase_c.json
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors, CompositionNode, GaussianParams, save_tree
from src.raum.densify import GaussianScene, DensifyConfig, densify_step


# ---------------------------------------------------------------------------
# SGS-native refinement
# ---------------------------------------------------------------------------

def load_template_library(templates_dir: Path) -> dict[str, torch.Tensor]:
    """
    Load all .pt template files from the Objaverse GS data directory.

    Returns dict mapping category -> stacked positions tensor [K, M, 3]
    where K is the number of objects in that category.
    """
    library = {}
    if not templates_dir.exists():
        return library

    for cat_dir in sorted(templates_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        tensors_list = []
        for obj_dir in sorted(cat_dir.iterdir()):
            model_path = obj_dir / "model.pt"
            if model_path.exists():
                data = torch.load(model_path, map_location="cpu", weights_only=True)
                tensors_list.append(data["positions"])
        if tensors_list:
            library[category] = tensors_list

    return library


def find_nearest_template(positions: torch.Tensor, library: dict[str, list[torch.Tensor]]) -> torch.Tensor | None:
    """
    Find the template whose point cloud is closest (Chamfer) to the given positions.

    Returns the best-matching template positions, or None if library is empty.
    """
    if not library:
        return None

    best_dist = float("inf")
    best_template = None

    # Normalize input
    center = positions.mean(dim=0)
    extent = (positions - center).abs().max() + 1e-6
    positions_norm = (positions - center) / extent

    for category, templates in library.items():
        for tpl_pos in templates:
            # Normalize template
            t_center = tpl_pos.mean(dim=0)
            t_extent = (tpl_pos - t_center).abs().max() + 1e-6
            tpl_norm = (tpl_pos - t_center) / t_extent

            # Subsample both to max 500 points for speed
            n1 = min(len(positions_norm), 500)
            n2 = min(len(tpl_norm), 500)
            p1 = positions_norm[:n1]
            p2 = tpl_norm[:n2]

            dists = torch.cdist(p1, p2)
            d1 = dists.min(dim=1).values.mean()
            d2 = dists.min(dim=0).values.mean()
            chamfer = (d1 + d2).item()

            if chamfer < best_dist:
                best_dist = chamfer
                best_template = tpl_norm

    return best_template


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer distance between two point clouds."""
    n1 = min(len(pred), 2000)
    n2 = min(len(target), 2000)
    p1 = pred[:n1]
    p2 = target[:n2]

    dists = torch.cdist(p1, p2)
    d1 = dists.min(dim=1).values.mean()
    d2 = dists.min(dim=0).values.mean()
    return d1 + d2


def find_local_templates(positions: torch.Tensor, library: dict[str, list[torch.Tensor]],
                         n_clusters: int = 8) -> list[torch.Tensor]:
    """
    Split the scene into spatial clusters and find the best template for each.

    Instead of matching the entire scene to one template, we partition into
    local regions (approximating composition tree nodes) and match each to
    the closest template from any category. Returns a list of target point
    clouds, one per cluster.
    """
    n = positions.shape[0]
    if n < n_clusters * 10:
        n_clusters = max(2, n // 50)

    # Simple spatial clustering via k-means (1 iteration for speed)
    # Pick random centroids
    idx = torch.randperm(n)[:n_clusters]
    centroids = positions[idx].clone()

    # Assign points to nearest centroid
    dists = torch.cdist(positions, centroids)
    assignments = dists.argmin(dim=1)

    # For each cluster, find best matching template
    all_templates_flat = []
    for cat_templates in library.values():
        all_templates_flat.extend(cat_templates)

    targets = []
    for c in range(n_clusters):
        mask = assignments == c
        if mask.sum() < 10:
            continue
        cluster_pos = positions[mask]

        # Normalize cluster
        center = cluster_pos.mean(dim=0)
        extent = (cluster_pos - center).abs().max() + 1e-6
        cluster_norm = (cluster_pos - center) / extent

        # Find best template for this cluster
        best_dist = float("inf")
        best_tpl = None
        sample_n = min(len(cluster_norm), 300)

        for tpl_pos in all_templates_flat:
            t_center = tpl_pos.mean(dim=0)
            t_extent = (tpl_pos - t_center).abs().max() + 1e-6
            tpl_norm = (tpl_pos - t_center) / t_extent

            p1 = cluster_norm[:sample_n]
            p2 = tpl_norm[:min(len(tpl_norm), 300)]
            d = torch.cdist(p1, p2)
            chamfer = d.min(dim=1).values.mean() + d.min(dim=0).values.mean()

            if chamfer.item() < best_dist:
                best_dist = chamfer.item()
                best_tpl = tpl_norm

        if best_tpl is not None:
            targets.append((mask, best_tpl, center, extent))

    return targets


def refine_sgs(tensors: dict[str, torch.Tensor], templates_dir: Path,
               n_iterations: int = 100, lr: float = 5e-4) -> dict[str, torch.Tensor]:
    """
    SGS-native refinement: optimize Gaussians toward nearest template shapes.

    Splits the scene into spatial clusters (approximating composition nodes)
    and matches each cluster to its best template. Each cluster is optimized
    toward its own target shape via Chamfer distance.

    Also applies:
    - Nearest-neighbor uniformity loss (no gaps)
    - Opacity regularization (encourage visible Gaussians)
    """
    print("Loading template library...")
    library = load_template_library(templates_dir)

    if not library:
        print(f"Warning: no templates found at {templates_dir}")
        print("Falling back to self-supervised refinement (uniformity + opacity only)")
        targets = None
    else:
        n_templates = sum(len(v) for v in library.values())
        print(f"Loaded {n_templates} templates across {len(library)} categories")

        print("Matching local clusters to templates...")
        targets = find_local_templates(tensors["means"], library, n_clusters=12)
        print(f"Matched {len(targets)} local regions to templates")

    scene = GaussianScene.from_tensors(tensors)
    if scene.n_gaussians == 0:
        print("Warning: refine_sgs received an empty scene; skipping refinement.")
        return scene.to_tensors()
    optimizer = torch.optim.Adam([scene.positions, scene.colors, scene.opacities], lr=lr)

    config = DensifyConfig(
        grad_threshold=0.0005,
        max_gaussians=min(scene.n_gaussians * 2, 100000),
        opacity_prune_threshold=0.01,
    )

    # Store template targets without masks (recompute assignments each time)
    template_targets = []
    if targets:
        for mask, target, center, extent in targets:
            template_targets.append((center, extent, target))

    print(f"Refining {scene.n_gaussians:,} Gaussians, {n_iterations} iterations...")

    for i in range(n_iterations):
        optimizer.zero_grad()

        loss = torch.tensor(0.0)

        # Loss 1: Per-cluster Chamfer distance to matched templates
        if template_targets:
            pos = scene.positions
            # Reassign points to nearest cluster center each iteration
            centers = torch.stack([t[0] for t in template_targets])
            dists_to_centers = torch.cdist(pos.detach(), centers)
            assignments = dists_to_centers.argmin(dim=1)

            for c_idx, (center, extent, target) in enumerate(template_targets):
                mask = assignments == c_idx
                if mask.sum() < 10:
                    continue
                cluster_pos = pos[mask]
                cluster_norm = (cluster_pos - center) / extent
                loss_chamfer = chamfer_loss(cluster_norm, target)
                loss = loss + loss_chamfer * (1.0 / len(template_targets))

        # Loss 2: Gap-filling (pull only the most isolated Gaussians inward).
        # NB: penalizing mean NN distance collapses the whole cloud into clumps
        # and destroys the macro shape. Instead, penalize only the worst gaps
        # (top-decile NN distance) with a small weight so structure survives.
        n = min(scene.n_gaussians, 2000)
        subset = scene.positions[:n]
        dists = torch.cdist(subset, subset)
        diag_mask = torch.eye(n, dtype=torch.bool, device=dists.device)
        dists = dists + diag_mask.float() * 1e10
        nn_dist = dists.min(dim=1).values
        k = max(1, n // 10)
        worst_gaps = torch.topk(nn_dist, k).values
        loss_uniform = worst_gaps.mean()
        loss = loss + loss_uniform * 0.05

        # Loss 3: Opacity encouragement (push toward visible)
        opacity_sigmoid = torch.sigmoid(scene.opacities)
        loss_opacity = (1.0 - opacity_sigmoid).mean()
        loss = loss + loss_opacity * 0.05

        loss.backward()
        optimizer.step()

        # Clamp colors
        with torch.no_grad():
            scene.colors.data.clamp_(0, 1)

        # Periodic densification
        if (i + 1) % 25 == 0:
            densify_step(scene, config)
            optimizer = torch.optim.Adam([scene.positions, scene.colors, scene.opacities], lr=lr)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_iterations}] loss={loss.item():.5f} N={scene.n_gaussians:,}")

    return scene.to_tensors()


# ---------------------------------------------------------------------------
# External: Multi-view consistency
# ---------------------------------------------------------------------------

def refine_multiview(tensors: dict[str, torch.Tensor], n_iterations: int = 100,
                     n_views: int = 16, lr: float = 1e-4) -> dict[str, torch.Tensor]:
    """Refine using multi-view consistency (no external model)."""
    scene = GaussianScene.from_tensors(tensors)
    optimizer = torch.optim.Adam([scene.positions, scene.colors, scene.opacities], lr=lr)

    config = DensifyConfig(
        grad_threshold=0.0005,
        max_gaussians=min(scene.n_gaussians * 2, 100000),
        opacity_prune_threshold=0.01,
    )

    print(f"Multi-view refinement: {scene.n_gaussians:,} Gaussians, {n_iterations} iterations...")

    for i in range(n_iterations):
        optimizer.zero_grad()

        # Uniformity loss
        n = min(scene.n_gaussians, 2000)
        subset = scene.positions[:n]
        dists = torch.cdist(subset, subset)
        diag_mask = torch.eye(n, dtype=torch.bool)
        dists = dists + diag_mask.float() * 1e10
        nn_dist = dists.min(dim=1).values
        loss = nn_dist.mean()

        # Local smoothness
        _, nn_idx = dists.topk(6, dim=1, largest=False)
        neighbors = subset[nn_idx]
        local_var = neighbors.var(dim=1).mean()
        loss = loss + local_var * 0.1

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            scene.colors.data.clamp_(0, 1)

        if (i + 1) % 20 == 0:
            densify_step(scene, config)
            optimizer = torch.optim.Adam([scene.positions, scene.colors, scene.opacities], lr=lr)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_iterations}] loss={loss.item():.5f} N={scene.n_gaussians:,}")

    return scene.to_tensors()


# ---------------------------------------------------------------------------
# External: SDS (Stable Diffusion)
# ---------------------------------------------------------------------------

def refine_sds(tensors: dict[str, torch.Tensor], prompt: str,
               n_iterations: int = 100) -> dict[str, torch.Tensor]:
    """Refine using Score Distillation Sampling (requires diffusers)."""
    try:
        from diffusers import StableDiffusionPipeline  # noqa: F401
    except ImportError:
        print("SDS mode requires: pip install diffusers transformers accelerate")
        print("Falling back to multiview mode.")
        return refine_multiview(tensors, n_iterations)

    print(f"SDS refinement with prompt: '{prompt}'")
    print("Note: full SDS pipeline requires differentiable image renderer.")
    print("Using multiview as approximation for now.")
    return refine_multiview(tensors, n_iterations)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tensors_to_tree(tensors: dict[str, torch.Tensor], name: str = "refined_scene") -> CompositionNode:
    """Convert flat tensors back into a single-node composition tree."""
    gaussians = []
    n = tensors["means"].shape[0]
    for i in range(n):
        gaussians.append(GaussianParams(
            position=tensors["means"][i].tolist(),
            scale=tensors["scales_log"][i].tolist(),
            opacity=tensors["opacities"][i].item(),
            color=tensors["colors"][i].tolist(),
            rotation=tensors["rotations"][i].tolist(),
        ))
    return CompositionNode(name=name, gaussians=gaussians)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Refine scene appearance (Phase C)")
    parser.add_argument("--input", required=True, help="Input scene JSON (Phase B output)")
    parser.add_argument("--output", required=True, help="Output refined scene JSON")
    parser.add_argument("--mode", choices=["sgs", "multiview", "sds"], default="sgs",
                        help="Refinement mode (default: sgs)")
    parser.add_argument("--templates", default="data/objaverse_gs",
                        help="Template library directory (for sgs mode)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of refinement iterations (default 100)")
    parser.add_argument("--n-views", type=int, default=16,
                        help="Number of viewpoints for multiview mode (default 16)")
    parser.add_argument("--prompt", default=None,
                        help="Text prompt for SDS mode")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default 5e-4)")
    args = parser.parse_args()

    tree = load_tree(args.input)
    tensors = tree_to_tensors(tree)
    n_before = tensors["means"].shape[0]
    print(f"Input: {n_before:,} Gaussians")
    print(f"Mode: {args.mode}")

    start = time.time()

    if args.mode == "sgs":
        result = refine_sgs(tensors, Path(args.templates), args.iterations, args.lr)
    elif args.mode == "multiview":
        result = refine_multiview(tensors, args.iterations, args.n_views, args.lr)
    elif args.mode == "sds":
        if not args.prompt:
            print("Error: --prompt required for SDS mode")
            sys.exit(1)
        result = refine_sds(tensors, args.prompt, args.iterations)

    elapsed = time.time() - start
    n_after = result["means"].shape[0]

    output_tree = tensors_to_tree(result)
    save_tree(output_tree, args.output)

    print(f"\nRefined: {n_before:,} -> {n_after:,} Gaussians")
    print(f"Wall-clock: {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
