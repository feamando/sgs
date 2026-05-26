"""
Compare two refined scenes (SGS-native vs. external) quantitatively.

Outputs a markdown report with metrics for A/B comparison.

Usage:
    python scripts/compare_scenes.py --sgs output/raum14_phase_c_sgs.json --external output/raum14_phase_c_ext.json --report output/refinement_comparison.md
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors


def compute_metrics(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute quality metrics for a scene."""
    means = tensors["means"]
    opacities = tensors["opacities"]
    colors = tensors["colors"]
    n = means.shape[0]

    # Bounding box
    bb_min = means.min(dim=0).values
    bb_max = means.max(dim=0).values
    bb_extent = (bb_max - bb_min).tolist()
    bb_volume = float((bb_max - bb_min).prod())

    # Surface uniformity: average nearest-neighbor distance
    sample_n = min(n, 3000)
    subset = means[:sample_n]
    dists = torch.cdist(subset, subset)
    dists.fill_diagonal_(float("inf"))
    nn_dists = dists.min(dim=1).values
    avg_nn_dist = nn_dists.mean().item()
    std_nn_dist = nn_dists.std().item()

    # Opacity statistics
    opacity_sigmoid = torch.sigmoid(opacities)
    avg_opacity = opacity_sigmoid.mean().item()
    low_opacity_frac = (opacity_sigmoid < 0.1).float().mean().item()

    # Color variance (higher = more diverse appearance)
    color_var = colors.var(dim=0).mean().item()

    # Density: Gaussians per unit volume
    density = n / max(bb_volume, 1e-10)

    return {
        "n_gaussians": n,
        "bb_extent_x": bb_extent[0],
        "bb_extent_y": bb_extent[1],
        "bb_extent_z": bb_extent[2],
        "bb_volume": bb_volume,
        "avg_nn_distance": avg_nn_dist,
        "std_nn_distance": std_nn_dist,
        "uniformity_ratio": std_nn_dist / max(avg_nn_dist, 1e-10),
        "avg_opacity": avg_opacity,
        "low_opacity_fraction": low_opacity_frac,
        "color_variance": color_var,
        "density_per_unit_vol": density,
    }


def generate_report(metrics_sgs: dict, metrics_ext: dict, output_path: Path):
    """Generate a markdown comparison report."""
    lines = [
        "# Refinement Comparison: SGS-native vs. External",
        "",
        f"Generated: {Path(output_path).name}",
        "",
        "## Metrics",
        "",
        "| Metric | SGS-native | External | Winner |",
        "|--------|-----------|----------|--------|",
    ]

    comparisons = {
        "n_gaussians": ("Gaussian count", "higher"),
        "avg_nn_distance": ("Avg NN distance", "lower"),
        "uniformity_ratio": ("Uniformity (std/mean NN)", "lower"),
        "avg_opacity": ("Avg opacity", "higher"),
        "low_opacity_fraction": ("Low-opacity fraction", "lower"),
        "color_variance": ("Color variance", "higher"),
        "density_per_unit_vol": ("Density (G/vol)", "higher"),
    }

    for key, (label, better) in comparisons.items():
        v_sgs = metrics_sgs[key]
        v_ext = metrics_ext[key]

        if better == "higher":
            winner = "SGS" if v_sgs > v_ext else ("External" if v_ext > v_sgs else "Tie")
        else:
            winner = "SGS" if v_sgs < v_ext else ("External" if v_ext < v_sgs else "Tie")

        lines.append(f"| {label} | {v_sgs:.4f} | {v_ext:.4f} | {winner} |")

    lines.extend([
        "",
        "## Bounding Box",
        "",
        f"- SGS: [{metrics_sgs['bb_extent_x']:.2f}, {metrics_sgs['bb_extent_y']:.2f}, {metrics_sgs['bb_extent_z']:.2f}] (vol={metrics_sgs['bb_volume']:.2f})",
        f"- Ext: [{metrics_ext['bb_extent_x']:.2f}, {metrics_ext['bb_extent_y']:.2f}, {metrics_ext['bb_extent_z']:.2f}] (vol={metrics_ext['bb_volume']:.2f})",
        "",
        "## Interpretation",
        "",
        "- **Avg NN distance**: lower = denser packing, more solid surfaces",
        "- **Uniformity ratio**: lower = more even distribution (fewer gaps/clusters)",
        "- **Low-opacity fraction**: lower = fewer invisible/useless Gaussians",
        "- **Color variance**: higher = more visual diversity (not all same color)",
        "- **Density**: higher = more detail per unit of space",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Compare two refined scenes")
    parser.add_argument("--sgs", required=True, help="SGS-native refined scene JSON")
    parser.add_argument("--external", required=True, help="Externally refined scene JSON")
    parser.add_argument("--report", required=True, help="Output markdown report path")
    args = parser.parse_args()

    print("Loading scenes...")
    tree_sgs = load_tree(args.sgs)
    tree_ext = load_tree(args.external)

    tensors_sgs = tree_to_tensors(tree_sgs)
    tensors_ext = tree_to_tensors(tree_ext)

    print(f"SGS: {tensors_sgs['means'].shape[0]:,} Gaussians")
    print(f"Ext: {tensors_ext['means'].shape[0]:,} Gaussians")

    print("Computing metrics...")
    metrics_sgs = compute_metrics(tensors_sgs)
    metrics_ext = compute_metrics(tensors_ext)

    report_path = Path(args.report)
    generate_report(metrics_sgs, metrics_ext, report_path)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n--- Quick Summary ---")
    for key in ["avg_nn_distance", "uniformity_ratio", "avg_opacity", "density_per_unit_vol"]:
        print(f"  {key}: SGS={metrics_sgs[key]:.4f}  Ext={metrics_ext[key]:.4f}")


if __name__ == "__main__":
    main()
