"""
Visual regression testing for Raum scenes.

Renders scenes from fixed viewpoints and compares against baseline
screenshots using SSIM. Fails if similarity drops below threshold.

Usage:
    # Generate baselines
    python scripts/visual_regression.py --generate-baselines

    # Run regression test
    python scripts/visual_regression.py --all --threshold 0.95
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import load_tree, tree_to_tensors


def render_scene_to_image(
    tensors: dict[str, torch.Tensor],
    width: int = 256,
    height: int = 256,
    camera_pos: tuple[float, float, float] = (3.0, 2.0, 3.0),
) -> np.ndarray:
    """
    Simple point-based rendering of Gaussians to an image.

    Returns [H, W, 3] uint8 numpy array.
    Projects Gaussian centers to screen space and draws circles.
    """
    means = tensors["means"].numpy()
    colors = tensors["colors"].numpy()
    opacities = tensors["opacities"].numpy()

    # Simple perspective projection
    cx, cy, cz = camera_pos
    cam = np.array([cx, cy, cz])

    # Look at origin
    forward = -cam / (np.linalg.norm(cam) + 1e-8)
    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    # Project points
    rel = means - cam
    z = rel @ forward
    x = rel @ right
    y = rel @ up

    # Perspective divide
    focal = 2.0
    valid = z > 0.1
    screen_x = np.where(valid, (x / z * focal + 1) * width / 2, -1)
    screen_y = np.where(valid, (1 - y / z * focal) * height / 2, -1)

    # Depth sort (back to front)
    order = np.argsort(-z)

    # Render
    image = np.zeros((height, width, 3), dtype=np.float32)
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacities))

    for idx in order:
        if not valid[idx]:
            continue
        sx, sy = int(screen_x[idx]), int(screen_y[idx])
        if 0 <= sx < width and 0 <= sy < height:
            alpha = opacity_sigmoid[idx]
            color = colors[idx]
            # Draw a small point (3x3)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = sx + dx, sy + dy
                    if 0 <= px < width and 0 <= py < height:
                        image[py, px] = image[py, px] * (1 - alpha) + color * alpha

    return (image * 255).clip(0, 255).astype(np.uint8)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity between two images (simplified SSIM)."""
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


# Standard camera positions for regression testing
CAMERAS = {
    "front": (0.0, 1.0, 4.0),
    "side": (4.0, 1.0, 0.0),
    "top": (0.0, 5.0, 0.1),
    "perspective": (3.0, 2.0, 3.0),
}


def main():
    parser = argparse.ArgumentParser(description="Visual regression tests")
    parser.add_argument("--scenes", default=None,
                        help="Comma-separated scene JSON paths")
    parser.add_argument("--baselines", default="tests/baselines",
                        help="Baselines directory")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="SSIM threshold for pass (default 0.95)")
    parser.add_argument("--generate-baselines", action="store_true",
                        help="Generate baseline images instead of comparing")
    parser.add_argument("--all", action="store_true",
                        help="Run all default scenes")
    args = parser.parse_args()

    baselines_dir = Path(args.baselines)

    if args.all:
        scenes_dir = Path("data/scenes")
        scene_paths = sorted(scenes_dir.glob("*.json"))
    elif args.scenes:
        scene_paths = [Path(s) for s in args.scenes.split(",")]
    else:
        print("Specify --scenes or --all")
        sys.exit(1)

    if not scene_paths:
        print("No scenes found")
        sys.exit(1)

    if args.generate_baselines:
        baselines_dir.mkdir(parents=True, exist_ok=True)
        for scene_path in scene_paths:
            tree = load_tree(str(scene_path))
            tensors = tree_to_tensors(tree)
            scene_name = scene_path.stem

            for cam_name, cam_pos in CAMERAS.items():
                img = render_scene_to_image(tensors, camera_pos=cam_pos)
                out_path = baselines_dir / f"{scene_name}_{cam_name}.npy"
                np.save(str(out_path), img)
                print(f"  Saved baseline: {out_path.name}")

        print(f"Baselines generated at {baselines_dir}")
        return

    # Compare against baselines
    n_pass = 0
    n_fail = 0
    failures = []

    for scene_path in scene_paths:
        tree = load_tree(str(scene_path))
        tensors = tree_to_tensors(tree)
        scene_name = scene_path.stem

        for cam_name, cam_pos in CAMERAS.items():
            baseline_path = baselines_dir / f"{scene_name}_{cam_name}.npy"
            if not baseline_path.exists():
                print(f"  SKIP {scene_name}/{cam_name} (no baseline)")
                continue

            baseline = np.load(str(baseline_path))
            current = render_scene_to_image(tensors, camera_pos=cam_pos)
            ssim = compute_ssim(current, baseline)

            if ssim >= args.threshold:
                n_pass += 1
            else:
                n_fail += 1
                failures.append(f"{scene_name}/{cam_name}: SSIM={ssim:.4f}")

    print(f"\nResults: {n_pass} passed, {n_fail} failed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("ALL PASSED")


if __name__ == "__main__":
    main()
