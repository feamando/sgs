"""
Training script for Raum PoC-C: Shared-Equation Bridge.

Usage:
    python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt
    python scripts/train_raum_bridge.py --glove data/glove.6B.300d.txt --backend gsplat

Designed for RTX 4090 (24GB VRAM). Falls back to CPU with simple renderer.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_glove
from src.gaussian import SemanticGaussianVocab
from src.raum.vocab import ALL_SCENE_WORDS, OBJECTS, COLORS, RELATIONS
from src.raum.data import generate_comp_gen_split, RaumDataset, collate_raum
from src.raum.bridge import RaumBridge, compute_bridge_loss
from src.raum.templates import build_template_library
from src.raum.cameras import orbit_cameras
from src.raum.render_3d import render_gaussians, check_backend


def parse_args():
    p = argparse.ArgumentParser(description="Train Raum PoC-C")
    p.add_argument("--glove", required=True, help="Path to glove.6B.300d.txt")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-views", type=int, default=4)
    p.add_argument("--img-size", type=int, default=64, help="Render resolution (small for speed)")
    p.add_argument("--K", type=int, default=32, help="Gaussians per word")
    p.add_argument("--d-s", type=int, default=64, help="Splatting space dim")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--save-dir", default="checkpoints/raum_c")
    p.add_argument("--backend", default="auto", choices=["auto", "gsplat", "simple"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def render_gt_scene(objects_gt, template_lib, cameras, img_size, backend):
    """Render ground truth scene from object list."""
    # Collect all Gaussians
    all_means = []
    all_scales = []
    all_opacities = []
    all_colors = []

    for obj in objects_gt:
        tname = list(OBJECTS.keys())[obj.obj_type]
        template = template_lib.get(tname)
        if template is None:
            continue
        N = template.n_gaussians
        pos = torch.tensor(obj.position, dtype=torch.float32)
        means = template.means * obj.scale + pos.unsqueeze(0)
        scales = template.scales + math.log(max(obj.scale, 0.1))
        color = torch.tensor(obj.color, dtype=torch.float32).unsqueeze(0).expand(N, 3)

        all_means.append(means)
        all_scales.append(scales)
        all_opacities.append(template.opacities)
        all_colors.append(color)

    if not all_means:
        # Empty scene → black images
        return [torch.zeros(3, img_size, img_size) for _ in cameras]

    means = torch.cat(all_means)
    scales = torch.cat(all_scales)
    opacities = torch.cat(all_opacities)
    colors = torch.cat(all_colors)

    images = []
    for cam in cameras:
        img = render_gaussians(
            means, scales, opacities, colors,
            cam.world_to_cam, cam.K,
            cam.width, cam.height,
            backend=backend,
        )
        images.append(img.detach())
    return images


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    backend = check_backend() if args.backend == "auto" else args.backend
    print(f"Rendering backend: {backend}")

    # ── Load GloVe ──
    print("\n=== Loading GloVe ===")
    word2idx, vectors, freqs, words = load_glove(args.glove, vocab_size=50000)
    glove_matrix = torch.from_numpy(vectors).float()
    d_f = glove_matrix.shape[1]

    # Build vocab for SGS-style Gaussian params
    vocab = SemanticGaussianVocab(len(words), d_s=args.d_s, d_f=d_f)
    vocab.init_from_glove(vectors, freqs)
    vocab.to(device)
    vocab.eval()

    # ── Data ──
    print("\n=== Generating data ===")
    train_scenes, val_scenes, test_scenes = generate_comp_gen_split(
        args.n_train, args.n_val, args.n_test, seed=args.seed,
    )
    train_ds = RaumDataset(train_scenes, word2idx)
    val_ds = RaumDataset(val_scenes, word2idx)

    from src.raum.data import collate_raum
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_raum, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_raum,
    )

    # ── Templates + cameras ──
    template_lib = build_template_library(n_gaussians=100)
    cameras = orbit_cameras(
        n_views=args.n_views, img_size=args.img_size,
        elevation_deg=30.0, radius=4.0,
    )

    # ── Model ──
    print("\n=== Creating model ===")
    model = RaumBridge(d_s=args.d_s, d_f=d_f, K=args.K).to(device)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )

    # ── Train ──
    print(f"\n=== Training ({args.epochs} epochs) ===")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_psnr = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            mask = batch["mask"].to(device)

            # Get semantic Gaussian params (frozen)
            with torch.no_grad():
                mu_s, log_var, alpha_s, features = vocab.get_params(token_ids)

            # Forward: semantic → spatial Gaussians
            scene = model(mu_s, features, mask)

            # Render each sample from each view, compare to GT
            B = token_ids.shape[0]
            total_loss = torch.tensor(0.0, device=device)
            total_psnr = 0.0

            for b in range(B):
                # Get GT objects for this sample
                n_obj = batch["n_objects"][b].item()
                gt_objects = []
                for oi in range(n_obj):
                    from src.raum.data import ObjectGT
                    gt_objects.append(ObjectGT(
                        obj_type=batch["object_types"][b, oi].item(),
                        color=batch["object_colors"][b, oi].tolist(),
                        scale=batch["object_scales"][b, oi].item(),
                        position=batch["object_positions"][b, oi].tolist(),
                    ))

                # Render GT
                gt_images = render_gt_scene(
                    gt_objects, template_lib, cameras, args.img_size, backend,
                )

                # Render predicted scene for this sample
                for v, cam in enumerate(cameras):
                    rendered = render_gaussians(
                        scene["means"][b], scene["scales"][b],
                        scene["opacities"][b], scene["colors"][b],
                        cam.world_to_cam.to(device), cam.K.to(device),
                        cam.width, cam.height, backend=backend,
                    )
                    gt_img = gt_images[v].to(device)
                    loss_v, metrics_v = compute_bridge_loss(rendered, gt_img, {
                        k: v[b:b+1] for k, v in scene.items()
                    })
                    total_loss = total_loss + loss_v
                    total_psnr += metrics_v["psnr"]

            total_loss = total_loss / (B * args.n_views)
            avg_psnr = total_psnr / (B * args.n_views)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += total_loss.item()
            n_batches += 1

            if global_step % args.log_interval == 0:
                print(f"  epoch {epoch+1} step {global_step:>5d} | "
                      f"loss {total_loss.item():.4f} psnr {avg_psnr:.1f} dB")

            # ── Eval ──
            if global_step % args.eval_interval == 0:
                val_psnr = eval_psnr(
                    model, vocab, val_loader, template_lib, cameras,
                    args.img_size, backend, device, max_batches=10,
                )
                print(f"  >>> val psnr {val_psnr:.1f} dB")
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    torch.save(model.state_dict(), save_dir / "best.pt")
                    print(f"  ** New best → saved")
                model.train()

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1} ({elapsed:.1f}s) | avg_loss={avg_loss:.4f}")

    # ── Final ──
    print(f"\nDone. Best val PSNR: {best_val_psnr:.1f} dB")
    print(f"Checkpoint: {save_dir / 'best.pt'}")
    print(f"\nRun analysis: python scripts/analyze_raum_bridge.py --checkpoint {save_dir / 'best.pt'} --glove {args.glove}")


@torch.no_grad()
def eval_psnr(model, vocab, loader, template_lib, cameras, img_size, backend, device, max_batches=10):
    model.eval()
    total_psnr = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        token_ids = batch["token_ids"].to(device)
        mask = batch["mask"].to(device)
        mu_s, _, _, features = vocab.get_params(token_ids)
        scene = model(mu_s, features, mask)

        B = token_ids.shape[0]
        for b in range(B):
            n_obj = batch["n_objects"][b].item()
            gt_objects = []
            for oi in range(n_obj):
                from src.raum.data import ObjectGT
                gt_objects.append(ObjectGT(
                    obj_type=batch["object_types"][b, oi].item(),
                    color=batch["object_colors"][b, oi].tolist(),
                    scale=batch["object_scales"][b, oi].item(),
                    position=batch["object_positions"][b, oi].tolist(),
                ))
            gt_images = render_gt_scene(gt_objects, template_lib, cameras, img_size, backend)
            for v, cam in enumerate(cameras):
                rendered = render_gaussians(
                    scene["means"][b], scene["scales"][b],
                    scene["opacities"][b], scene["colors"][b],
                    cam.world_to_cam.to(device), cam.K.to(device),
                    cam.width, cam.height, backend=backend,
                )
                mse = F.mse_loss(rendered, gt_images[v].to(device))
                psnr = -10.0 * torch.log10(mse.clamp(min=1e-8))
                total_psnr += psnr.item()
                n += 1
    return total_psnr / max(n, 1)


if __name__ == "__main__":
    main()
