"""
Raum 1.7 Stage 1: differentiable render + SDS reachability probe.

The question this answers (NOT a model train): does a Score Distillation
Sampling gradient, backpropagated through a differentiable Gaussian renderer,
measurably improve a FIXED scene's appearance against a text prompt? If yes,
1.7 (learned geometry) is viable. If the gradient is too noisy/slow on a ~50K
splat scene, we learn that here, cheaply, before touching the decomposer.

Pipeline:
  load scene -> Gaussians (means/scales/colors as leaf params with grad)
  for each step:
    render from a random orbit camera (differentiable: gsplat on CUDA, or the
      CPU alpha-composite fallback in render_3d.py)
    SDS loss: add noise to the latent, ask Stable Diffusion to predict it given
      the prompt, the (eps_pred - eps) residual is the SDS gradient on the image
    backprop -> step the Gaussian params
  save the refined scene

Runs three ways:
  --selftest        : tiny scene, CPU renderer, NO diffusion -- proves the
                      render+optimize loop is differentiable end to end (works
                      on this Mac; no gsplat/CUDA/SD needed).
  (default)         : full SDS on the 4090 (gsplat + Stable Diffusion).
  --photometric     : optimize toward a fixed target image instead of SDS (a
                      cheap sanity check that gradients flow to the splats).

Usage:
  python scripts/sds_refine.py --selftest
  python scripts/sds_refine.py --scene output/castle_06.json \
    --prompt "a stone castle on a green hill" --iters 200 --out output/castle_sds.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import CompositionNode, tree_to_tensors, save_tree
from src.raum.render_3d import render_gaussians
from src.raum import cameras as camlib


# ── scene <-> optimizable params ──────────────────────────────────────

def load_scene_tensors(path: str, device):
    tree = CompositionNode.from_dict(json.load(open(path)))
    t = tree_to_tensors(tree)
    return {k: v.to(device) for k, v in t.items()}, tree


def make_params(tensors):
    """Positions, log-scales, and colors become leaf params with grad.
    Opacities/rotations stay fixed (Stage 1 optimizes appearance + placement)."""
    means = tensors["means"].clone().requires_grad_(True)
    scales = tensors["scales_log"].clone().requires_grad_(True)
    colors = tensors["colors"].clone().requires_grad_(True)
    opacities = tensors["opacities"].clone()  # fixed
    return means, scales, colors, opacities


# ── camera sampling ───────────────────────────────────────────────────

def _look_at_posZ(eye, target, up):
    """World-to-camera with +Z FORWARD (points in front have positive depth),
    matching render_3d's depth convention. cameras.look_at uses OpenGL -Z
    forward, which makes every point's depth negative -> clamped -> garbage.
    """
    forward = target - eye
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    new_up = torch.linalg.cross(right, forward)
    R = torch.stack([right, new_up, forward], dim=0)  # +forward (not -forward)
    t = -R @ eye
    W = torch.eye(4, device=eye.device)
    W[:3, :3] = R
    W[:3, 3] = t
    return W


def random_orbit_camera(tensors, device, img=128, rng=None):
    """A camera orbiting the scene's bounding sphere, looking at its center."""
    import math
    P = tensors["means"]
    center = P.mean(0)
    radius = (P - center).norm(dim=1).max().item()
    a = (rng if rng is not None else 0.0)
    az = a * 2 * math.pi
    el = 0.35 + 0.25 * math.sin(a * 3.1)
    dist = radius * 2.6
    eye = center + torch.tensor([
        dist * math.cos(el) * math.cos(az),
        dist * math.sin(el),
        dist * math.cos(el) * math.sin(az),
    ], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    W = _look_at_posZ(eye, center, up).to(device)
    K = camlib.make_intrinsic(50.0, img, img).to(device)
    return W, K, img


# ── self test: render+optimize loop is differentiable (CPU, no SD) ────

def selftest(device):
    """Build a trivial 3-Gaussian scene, render it, and optimize the splats to
    match a shifted copy. Proves gradients flow render -> params. No SD/gsplat."""
    print("[selftest] differentiable render+optimize loop (CPU ok, no SD)")
    torch.manual_seed(0)
    N = 40
    means = (torch.rand(N, 3, device=device) - 0.5).requires_grad_(True)
    scales = torch.full((N, 3), -2.5, device=device).requires_grad_(True)
    colors = torch.rand(N, 3, device=device).requires_grad_(True)
    opac = torch.full((N,), 3.0, device=device)
    # target: a fixed render we try to match (shift colors toward red)
    with torch.no_grad():
        tgt_colors = colors.clone(); tgt_colors[:, 0] = 1.0; tgt_colors[:, 1:] = 0.1
    W, K, img = random_orbit_camera({"means": means.detach()}, device, img=64, rng=0.2)
    opt = torch.optim.Adam([means, scales, colors], lr=0.05)
    target = render_gaussians(means.detach(), scales.detach(), opac, tgt_colors,
                              W, K, img, img, backend="simple").detach()
    losses = []
    for i in range(40):
        opt.zero_grad()
        rgb = render_gaussians(means, scales, opac, colors, W, K, img, img, backend="simple")
        loss = F.mse_loss(rgb, target)
        loss.backward()
        # grad must reach the params
        assert colors.grad is not None and colors.grad.abs().sum() > 0, "no grad to colors!"
        opt.step()
        losses.append(loss.item())
    print(f"[selftest] loss {losses[0]:.4f} -> {losses[-1]:.4f}  "
          f"({'DECREASING, grads flow' if losses[-1] < losses[0] else 'NOT decreasing -- bug'})")
    return losses[-1] < losses[0]


# ── SDS loss (4090 path) ──────────────────────────────────────────────

class SDSGuidance:
    """Stable Diffusion score-distillation guidance on rendered images."""
    def __init__(self, prompt: str, device, model="runwayml/stable-diffusion-v1-5",
                 guidance_scale=100.0):
        from diffusers import StableDiffusionPipeline
        print(f"[sds] loading {model} ...")
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.scheduler = pipe.scheduler
        self.device = device
        self.guidance_scale = guidance_scale
        self.alphas = self.scheduler.alphas_cumprod.to(device)
        for m in (self.unet, self.vae, self.text_encoder):
            m.requires_grad_(False)
        # text embeddings (cond + uncond)
        self.text_emb = self._embed([prompt])
        self.uncond_emb = self._embed([""])

    def _embed(self, prompts):
        tok = self.tokenizer(prompts, padding="max_length",
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.text_encoder(tok.input_ids.to(self.device))[0]

    def loss(self, image, rng_t):
        """image: [3,H,W] in [0,1]. Returns a scalar whose grad is the SDS push."""
        # to latent: VAE wants [B,3,512,512] in [-1,1]
        img = F.interpolate(image.unsqueeze(0), size=(512, 512), mode="bilinear",
                            align_corners=False)
        img = (img * 2 - 1).half()
        latents = self.vae.encode(img).latent_dist.sample() * 0.18215
        t = int(20 + rng_t * (980 - 20))
        t = torch.tensor([t], device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = self.scheduler.add_noise(latents, noise, t)
        emb = torch.cat([self.uncond_emb, self.text_emb])
        with torch.no_grad():
            noise_pred = self.unet(torch.cat([noisy] * 2), t, encoder_hidden_states=emb).sample
        nu, nc = noise_pred.chunk(2)
        noise_pred = nu + self.guidance_scale * (nc - nu)
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # SDS: define a surrogate loss whose grad equals `grad`
        return (latents * grad.detach()).sum()


def run_sds(args, device):
    tensors, tree = load_scene_tensors(args.scene, device)
    means, scales, colors, opac = make_params(tensors)
    n = means.shape[0]
    print(f"[sds] scene {args.scene}: {n} gaussians, prompt='{args.prompt}'")

    guidance = None
    if not args.photometric:
        guidance = SDSGuidance(args.prompt, device, guidance_scale=args.guidance)

    target = None
    if args.photometric:
        # cheap gradient sanity: perturb the colors, then optimize back toward
        # the ORIGINAL per-camera renders. Loss should fall toward 0 -> grads
        # flow to the splats on the real scene. (Fixed camera per the loop.)
        target = {"means": means.detach().clone(), "scales": scales.detach().clone(),
                  "colors": colors.detach().clone()}
        with torch.no_grad():
            colors.add_(torch.randn_like(colors) * 0.2).clamp_(0, 1)

    opt = torch.optim.Adam([
        {"params": [means], "lr": args.lr * 0.5},
        {"params": [scales], "lr": args.lr * 0.5},
        {"params": [colors], "lr": args.lr},
    ])
    for i in range(args.iters):
        opt.zero_grad()
        frac = (i % 16) / 16.0
        W, K, img = random_orbit_camera(tensors, device, img=args.img, rng=frac)
        rgb = render_gaussians(means, scales, opac, colors, W, K, img, img)
        if args.photometric:
            # render the same camera from the original (target) params
            with torch.no_grad():
                tgt = render_gaussians(target["means"], target["scales"], opac,
                                       target["colors"], W, K, img, img)
            loss = F.mse_loss(rgb, tgt)
        else:
            loss = guidance.loss(rgb, rng_t=frac)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([means, scales, colors], 1.0)
        opt.step()
        with torch.no_grad():
            colors.clamp_(0, 1)
        if i % 10 == 0 or i == args.iters - 1:
            print(f"  [{i:4d}/{args.iters}] loss={loss.item():.5f}")

    # write back the refined gaussians as a flat scene
    out_tree = CompositionNode(name="scene")
    from src.raum.decomposition import GaussianParams
    leaf = CompositionNode(name="refined")
    for j in range(n):
        leaf.gaussians.append(GaussianParams(
            position=means[j].detach().tolist(),
            scale=scales[j].detach().tolist(),
            opacity=float(opac[j].item()),
            color=colors[j].detach().clamp(0, 1).tolist(),
            rotation=tensors["rotations"][j].tolist(),
        ))
    out_tree.children.append(leaf)
    save_tree(out_tree, args.out)
    print(f"[sds] saved refined scene -> {args.out}")


def main():
    p = argparse.ArgumentParser(description="Raum 1.7 Stage 1: SDS reachability probe")
    p.add_argument("--selftest", action="store_true",
                   help="CPU differentiability check, no SD/gsplat")
    p.add_argument("--photometric", action="store_true",
                   help="optimize toward a fixed render (gradient sanity, no SD)")
    p.add_argument("--scene", default="output/castle_06.json")
    p.add_argument("--prompt", default="a stone castle on a green hill")
    p.add_argument("--out", default="output/castle_sds.json")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--guidance", type=float, default=100.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    from src.raum.render_3d import check_backend
    check_backend()

    if args.selftest:
        ok = selftest(device)
        sys.exit(0 if ok else 1)
    run_sds(args, device)


if __name__ == "__main__":
    main()
