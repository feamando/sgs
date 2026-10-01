# Training Acceleration for Semantic Gaussian Splatting

## Context

SGS uses a 100M-param causal transformer (Planck) as an encoder that outputs
Gaussian splat parameters. Previous acceleration attempts (Planck 1.2.1/1.2.2)
failed: Muon regressed +0.48 nats, compound SGS-native recipes OOM'd because
the forward activation `[B*H, L, k, d_f]` is ~8 GiB BF16 x 3 passes on a
24 GB RTX 4090 with Windows shared GPU memory.

This whitepaper analyzes why those failed and identifies viable alternatives.

## Why Previous Attempts Failed

### Architecture mismatch

Most modern optimizer innovations target dense matmul parameters:
- **Muon** orthogonalizes momentum for 2D weight matrices. SGS has a mixed
  parameter landscape: 2D attention/FFN weights + 1D Gaussian parameters
  (position, scale, opacity). Muon helped the 2D params but hurt the 1D
  rendering outputs.
- **Flash Attention** saves memory only in the attention layers. The bottleneck
  was the rendering passes, not attention.

### Memory ceiling

The RTX 4090 has 24 GB VRAM but Windows reserves ~2 GB for display. The
three-pass rendering (forward + loss + backward through Gaussian params)
materializes the full `[B*H, L, k, d_f]` tensor three times. At 100M params
with k=200k blobs, this exceeds available memory regardless of optimizer.

## Acceleration Taxonomy

### Architecture-agnostic (works for SGS)

| Technique | Mechanism | Expected gain | Risk |
|-----------|-----------|--------------|------|
| **Mixed precision (BF16/FP8)** | Lower precision reduces memory bandwidth and storage. BF16 preserves dynamic range. FP8 needs per-tensor scaling. | 30-50% memory, 10-20% speed | FP8 on Gaussian params may lose geometric precision |
| **Gradient accumulation** | Sum gradients over N micro-batches, step once. Simulates larger batch at constant memory. | Linear memory reduction per micro-batch | Slower wall-clock per step |
| **Gradient checkpointing** | Don't store intermediate activations; recompute on backward pass. | 2-3x memory reduction | ~30% speed overhead |
| **Kernel fusion (Triton/CUDA)** | Merge sequential ops into one kernel, eliminating memory round-trips. | 2-4x memory, 10-30% speed | High engineering effort |
| **Progressive scheduling** | Start with fewer Gaussians/blobs, grow during training. | 20-40% faster convergence | May miss fine-grained features early |

### Architecture-specific (Transformer encoder only)

| Technique | Mechanism | Applies to |
|-----------|-----------|-----------|
| **Flash Attention** | Fused attention kernel, O(N) memory | Attention layers only |
| **Muon/LION** | Specialized momentum for 2D params | Weight matrices only, not GS output |
| **Model parallelism** | Shard layers across GPUs | Only useful at 1B+ params |

### 3DGS community techniques (applicable to rendering passes)

| Technique | Source | Mechanism | Applicability |
|-----------|--------|-----------|--------------|
| **Adaptive densification** | Kerbl et al. 2023 | Clone small/split large splats based on gradient magnitude | Maps to SGS blob management |
| **Fused rasterization kernel** | gsplat library | Single CUDA kernel for sort + render + backward | Directly applicable to SGS render pass |
| **Optimal transport merging** | MMGS (May 2026) | Merge redundant Gaussians via OT | Reduces blob count, lower memory |
| **Diagonal Hessian + trust regions** | 3DGS^2-TR (Feb 2026) | Per-parameter curvature via Hutchinson's trace estimator | Applicable to GS output head |
| **Resolution scheduling** | DashGaussian (Mar 2025) | Lower render resolution early, full resolution later | Applicable to any differentiable render |
| **Compact densification** | AdpSplit (May 2026) | Error-driven splitting only where needed | 9-22% parameter reduction |

## Recommended Strategy

### Phase 1: Memory relief (immediate, 1-2 days)

1. **Gradient checkpointing on rendering passes only.** Keep the transformer
   encoder normal (fast), checkpoint only the 3-pass render. Expected: fits
   in 24 GB that currently OOMs.

2. **FP8 on transformer encoder.** The encoder's attention/FFN layers are
   standard and tolerate FP8 well. Keep Gaussian output params in BF16 for
   geometric precision.

Combined: ~60% memory reduction. Should unblock training that previously OOM'd.

### Phase 2: Speed (1 week)

3. **Progressive Gaussian scheduling.** Start training with k=10k blobs,
   grow to 200k over the first 30% of training. Early iterations are cheap,
   fine-grained structure emerges later.

4. **Diagonal Hessian on GS output head.** Use Sophia-style per-parameter
   learning rates on the projection that outputs Gaussian params. Keep AdamW
   on the transformer body. Expected: 30-50% fewer iterations to convergence
   on geometric quality.

### Phase 3: Throughput (2-4 weeks, for Hertz)

5. **Fused CUDA render kernel (gsplat-style).** Write a custom kernel that
   does sort + render + backward in one pass. The gsplat library demonstrates
   4x memory reduction and 15% speed improvement. This is the highest-ceiling
   lever but requires CUDA expertise.

6. **Adaptive densification during training.** Instead of fixed k=200k blobs,
   let the model grow/prune blobs during training. Start sparse, densify where
   reconstruction error is high. Converges faster than fixed-count.

### Phase 4: Scale (Hertz 1.2+)

7. **Multi-GPU data parallelism.** At Hertz scale (1B params), standard DDP
   across 2-4 GPUs with the above optimizations.

## What NOT to retry

- Muon optimizer at any scale (confirmed regression on mixed-param landscape)
- Compound recipes that add memory overhead without first solving the render OOM
- Any technique that requires the full `[B*H, L, k, d_f]` tensor materialized

## Hardware Note

All above strategies assume the Windows shared-GPU-memory limitation persists.
On a dedicated Linux box with full 24 GB exclusive VRAM, Phase 1 alone may
suffice for Planck-scale (100M) training. Phase 3+ is primarily for Hertz.

## Literature

- Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (2023)
- gsplat library: 4x memory reduction via fused CUDA kernels
- FastGS (Nov 2025): 3.3-15x speedup via multi-view consistency densification
- DashGaussian (Mar 2025): 45% acceleration via resolution scheduling
- 3DGS^2-TR (Feb 2026): diagonal Hessian + trust regions, 50% fewer iterations
- MMGS (May 2026): optimal transport merging, 10x training compression
- AdpSplit (May 2026): error-driven adaptive splitting, 9-22% reduction
- Faster-GS (Feb 2026): numerical stability + gradient approximation, 5x speedup
