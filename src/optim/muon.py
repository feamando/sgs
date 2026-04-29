"""Muon optimizer (single-GPU, PyTorch-only).

Muon replaces AdamW on 2D weight matrices. It tracks momentum, then
orthogonalises the momentum buffer via a Newton-Schulz iteration before
applying the update. The orthogonalisation converts the momentum matrix
into its closest semi-orthogonal matrix (all singular values ≈ 1),
which dramatically improves second-order behaviour relative to AdamW
at a tiny constant overhead.

Non-2D params (embeddings, norms, 1D scalars, biases) are typed out of
Muon's shape contract — use the `MuonWithAuxAdam` wrapper in this file
to route 2D params to Muon and everything else to AdamW.

Reference: Keller Jordan's original Muon, and the Moonshot-style
single-GPU variant. The Newton-Schulz coefficients (a, b, c) and the
T=5 iteration count match the canonical values.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration for the matrix-sign / orthogonalisation.

    Given a matrix G, returns an approximation of `U @ V.T` where
    `G = U @ S @ V.T` is the SVD — i.e. G with all singular values
    clamped to 1. Runs in bf16 for speed; output is cast back to G's
    dtype.
    """
    assert G.ndim == 2, "Newton-Schulz requires a 2D tensor"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    # Normalise so the spectral norm ≤ 1 (prerequisite for convergence).
    X = X / (X.norm() + 1e-7)
    # Work on the shorter side.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer for 2D matrix parameters.

    Args:
        params: iterable of 2D parameters (other shapes will raise).
        lr: learning rate (much higher than AdamW; 0.02 is a good default
            because the orthogonalised update has bounded spectral norm).
        momentum: Nesterov-ish momentum coefficient (0.95 is canonical).
        weight_decay: decoupled weight decay.
        ns_steps: Newton-Schulz iterations per step (5 is canonical).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D params; got shape {tuple(p.shape)}"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Nesterov-style blend: update buf first, then take a
                # weighted average (equivalent to look-ahead gradient).
                buf.mul_(momentum).add_(g)
                g_ns = g.add(buf, alpha=momentum)

                # Orthogonalise.
                update = _zeropower_via_newtonschulz5(g_ns, steps=ns_steps)

                # Scale so the effective update size is shape-invariant.
                # Canonical Muon scales by max(1, out/in)^0.5 — keeps
                # rectangular matrices on the same footing as square ones.
                out_dim, in_dim = p.shape
                scale = max(1.0, (out_dim / in_dim) ** 0.5)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr * scale)

        return loss


class MuonWithAuxAdam(Optimizer):
    """Holds a Muon for 2D params and an AdamW for the rest.

    Subclasses `torch.optim.Optimizer` so PyTorch's LR schedulers
    (`LinearLR`, `CosineAnnealingLR`, `SequentialLR`) accept it —
    they type-check `isinstance(optimizer, Optimizer)` in
    `LRScheduler.__init__`. 1.2.1's duck-typed wrapper failed that
    check and crashed at model-init.

    The LR scheduler sees one flat `param_groups` list:
    Muon's groups first, then AdamW's. Scaling all groups by the same
    factor (the warmup + cosine shape) is what we want — Muon's
    absolute LR is higher but the schedule shape is shared.
    """

    def __init__(
        self,
        params_2d,
        params_other,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_wd: float = 0.0,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_wd: float = 0.1,
        adam_no_decay_params: list | None = None,
    ):
        # Inner optimizers hold all real state and param groups.
        self.muon = Muon(
            params_2d,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_wd,
        )

        no_decay_set = (
            {id(p) for p in adam_no_decay_params} if adam_no_decay_params else set()
        )
        decay = [p for p in params_other if id(p) not in no_decay_set]
        no_decay = [p for p in params_other if id(p) in no_decay_set]
        self.adam = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": adam_wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=adam_lr,
            betas=adam_betas,
        )

        # Optimizer base-class init with no params of its own. We then
        # overwrite param_groups to expose both inner optimizers' groups
        # as one flat list. This is what LRScheduler iterates.
        super().__init__([{"params": []}], defaults={})
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step(closure)
        self.adam.step(closure)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])
        # Re-link param_groups to the loaded inner groups.
        self.param_groups = self.muon.param_groups + self.adam.param_groups
