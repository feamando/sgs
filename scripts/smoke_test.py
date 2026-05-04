"""End-to-end smoke test. One command, PowerShell-safe.

Runs the two-minute verification flow from SETUP_202605.md §1.6: a
tiny SGS render through the semantic vocab + kernel + renderer, then
the Raum template library, scene generator, and compositional bridge
param count. Exits non-zero on any import or shape failure.
"""

import torch

from src.gaussian import SemanticGaussianVocab
from src.kernel import gaussian_kernel_diag
from src.raum.compositional import RaumCompositional
from src.raum.data import generate_dataset
from src.raum.templates import build_template_library
from src.rendering import render


def main() -> None:
    vocab = SemanticGaussianVocab(100, d_s=64, d_f=300)
    ids = torch.randint(0, 100, (2, 10))
    mu, lv, alpha, feat = vocab.get_params(ids)
    q = mu.mean(dim=1)
    K = gaussian_kernel_diag(q, mu, lv, torch.tensor(64.0))
    meaning, _ = render(feat, alpha, K)
    print(f"SGS render: meaning shape = {meaning.shape}")

    templates = build_template_library(n_gaussians=50)
    print(f"Templates: {list(templates.keys())}")

    scenes = generate_dataset(10)
    print(f'Generated {len(scenes)} scenes, first: "{scenes[0].sentence}"')

    model = RaumCompositional(d_f=300)
    print(f"RaumCompositional: {model.count_parameters():,} params")

    print("All OK")


if __name__ == "__main__":
    main()
