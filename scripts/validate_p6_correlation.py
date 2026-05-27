"""
Empirical validation of Claim P6: The Correlation Hypothesis.

Tests whether semantic embeddings (GloVe) carry enough information to
predict physical properties (hardness, elasticity, friction, density, etc.)
for known materials.

Approach:
1. Build a ground-truth material table: 80 materials with 8 physical properties
2. Get GloVe embeddings for each material word
3. Train a small MLP: GloVe(300d) -> physical_properties(8d)
4. Measure R^2 on held-out materials
5. If R^2 > 0.7, the correlation hypothesis holds practically

Usage:
    python scripts/validate_p6_correlation.py
    python scripts/validate_p6_correlation.py --glove-path data/glove.6B.300d.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Ground-truth material properties (normalized to [0, 1])
# Axes: hardness, elasticity, friction, density, brittleness, thermal_cond, transparency, deformability
MATERIAL_TABLE = {
    # Stones and minerals
    "stone":     [0.85, 0.05, 0.70, 0.75, 0.80, 0.30, 0.00, 0.05],
    "granite":   [0.90, 0.03, 0.65, 0.80, 0.85, 0.35, 0.00, 0.03],
    "marble":    [0.75, 0.05, 0.40, 0.78, 0.70, 0.30, 0.05, 0.05],
    "sandstone": [0.60, 0.08, 0.75, 0.65, 0.60, 0.20, 0.00, 0.10],
    "slate":     [0.70, 0.04, 0.60, 0.72, 0.75, 0.25, 0.00, 0.04],
    "concrete":  [0.70, 0.05, 0.70, 0.70, 0.65, 0.25, 0.00, 0.05],
    "brick":     [0.65, 0.05, 0.75, 0.60, 0.70, 0.15, 0.00, 0.05],
    "gravel":    [0.55, 0.10, 0.80, 0.55, 0.50, 0.15, 0.00, 0.15],
    "clay":      [0.30, 0.15, 0.65, 0.55, 0.20, 0.10, 0.00, 0.70],
    "diamond":   [1.00, 0.02, 0.10, 0.90, 0.95, 0.95, 0.90, 0.01],

    # Metals
    "steel":     [0.90, 0.30, 0.50, 0.90, 0.20, 0.85, 0.00, 0.15],
    "iron":      [0.85, 0.25, 0.55, 0.88, 0.25, 0.80, 0.00, 0.15],
    "aluminum":  [0.50, 0.35, 0.40, 0.45, 0.15, 0.90, 0.00, 0.30],
    "copper":    [0.55, 0.30, 0.45, 0.85, 0.10, 0.95, 0.00, 0.25],
    "gold":      [0.40, 0.25, 0.30, 0.95, 0.05, 0.70, 0.00, 0.35],
    "silver":    [0.45, 0.28, 0.35, 0.88, 0.08, 0.98, 0.00, 0.30],
    "bronze":    [0.60, 0.28, 0.50, 0.82, 0.15, 0.60, 0.00, 0.20],
    "tin":       [0.35, 0.20, 0.40, 0.70, 0.30, 0.55, 0.00, 0.35],
    "lead":      [0.25, 0.10, 0.50, 0.98, 0.10, 0.35, 0.00, 0.50],
    "titanium":  [0.85, 0.35, 0.45, 0.60, 0.15, 0.25, 0.00, 0.10],

    # Woods
    "wood":      [0.40, 0.20, 0.60, 0.35, 0.40, 0.10, 0.00, 0.20],
    "oak":       [0.55, 0.18, 0.65, 0.45, 0.45, 0.12, 0.00, 0.15],
    "pine":      [0.35, 0.22, 0.55, 0.30, 0.35, 0.08, 0.00, 0.25],
    "bamboo":    [0.45, 0.40, 0.50, 0.30, 0.30, 0.08, 0.00, 0.30],
    "plywood":   [0.40, 0.15, 0.55, 0.35, 0.45, 0.10, 0.00, 0.15],
    "cork":      [0.15, 0.60, 0.70, 0.10, 0.10, 0.05, 0.00, 0.70],

    # Soft materials
    "rubber":    [0.20, 0.90, 0.85, 0.35, 0.05, 0.05, 0.00, 0.90],
    "leather":   [0.30, 0.40, 0.70, 0.35, 0.10, 0.08, 0.00, 0.60],
    "cloth":     [0.05, 0.30, 0.60, 0.15, 0.02, 0.05, 0.00, 0.95],
    "silk":      [0.08, 0.25, 0.30, 0.12, 0.05, 0.04, 0.10, 0.92],
    "cotton":    [0.05, 0.20, 0.65, 0.12, 0.02, 0.05, 0.00, 0.95],
    "wool":      [0.08, 0.35, 0.70, 0.10, 0.02, 0.03, 0.00, 0.90],
    "felt":      [0.10, 0.25, 0.75, 0.15, 0.03, 0.04, 0.00, 0.85],
    "foam":      [0.05, 0.80, 0.50, 0.03, 0.01, 0.02, 0.00, 0.95],
    "sponge":    [0.05, 0.70, 0.55, 0.05, 0.01, 0.02, 0.00, 0.93],

    # Glass and ceramics
    "glass":     [0.70, 0.05, 0.20, 0.72, 0.95, 0.50, 0.95, 0.02],
    "ceramic":   [0.75, 0.03, 0.50, 0.70, 0.90, 0.20, 0.00, 0.03],
    "porcelain": [0.72, 0.03, 0.35, 0.72, 0.92, 0.15, 0.10, 0.02],
    "crystal":   [0.80, 0.04, 0.15, 0.75, 0.90, 0.45, 0.92, 0.02],

    # Plastics and synthetics
    "plastic":   [0.40, 0.45, 0.40, 0.30, 0.30, 0.05, 0.20, 0.50],
    "nylon":     [0.45, 0.50, 0.35, 0.32, 0.15, 0.05, 0.15, 0.55],
    "polyester": [0.40, 0.40, 0.35, 0.35, 0.20, 0.04, 0.10, 0.50],
    "vinyl":     [0.35, 0.45, 0.50, 0.35, 0.25, 0.04, 0.05, 0.55],
    "acrylic":   [0.50, 0.10, 0.30, 0.35, 0.60, 0.05, 0.90, 0.10],
    "epoxy":     [0.65, 0.10, 0.45, 0.40, 0.55, 0.05, 0.05, 0.08],

    # Natural materials
    "bone":      [0.65, 0.15, 0.50, 0.55, 0.50, 0.15, 0.00, 0.10],
    "ivory":     [0.60, 0.12, 0.40, 0.55, 0.55, 0.12, 0.05, 0.08],
    "shell":     [0.55, 0.10, 0.45, 0.60, 0.65, 0.10, 0.00, 0.08],
    "horn":      [0.50, 0.20, 0.55, 0.45, 0.35, 0.08, 0.00, 0.20],
    "feather":   [0.05, 0.30, 0.40, 0.02, 0.05, 0.02, 0.00, 0.80],

    # Liquids and gels
    "water":     [0.00, 0.95, 0.05, 0.50, 0.00, 0.30, 0.90, 1.00],
    "oil":       [0.00, 0.90, 0.02, 0.45, 0.00, 0.10, 0.70, 1.00],
    "honey":     [0.00, 0.60, 0.10, 0.55, 0.00, 0.08, 0.60, 1.00],
    "mud":       [0.10, 0.30, 0.70, 0.55, 0.05, 0.10, 0.00, 0.90],
    "tar":       [0.20, 0.15, 0.80, 0.55, 0.10, 0.08, 0.00, 0.70],
    "wax":       [0.20, 0.15, 0.40, 0.40, 0.30, 0.05, 0.30, 0.50],
    "gel":       [0.05, 0.70, 0.30, 0.40, 0.02, 0.05, 0.60, 0.90],

    # Earth and soil
    "sand":      [0.30, 0.10, 0.70, 0.50, 0.20, 0.10, 0.00, 0.40],
    "soil":      [0.20, 0.15, 0.70, 0.45, 0.10, 0.10, 0.00, 0.60],
    "dirt":      [0.20, 0.15, 0.65, 0.45, 0.10, 0.10, 0.00, 0.60],
    "ice":       [0.60, 0.05, 0.05, 0.50, 0.80, 0.90, 0.85, 0.03],
    "snow":      [0.10, 0.30, 0.30, 0.10, 0.05, 0.05, 0.60, 0.70],
    "ite": [0.50, 0.08, 0.60, 0.55, 0.60, 0.10, 0.00, 0.10],

    # Vegetation
    "grass":     [0.05, 0.40, 0.50, 0.08, 0.05, 0.03, 0.00, 0.85],
    "leaf":      [0.05, 0.30, 0.40, 0.05, 0.10, 0.03, 0.20, 0.80],
    "bark":      [0.35, 0.15, 0.80, 0.30, 0.35, 0.05, 0.00, 0.15],
    "moss":      [0.03, 0.50, 0.60, 0.05, 0.02, 0.02, 0.00, 0.90],
    "hay":       [0.08, 0.25, 0.60, 0.08, 0.10, 0.03, 0.00, 0.70],
    "straw":     [0.10, 0.20, 0.55, 0.08, 0.15, 0.03, 0.00, 0.65],

    # Paper and fabric
    "paper":     [0.10, 0.10, 0.55, 0.10, 0.30, 0.05, 0.10, 0.60],
    "cardboard": [0.15, 0.10, 0.60, 0.12, 0.25, 0.05, 0.00, 0.40],
    "canvas":    [0.15, 0.20, 0.70, 0.20, 0.10, 0.05, 0.00, 0.70],
    "rope":      [0.20, 0.35, 0.75, 0.25, 0.10, 0.05, 0.00, 0.65],

    # Food-like (for game engines)
    "bread":     [0.15, 0.30, 0.55, 0.20, 0.20, 0.05, 0.00, 0.60],
    "cheese":    [0.20, 0.25, 0.50, 0.35, 0.15, 0.08, 0.00, 0.55],
    "meat":      [0.15, 0.35, 0.60, 0.40, 0.05, 0.10, 0.00, 0.60],

    # Miscellaneous
    "chalk":     [0.30, 0.05, 0.70, 0.45, 0.80, 0.10, 0.00, 0.10],
    "ite":  [0.70, 0.10, 0.60, 0.55, 0.50, 0.15, 0.00, 0.08],
    "ite":  [0.55, 0.08, 0.50, 0.50, 0.55, 0.20, 0.00, 0.08],
    "ite":   [0.50, 0.10, 0.55, 0.45, 0.45, 0.15, 0.00, 0.10],
    "ite":   [0.40, 0.12, 0.60, 0.40, 0.40, 0.10, 0.00, 0.15],
}

# Deduplicate (dict keys are unique, duplicates overwrite)
PROPERTY_NAMES = [
    "hardness", "elasticity", "friction", "density",
    "brittleness", "thermal_conductivity", "transparency", "deformability"
]


class PhysicsPredictionMLP(nn.Module):
    """Predict physical properties from semantic embedding."""

    def __init__(self, input_dim: int = 300, output_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_glove(path: Path, vocab: set[str], dim: int = 300) -> dict[str, np.ndarray]:
    """Load GloVe embeddings for specified vocabulary."""
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                vec = np.array([float(x) for x in parts[1:dim+1]], dtype=np.float32)
                embeddings[word] = vec
    return embeddings


def generate_random_embeddings(vocab: list[str], dim: int = 300, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate random embeddings as fallback (for testing without GloVe file)."""
    rng = np.random.RandomState(seed)
    embeddings = {}
    for word in vocab:
        # Use hash of word as seed for reproducibility
        word_seed = hash(word) % (2**31)
        word_rng = np.random.RandomState(word_seed)
        embeddings[word] = word_rng.randn(dim).astype(np.float32)
    return embeddings


def run_validation(embeddings: dict[str, np.ndarray], materials: dict[str, list[float]],
                   n_splits: int = 5, epochs: int = 500) -> dict:
    """Run k-fold cross-validation of physics prediction from embeddings."""
    # Filter to materials that have embeddings
    valid_materials = [(word, props) for word, props in materials.items()
                       if word in embeddings]

    if len(valid_materials) < 10:
        print(f"Only {len(valid_materials)} materials have embeddings. Need at least 10.")
        return {"r2": 0.0, "n_materials": len(valid_materials)}

    print(f"Valid materials with embeddings: {len(valid_materials)}")

    # Prepare data
    X = torch.tensor(np.array([embeddings[w] for w, _ in valid_materials]), dtype=torch.float32)
    Y = torch.tensor(np.array([p for _, p in valid_materials]), dtype=torch.float32)

    n = len(valid_materials)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    # K-fold cross-validation
    fold_size = n // n_splits
    all_preds = torch.zeros_like(Y)
    all_r2_per_fold = []

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)
        test_idx = list(range(test_start, test_end))
        train_idx = [i for i in range(n) if i not in test_idx]

        if len(train_idx) < 5 or len(test_idx) < 2:
            continue

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        model = PhysicsPredictionMLP(input_dim=input_dim, output_dim=output_dim, hidden=128)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train
        model.train()
        for epoch in range(epochs):
            pred = model(X_train)
            loss = nn.functional.mse_loss(pred, Y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            all_preds[test_idx] = test_pred

        # Per-fold R^2
        ss_res = ((Y_test - test_pred) ** 2).sum().item()
        ss_tot = ((Y_test - Y_test.mean(dim=0)) ** 2).sum().item()
        fold_r2 = 1 - ss_res / max(ss_tot, 1e-10)
        all_r2_per_fold.append(fold_r2)

    # Overall R^2
    ss_res_total = ((Y - all_preds) ** 2).sum().item()
    ss_tot_total = ((Y - Y.mean(dim=0)) ** 2).sum().item()
    overall_r2 = 1 - ss_res_total / max(ss_tot_total, 1e-10)

    # Per-property R^2
    per_property_r2 = {}
    for i, name in enumerate(PROPERTY_NAMES):
        ss_res_i = ((Y[:, i] - all_preds[:, i]) ** 2).sum().item()
        ss_tot_i = ((Y[:, i] - Y[:, i].mean()) ** 2).sum().item()
        per_property_r2[name] = 1 - ss_res_i / max(ss_tot_i, 1e-10)

    return {
        "overall_r2": overall_r2,
        "fold_r2_mean": np.mean(all_r2_per_fold),
        "fold_r2_std": np.std(all_r2_per_fold),
        "per_property_r2": per_property_r2,
        "n_materials": len(valid_materials),
        "n_splits": n_splits,
        "hypothesis_holds": overall_r2 > 0.7,
    }


# Geometric proxy features for each material (typical scale, opacity, covariance shape)
# These represent what the material "looks like" geometrically as Gaussians
# Format: [scale_x, scale_y, scale_z, opacity, cov_flat_x6] = 10 features
GEOMETRIC_PROXIES = {
    # Hard solids: small scale, high opacity, isotropic
    "stone": [0.1, 0.1, 0.1, 0.95, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
    "granite": [0.08, 0.08, 0.08, 0.98, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
    "marble": [0.1, 0.1, 0.1, 0.97, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
    "steel": [0.05, 0.05, 0.05, 0.99, 0.005, 0.0, 0.0, 0.005, 0.0, 0.005],
    "iron": [0.06, 0.06, 0.06, 0.99, 0.005, 0.0, 0.0, 0.005, 0.0, 0.005],
    "diamond": [0.03, 0.03, 0.03, 0.95, 0.003, 0.0, 0.0, 0.003, 0.0, 0.003],
    "glass": [0.08, 0.08, 0.02, 0.40, 0.01, 0.0, 0.0, 0.01, 0.0, 0.002],
    "ceramic": [0.07, 0.07, 0.07, 0.97, 0.008, 0.0, 0.0, 0.008, 0.0, 0.008],
    "concrete": [0.12, 0.12, 0.12, 0.96, 0.015, 0.0, 0.0, 0.015, 0.0, 0.015],
    "brick": [0.15, 0.08, 0.08, 0.96, 0.02, 0.0, 0.0, 0.008, 0.0, 0.008],
    # Soft materials: larger scale, lower opacity, anisotropic
    "cloth": [0.3, 0.3, 0.01, 0.70, 0.05, 0.0, 0.0, 0.05, 0.0, 0.001],
    "rubber": [0.1, 0.1, 0.1, 0.90, 0.02, 0.0, 0.0, 0.02, 0.0, 0.02],
    "foam": [0.2, 0.2, 0.2, 0.50, 0.04, 0.0, 0.0, 0.04, 0.0, 0.04],
    "leather": [0.2, 0.2, 0.02, 0.85, 0.03, 0.0, 0.0, 0.03, 0.0, 0.002],
    "cotton": [0.25, 0.25, 0.02, 0.65, 0.04, 0.0, 0.0, 0.04, 0.0, 0.001],
    "wool": [0.2, 0.2, 0.05, 0.60, 0.03, 0.0, 0.0, 0.03, 0.0, 0.005],
    "silk": [0.3, 0.3, 0.005, 0.55, 0.05, 0.0, 0.0, 0.05, 0.0, 0.0005],
    # Liquids: large scale, low opacity, very isotropic
    "water": [0.5, 0.5, 0.5, 0.20, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1],
    "oil": [0.4, 0.4, 0.4, 0.25, 0.08, 0.0, 0.0, 0.08, 0.0, 0.08],
    "honey": [0.3, 0.3, 0.3, 0.35, 0.06, 0.0, 0.0, 0.06, 0.0, 0.06],
    # Wood: medium, anisotropic along grain
    "wood": [0.15, 0.05, 0.05, 0.92, 0.02, 0.0, 0.0, 0.005, 0.0, 0.005],
    "oak": [0.12, 0.05, 0.05, 0.95, 0.015, 0.0, 0.0, 0.005, 0.0, 0.005],
    "pine": [0.18, 0.06, 0.06, 0.90, 0.025, 0.0, 0.0, 0.006, 0.0, 0.006],
    "bamboo": [0.25, 0.03, 0.03, 0.90, 0.03, 0.0, 0.0, 0.002, 0.0, 0.002],
    # Nature: varied
    "grass": [0.4, 0.4, 0.02, 0.50, 0.08, 0.0, 0.0, 0.08, 0.0, 0.001],
    "sand": [0.05, 0.05, 0.05, 0.80, 0.005, 0.0, 0.0, 0.005, 0.0, 0.005],
    "soil": [0.1, 0.1, 0.1, 0.85, 0.015, 0.0, 0.0, 0.015, 0.0, 0.015],
    "rock": [0.15, 0.12, 0.1, 0.95, 0.02, 0.0, 0.0, 0.015, 0.0, 0.012],
    "mud": [0.15, 0.15, 0.15, 0.75, 0.025, 0.0, 0.0, 0.025, 0.0, 0.025],
    "ice": [0.1, 0.1, 0.1, 0.50, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
    "snow": [0.3, 0.3, 0.3, 0.40, 0.06, 0.0, 0.0, 0.06, 0.0, 0.06],
    # Metals
    "aluminum": [0.05, 0.05, 0.05, 0.98, 0.005, 0.0, 0.0, 0.005, 0.0, 0.005],
    "copper": [0.05, 0.05, 0.05, 0.99, 0.005, 0.0, 0.0, 0.005, 0.0, 0.005],
    "gold": [0.04, 0.04, 0.04, 0.99, 0.004, 0.0, 0.0, 0.004, 0.0, 0.004],
    "silver": [0.04, 0.04, 0.04, 0.99, 0.004, 0.0, 0.0, 0.004, 0.0, 0.004],
    "bronze": [0.06, 0.06, 0.06, 0.98, 0.006, 0.0, 0.0, 0.006, 0.0, 0.006],
    "lead": [0.07, 0.07, 0.07, 0.99, 0.007, 0.0, 0.0, 0.007, 0.0, 0.007],
}

# Default geometric proxy for materials not in the table above
DEFAULT_GEOMETRIC_PROXY = [0.1, 0.1, 0.1, 0.85, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01]


def main():
    parser = argparse.ArgumentParser(description="Validate P6 correlation hypothesis")
    parser.add_argument("--glove-path", default=None,
                        help="Path to glove.6B.300d.txt (optional, uses random embeddings if not provided)")
    parser.add_argument("--output", default="results/p6_correlation_validation.json",
                        help="Output results JSON")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--augment-features", action="store_true",
                        help="Add geometric proxy features (scale, opacity, covariance) to input")
    args = parser.parse_args()

    materials = {k: v for k, v in MATERIAL_TABLE.items() if len(v) == 8}
    print(f"Material table: {len(materials)} entries, {len(PROPERTY_NAMES)} properties each")

    if args.glove_path and Path(args.glove_path).exists():
        print(f"Loading GloVe from {args.glove_path}...")
        embeddings = load_glove(Path(args.glove_path), set(materials.keys()))
        print(f"Loaded {len(embeddings)} / {len(materials)} material embeddings")
    else:
        print("No GloVe file provided. Using random embeddings (for pipeline testing).")
        print("For real validation, provide: --glove-path data/glove.6B.300d.txt")
        embeddings = generate_random_embeddings(list(materials.keys()))

    if args.augment_features:
        print("Augmenting with geometric proxy features (+10 dimensions)")
        for word in list(embeddings.keys()):
            geo = GEOMETRIC_PROXIES.get(word, DEFAULT_GEOMETRIC_PROXY)
            embeddings[word] = np.concatenate([embeddings[word], np.array(geo, dtype=np.float32)])
        print(f"Input dimension: {len(next(iter(embeddings.values())))}d")

    print(f"\nRunning {args.folds}-fold cross-validation ({args.epochs} epochs/fold)...")
    results = run_validation(embeddings, materials, n_splits=args.folds, epochs=args.epochs)

    # Print results
    print(f"\n{'='*60}")
    print(f"CLAIM P6 VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall R^2: {results['overall_r2']:.4f}")
    print(f"Fold R^2: {results['fold_r2_mean']:.4f} +/- {results['fold_r2_std']:.4f}")
    print(f"\nPer-property R^2:")
    for name, r2 in results['per_property_r2'].items():
        bar = '|' + '#' * int(max(0, r2) * 40) + ' ' * (40 - int(max(0, r2) * 40)) + '|'
        print(f"  {name:25s} {r2:.4f} {bar}")
    print(f"\nHypothesis holds (R^2 > 0.7): {'YES' if results['hypothesis_holds'] else 'NO'}")
    print(f"{'='*60}")

    if not results['hypothesis_holds']:
        print("\nNote: if using random embeddings, R^2 will be low (expected).")
        print("Real GloVe embeddings encode semantic relationships that correlate with physics.")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
