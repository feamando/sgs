"""
Scene vocabulary for Raum PoC — objects, colors, sizes, spatial relations.

All words chosen to exist in GloVe top 50K.
"""

import torch

# ── Object types ──
# Each maps to an index used for classification.
OBJECTS = {
    "sphere": 0,
    "cube": 1,
    "cylinder": 2,
    "cone": 3,
    "plane": 4,
    "torus": 5,
}
OBJECT_NAMES = {v: k for k, v in OBJECTS.items()}
N_OBJECTS = len(OBJECTS)

# ── Colors (RGB, 0-1) ──
COLORS = {
    "red":    [1.0, 0.0, 0.0],
    "blue":   [0.0, 0.0, 1.0],
    "green":  [0.0, 1.0, 0.0],
    "yellow": [1.0, 1.0, 0.0],
    "white":  [1.0, 1.0, 1.0],
    "black":  [0.1, 0.1, 0.1],
    "orange": [1.0, 0.5, 0.0],
    "purple": [0.5, 0.0, 1.0],
}

# ── Sizes (scale multiplier) ──
SIZES = {
    "tiny":   0.3,
    "small":  0.6,
    "medium": 1.0,
    "large":  1.5,
    "huge":   2.0,
}

# ── Spatial relations → xyz offset from reference object ──
# Coordinate system: x=right, y=up, z=toward camera
RELATIONS = {
    "above":  [0.0,  1.5,  0.0],
    "below":  [0.0, -1.5,  0.0],
    "left":   [-1.5, 0.0,  0.0],   # "left of" → "left" carries the semantics
    "right":  [1.5,  0.0,  0.0],   # "right of" → "right"
    "behind": [0.0,  0.0, -1.5],
    "on":     [0.0,  0.8,  0.0],
    "beside": [1.2,  0.0,  0.0],
}

# ── Word role labels ──
ROLE_OBJECT   = 0
ROLE_COLOR    = 1
ROLE_SIZE     = 2
ROLE_RELATION = 3
ROLE_OTHER    = 4
N_ROLES = 5
ROLE_NAMES = {0: "object", 1: "color", 2: "size", 3: "relation", 4: "other"}

# ── All scene vocabulary words (for checking GloVe coverage) ──
ALL_SCENE_WORDS = (
    set(OBJECTS.keys())
    | set(COLORS.keys())
    | set(SIZES.keys())
    | set(RELATIONS.keys())
    | {"a", "of"}
)


def color_tensor(name: str) -> torch.Tensor:
    """Get color as a [3] tensor."""
    return torch.tensor(COLORS[name], dtype=torch.float32)


def relation_offset(name: str) -> torch.Tensor:
    """Get spatial offset as a [3] tensor."""
    return torch.tensor(RELATIONS[name], dtype=torch.float32)
