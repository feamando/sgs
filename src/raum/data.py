"""
Synthetic scene data generation for Raum.

Generates (sentence, scene_ground_truth) pairs from templates.
Handles tokenization via GloVe word2idx or SentencePiece.

When a blob library is provided (via `object_names` parameter), the
scene generator samples from the full blob vocabulary rather than the
6 procedural primitives.
"""

import json
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from torch.utils.data import Dataset

from .vocab import (
    OBJECTS, COLORS, SIZES, RELATIONS,
    ROLE_OBJECT, ROLE_COLOR, ROLE_SIZE, ROLE_RELATION, ROLE_OTHER,
    N_OBJECTS, N_ROLES,
)


def load_blob_object_names(blobs_dir: str | Path) -> dict[str, int]:
    """Load object names from blob library index.json.

    Returns dict mapping object_name -> blob_id (same format as OBJECTS).
    """
    index_path = Path(blobs_dir) / "index.json"
    with open(index_path) as f:
        names = json.load(f)
    if isinstance(names, list):
        return {name: i for i, name in enumerate(names)}
    return {name: i for i, name in enumerate(names.keys())}


@dataclass
class ObjectGT:
    """Ground truth for one object in a scene."""
    obj_type: int              # index into OBJECTS
    color: list[float]         # [r, g, b]
    scale: float               # size multiplier
    position: list[float]      # [x, y, z]


@dataclass
class SceneGT:
    """Ground truth for a complete scene."""
    sentence: str
    words: list[str]
    objects: list[ObjectGT]
    # Per-word labels (aligned with words list)
    role_labels: list[int]
    obj_labels: list[int]      # object class (-1 if not an object word)
    color_labels: list[list[float]]  # RGB (-1 if not a color word)
    size_labels: list[float]   # scale (-1 if not a size word)
    relation_label: list[float] | None  # xyz offset (None if no relation)


# ── Sentence generators ──

def _gen_single_object(
    color: str | None = None,
    size: str | None = None,
    obj: str | None = None,
    objects: dict[str, int] | None = None,
) -> SceneGT:
    """Template: 'a [size] {color} {object}'"""
    objects = objects or OBJECTS
    color = color or random.choice(list(COLORS.keys()))
    obj = obj or random.choice(list(objects.keys()))
    use_size = size is not None or random.random() < 0.5
    size = size or random.choice(list(SIZES.keys()))

    if use_size:
        words = ["a", size, color, obj]
        roles = [ROLE_OTHER, ROLE_SIZE, ROLE_COLOR, ROLE_OBJECT]
        scale = SIZES[size]
    else:
        words = ["a", color, obj]
        roles = [ROLE_OTHER, ROLE_COLOR, ROLE_OBJECT]
        scale = 1.0

    obj_gt = ObjectGT(
        obj_type=objects[obj],
        color=COLORS[color],
        scale=scale,
        position=[0.0, 0.0, 0.0],
    )

    obj_labels = [objects[obj] if r == ROLE_OBJECT else -1 for r in roles]
    color_labels = [COLORS[color] if r == ROLE_COLOR else [-1, -1, -1] for r in roles]
    size_labels = [scale if r == ROLE_SIZE else -1.0 for r in roles]

    return SceneGT(
        sentence=" ".join(words),
        words=words,
        objects=[obj_gt],
        role_labels=roles,
        obj_labels=obj_labels,
        color_labels=color_labels,
        size_labels=size_labels,
        relation_label=None,
    )


def _gen_two_objects(
    color1: str | None = None, obj1: str | None = None,
    relation: str | None = None,
    color2: str | None = None, obj2: str | None = None,
    objects: dict[str, int] | None = None,
) -> SceneGT:
    """Template: 'a {color1} {object1} {relation} a {color2} {object2}'"""
    objects = objects or OBJECTS
    color1 = color1 or random.choice(list(COLORS.keys()))
    obj1 = obj1 or random.choice(list(objects.keys()))
    relation = relation or random.choice(list(RELATIONS.keys()))
    color2 = color2 or random.choice(list(COLORS.keys()))
    obj2 = obj2 or random.choice(list(objects.keys()))

    # Some relations are multi-word in natural English but we use single-word keys
    # "left" means "left of", etc.
    rel_word = relation
    words = ["a", color1, obj1, rel_word, "a", color2, obj2]
    roles = [
        ROLE_OTHER, ROLE_COLOR, ROLE_OBJECT,
        ROLE_RELATION,
        ROLE_OTHER, ROLE_COLOR, ROLE_OBJECT,
    ]

    # "X rel Y" in English anchors Y and offsets X.
    # "cone above cylinder" → cylinder at origin, cone at +y.
    offset = RELATIONS[relation]
    obj1_gt = ObjectGT(
        obj_type=objects[obj1], color=COLORS[color1],
        scale=1.0, position=list(offset),
    )
    obj2_gt = ObjectGT(
        obj_type=objects[obj2], color=COLORS[color2],
        scale=1.0, position=[0.0, 0.0, 0.0],
    )

    obj_labels = []
    color_labels = []
    size_labels = []
    obj_idx = 0
    for i, r in enumerate(roles):
        if r == ROLE_OBJECT:
            obj_labels.append(objects[obj1] if obj_idx == 0 else objects[obj2])
            obj_idx += 1
        else:
            obj_labels.append(-1)

        if r == ROLE_COLOR:
            c = COLORS[color1] if i < 3 else COLORS[color2]
            color_labels.append(c)
        else:
            color_labels.append([-1, -1, -1])

        size_labels.append(-1.0)

    return SceneGT(
        sentence=" ".join(words),
        words=words,
        objects=[obj1_gt, obj2_gt],
        role_labels=roles,
        obj_labels=obj_labels,
        color_labels=color_labels,
        size_labels=size_labels,
        relation_label=offset,
    )


def _gen_three_objects(
    obj1: str | None = None, obj2: str | None = None, obj3: str | None = None,
    objects: dict[str, int] | None = None,
) -> SceneGT:
    """Template: 'a {c1} {o1} {rel1} a {c2} {o2} and a {c3} {o3} {rel2} the {o2}'"""
    objects = objects or OBJECTS
    obj1 = obj1 or random.choice(list(objects.keys()))
    obj2 = obj2 or random.choice(list(objects.keys()))
    obj3 = obj3 or random.choice(list(objects.keys()))
    c1 = random.choice(list(COLORS.keys()))
    c2 = random.choice(list(COLORS.keys()))
    c3 = random.choice(list(COLORS.keys()))
    rel1 = random.choice(list(RELATIONS.keys()))
    rel2 = random.choice(list(RELATIONS.keys()))

    words = ["a", c1, obj1, rel1, "a", c2, obj2, "and", "a", c3, obj3, rel2, "the", obj2]
    roles = [
        ROLE_OTHER, ROLE_COLOR, ROLE_OBJECT,
        ROLE_RELATION,
        ROLE_OTHER, ROLE_COLOR, ROLE_OBJECT,
        ROLE_OTHER,
        ROLE_OTHER, ROLE_COLOR, ROLE_OBJECT,
        ROLE_RELATION,
        ROLE_OTHER, ROLE_OTHER,
    ]

    offset1 = RELATIONS[rel1]
    offset2 = RELATIONS[rel2]
    # obj2 at origin, obj1 offset by rel1, obj3 offset by rel2
    obj1_gt = ObjectGT(
        obj_type=objects[obj1], color=COLORS[c1],
        scale=1.0, position=list(offset1),
    )
    obj2_gt = ObjectGT(
        obj_type=objects[obj2], color=COLORS[c2],
        scale=1.0, position=[0.0, 0.0, 0.0],
    )
    obj3_gt = ObjectGT(
        obj_type=objects[obj3], color=COLORS[c3],
        scale=1.0, position=list(offset2),
    )

    obj_labels = []
    color_labels = []
    size_labels = []
    obj_idx = 0
    obj_list = [obj1, obj2, obj3]
    color_list = [c1, c2, c3]
    for i, r in enumerate(roles):
        if r == ROLE_OBJECT and obj_idx < 3:
            obj_labels.append(objects[obj_list[obj_idx]])
            obj_idx += 1
        else:
            obj_labels.append(-1)

        if r == ROLE_COLOR:
            ci = min(len(color_labels), 2)
            obj_color_idx = sum(1 for rr in roles[:i] if rr == ROLE_COLOR)
            if obj_color_idx < 3:
                color_labels.append(COLORS[color_list[obj_color_idx]])
            else:
                color_labels.append([-1, -1, -1])
        else:
            color_labels.append([-1, -1, -1])

        size_labels.append(-1.0)

    return SceneGT(
        sentence=" ".join(words),
        words=words,
        objects=[obj1_gt, obj2_gt, obj3_gt],
        role_labels=roles,
        obj_labels=obj_labels,
        color_labels=color_labels,
        size_labels=size_labels,
        relation_label=offset1,
    )


def generate_dataset(
    n_samples: int,
    two_object_ratio: float = 0.7,
    n_objects_max: int = 2,
    seed: int = 42,
) -> list[SceneGT]:
    """Generate a dataset of random scenes."""
    rng = random.Random(seed)
    old_state = random.getstate()
    random.seed(seed)

    scenes = []
    for _ in range(n_samples):
        r = random.random()
        if n_objects_max >= 3 and r < 0.3:
            scenes.append(_gen_three_objects())
        elif r < (0.3 + 0.5 if n_objects_max >= 3 else two_object_ratio):
            scenes.append(_gen_two_objects())
        else:
            scenes.append(_gen_single_object())

    random.setstate(old_state)
    return scenes


def generate_comp_gen_split(
    n_train: int = 15000,
    n_val: int = 2500,
    n_test: int = 2500,
    n_objects_max: int = 2,
    objects: dict[str, int] | None = None,
    seed: int = 42,
) -> tuple[list[SceneGT], list[SceneGT], list[SceneGT]]:
    """
    Generate train/val/test with compositional generalization test.

    Test set contains object PAIRS never seen together in training.
    Individual objects all appear in training (just not in these combinations).
    When n_objects_max >= 3, training includes 3-object scenes.

    If `objects` is provided, use that vocabulary instead of the default
    6 procedural primitives. Pass the output of load_blob_object_names().
    """
    objects = objects or OBJECTS
    obj_names = list(objects.keys())
    n_obj = len(obj_names)

    # All possible ordered pairs
    all_pairs = [(a, b) for a in obj_names for b in obj_names]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    # Hold out ~20% of pairs for test
    n_held = max(len(all_pairs) // 5, 2)
    # Ensure each object appears in at least one training pair
    held_out = set()
    for pair in all_pairs:
        if len(held_out) >= n_held:
            break
        candidate = held_out | {pair}
        remaining = [p for p in all_pairs if p not in candidate]
        objs_in_train = {o for p in remaining for o in p}
        if objs_in_train == set(obj_names):
            held_out.add(pair)

    train_pairs = [p for p in all_pairs if p not in held_out]

    old_state = random.getstate()
    random.seed(seed)

    def _gen_scene(pairs):
        r = random.random()
        if n_objects_max >= 3 and r < 0.3:
            obj1, obj2 = random.choice(pairs)
            obj3 = random.choice(obj_names)
            return _gen_three_objects(obj1=obj1, obj2=obj2, obj3=obj3, objects=objects)
        elif r < (0.8 if n_objects_max >= 3 else 0.7):
            obj1, obj2 = random.choice(pairs)
            return _gen_two_objects(obj1=obj1, obj2=obj2, objects=objects)
        else:
            return _gen_single_object(objects=objects)

    train = [_gen_scene(train_pairs) for _ in range(n_train)]
    val = [_gen_scene(train_pairs) for _ in range(n_val)]

    # Test: ONLY held-out pairs (2- and 3-object scenes)
    test = []
    held_list = list(held_out)
    for _ in range(n_test):
        if n_objects_max >= 3 and random.random() < 0.4:
            obj1, obj2 = random.choice(held_list)
            obj3 = random.choice(obj_names)
            test.append(_gen_three_objects(obj1=obj1, obj2=obj2, obj3=obj3, objects=objects))
        else:
            obj1, obj2 = random.choice(held_list)
            test.append(_gen_two_objects(obj1=obj1, obj2=obj2, objects=objects))

    random.setstate(old_state)

    print(f"Comp-gen split (n_objects_max={n_objects_max}):")
    print(f"  Train pairs: {len(train_pairs)}, Held-out pairs: {len(held_out)}")
    print(f"  Held-out: {held_out}")
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ── Tokenization ──

def tokenize_scene(scene: SceneGT, word2idx: dict[str, int], max_objects: int = 3) -> dict:
    """
    Convert a SceneGT to tensors for model input.

    Returns dict with:
        token_ids: [n_words] long tensor
        mask: [n_words] float tensor (1 for real, 0 for pad)
        role_labels: [n_words] long
        obj_labels: [n_words] long (-1 for non-object)
        color_labels: [n_words, 3] float (-1 for non-color)
        size_labels: [n_words] float (-1 for non-size)
        relation_label: [3] float (or zeros if no relation)
        n_objects: int
        object_positions: [max_objects, 3] float
        object_types: [max_objects] long
        object_colors: [max_objects, 3] float
        object_scales: [max_objects] float
    """
    unk_idx = word2idx.get("<unk>", word2idx.get("unk", 0))

    ids = [word2idx.get(w.lower(), unk_idx) for w in scene.words]

    max_obj = max_objects
    positions = torch.zeros(max_obj, 3)
    types = torch.full((max_obj,), -1, dtype=torch.long)
    colors = torch.zeros(max_obj, 3)
    scales = torch.ones(max_obj)

    for i, obj in enumerate(scene.objects[:max_obj]):
        positions[i] = torch.tensor(obj.position)
        types[i] = obj.obj_type
        colors[i] = torch.tensor(obj.color)
        scales[i] = obj.scale

    return {
        "token_ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.ones(len(ids), dtype=torch.float32),
        "role_labels": torch.tensor(scene.role_labels, dtype=torch.long),
        "obj_labels": torch.tensor(scene.obj_labels, dtype=torch.long),
        "color_labels": torch.tensor(scene.color_labels, dtype=torch.float32),
        "size_labels": torch.tensor(scene.size_labels, dtype=torch.float32),
        "relation_label": torch.tensor(scene.relation_label or [0, 0, 0], dtype=torch.float32),
        "n_objects": len(scene.objects),
        "object_positions": positions,
        "object_types": types,
        "object_colors": colors,
        "object_scales": scales,
    }


# ── Dataset ──

class RaumDataset(Dataset):
    """PyTorch dataset wrapping a list of SceneGT + word2idx."""

    def __init__(self, scenes: list[SceneGT], word2idx: dict[str, int], max_objects: int = 3):
        self.scenes = scenes
        self.word2idx = word2idx
        self.max_objects = max_objects

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return tokenize_scene(self.scenes[idx], self.word2idx, max_objects=self.max_objects)


def collate_raum(batch: list[dict]) -> dict:
    """Pad variable-length sequences and stack into batch tensors."""
    max_len = max(d["token_ids"].shape[0] for d in batch)

    out = {}
    for key in ["token_ids", "role_labels", "obj_labels", "size_labels"]:
        padded = []
        for d in batch:
            t = d[key]
            pad_val = 0 if key == "token_ids" else -1
            padded.append(torch.nn.functional.pad(
                t, (0, max_len - t.shape[0]), value=pad_val,
            ))
        out[key] = torch.stack(padded)

    # mask
    masks = []
    for d in batch:
        m = d["mask"]
        masks.append(torch.nn.functional.pad(m, (0, max_len - m.shape[0]), value=0.0))
    out["mask"] = torch.stack(masks)

    # color_labels: [B, max_len, 3]
    colors = []
    for d in batch:
        c = d["color_labels"]
        pad_len = max_len - c.shape[0]
        if pad_len > 0:
            c = torch.cat([c, torch.full((pad_len, 3), -1.0)], dim=0)
        colors.append(c)
    out["color_labels"] = torch.stack(colors)

    # Fixed-size tensors: just stack
    for key in ["relation_label", "object_positions", "object_types",
                "object_colors", "object_scales"]:
        out[key] = torch.stack([d[key] for d in batch])

    out["n_objects"] = torch.tensor([d["n_objects"] for d in batch])

    return out
