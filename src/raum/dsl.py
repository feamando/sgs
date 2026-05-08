"""
DSL v1: editable intermediate between bridge output and renderer.

The DSL is a JSON structure describing a 3D scene. The bridge produces
it, the renderer consumes it, and the user can edit it in the demo UI
to tweak the scene without re-running the bridge.

Schema:
{
  "version": 1,
  "objects": [
    {"id": "obj_0", "blob": "car", "color": [0.8, 0.1, 0.1],
     "scale": 1.0, "position": [0.0, 0.0, 0.0]},
    ...
  ],
  "relations": [
    {"subject": "obj_0", "rel": "left", "anchor": "obj_1"},
    ...
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .data import RELATION_NAMES


DSL_VERSION = 1

VALID_RELATIONS = set(RELATION_NAMES)


def validate(dsl: dict) -> tuple[bool, list[str]]:
    """Validate a DSL dict. Returns (is_valid, list_of_errors)."""
    errors = []

    if not isinstance(dsl, dict):
        return False, ["DSL must be a JSON object"]

    if dsl.get("version") != DSL_VERSION:
        errors.append(f"unsupported version: {dsl.get('version')} (expected {DSL_VERSION})")

    objects = dsl.get("objects", [])
    if not isinstance(objects, list):
        errors.append("'objects' must be an array")
        objects = []

    obj_ids = set()
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            errors.append(f"objects[{i}]: must be an object")
            continue
        oid = obj.get("id")
        if not oid or not isinstance(oid, str):
            errors.append(f"objects[{i}]: missing or invalid 'id'")
        elif oid in obj_ids:
            errors.append(f"objects[{i}]: duplicate id '{oid}'")
        else:
            obj_ids.add(oid)

        if "blob" not in obj:
            errors.append(f"objects[{i}]: missing 'blob'")

        pos = obj.get("position")
        if pos is not None and (not isinstance(pos, list) or len(pos) != 3):
            errors.append(f"objects[{i}]: 'position' must be [x, y, z]")

        color = obj.get("color")
        if color is not None and (not isinstance(color, list) or len(color) != 3):
            errors.append(f"objects[{i}]: 'color' must be [r, g, b]")

    relations = dsl.get("relations", [])
    if not isinstance(relations, list):
        errors.append("'relations' must be an array")
        relations = []

    for i, rel in enumerate(relations):
        if not isinstance(rel, dict):
            errors.append(f"relations[{i}]: must be an object")
            continue
        if rel.get("subject") not in obj_ids:
            errors.append(f"relations[{i}]: unknown subject '{rel.get('subject')}'")
        if rel.get("anchor") not in obj_ids:
            errors.append(f"relations[{i}]: unknown anchor '{rel.get('anchor')}'")
        if rel.get("rel") not in VALID_RELATIONS:
            errors.append(f"relations[{i}]: unknown relation '{rel.get('rel')}'")

    return len(errors) == 0, errors


def bridge_output_to_dsl(
    out: dict,
    mask,
    blob_names: list[str] | None = None,
    sample_index: int = 0,
    object_role_id: int = 0,
) -> dict:
    """
    Convert bridge head outputs to a DSL v1 dict.

    Args:
        out: bridge forward output (positions, template_logits, colors, scales, role_logits)
        mask: [B, N] mask tensor
        blob_names: ordered list of blob class names (index matches template_logits dim)
        sample_index: which sample in the batch
        object_role_id: role ID for object tokens (default 0 = ROLE_OBJECT)

    Returns:
        DSL v1 dict ready for JSON serialization or renderer consumption.
    """
    import torch

    b = sample_index
    positions = out["positions"][b].detach().cpu()
    tpl_logits = out["template_logits"][b].detach().cpu()
    role_logits = out["role_logits"][b].detach().cpu()
    colors = out["colors"][b].detach().cpu()
    scales = out["scales"][b].detach().cpu()
    m = mask[b].detach().cpu()

    role_pred = role_logits.argmax(dim=-1)

    objects = []
    for i in range(positions.shape[0]):
        if m[i] < 0.5:
            continue
        if int(role_pred[i]) != object_role_id:
            continue

        tpl_id = int(tpl_logits[i].argmax())
        blob_name = blob_names[tpl_id] if blob_names and tpl_id < len(blob_names) else f"blob_{tpl_id}"
        conf = float(torch.softmax(tpl_logits[i], dim=-1)[tpl_id])

        if conf < 0.3:
            continue

        objects.append({
            "id": f"obj_{len(objects)}",
            "blob": blob_name,
            "color": [round(float(colors[i, c]), 3) for c in range(3)],
            "scale": round(float(scales[i]), 3),
            "position": [round(float(positions[i, c]), 3) for c in range(3)],
        })

    # Infer relations from positions (simple heuristic: pairwise offsets)
    relations = []
    if len(objects) >= 2:
        from .data import RELATIONS
        anchor = objects[1]
        for obj in objects:
            if obj["id"] == anchor["id"]:
                continue
            diff = [obj["position"][c] - anchor["position"][c] for c in range(3)]
            best_rel = None
            best_dist = float("inf")
            for rel_name, offset in RELATIONS.items():
                dist = sum((diff[c] - offset[c]) ** 2 for c in range(3))
                if dist < best_dist:
                    best_dist = dist
                    best_rel = rel_name
            if best_rel and best_dist < 3.0:
                relations.append({
                    "subject": obj["id"],
                    "rel": best_rel,
                    "anchor": anchor["id"],
                })

    return {
        "version": DSL_VERSION,
        "objects": objects,
        "relations": relations,
    }


def dsl_to_json(dsl: dict, indent: int = 2) -> str:
    return json.dumps(dsl, indent=indent)


def json_to_dsl(text: str) -> tuple[dict | None, list[str]]:
    """Parse JSON text into a DSL dict with validation."""
    try:
        dsl = json.loads(text)
    except json.JSONDecodeError as e:
        return None, [f"JSON parse error: {e}"]
    valid, errors = validate(dsl)
    return dsl, errors
