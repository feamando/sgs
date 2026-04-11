"""
Scene assembly: convert model predictions → 3D Gaussian scene.
"""

import torch
from .vocab import ROLE_OBJECT, ROLE_COLOR, ROLE_SIZE, ROLE_RELATION, OBJECT_NAMES
from .templates import GaussianTemplate, build_template_library


@torch.no_grad()
def assemble_scene(
    preds: dict,
    template_lib: dict[str, GaussianTemplate],
    sample_idx: int = 0,
) -> list[dict]:
    """
    Convert per-word predictions into a positioned, colored scene.

    Returns list of objects, each with:
        type_name: str
        type_idx: int
        color: [3] tensor
        scale: float
        position: [3] tensor
        template: GaussianTemplate (transformed)
    """
    roles = preds["role_logits"][sample_idx].argmax(dim=-1)   # [L]
    L = roles.shape[0]

    objects = []
    current_color = torch.tensor([0.5, 0.5, 0.5])
    current_scale = 1.0

    for i in range(L):
        role = roles[i].item()

        if role == ROLE_COLOR:
            current_color = preds["color_pred"][sample_idx, i].cpu()

        elif role == ROLE_SIZE:
            current_scale = preds["size_pred"][sample_idx, i].item()

        elif role == ROLE_OBJECT:
            obj_idx = preds["obj_logits"][sample_idx, i].argmax().item()
            obj_name = OBJECT_NAMES[obj_idx]

            # Position: first object at origin, subsequent offset by relation
            if len(objects) == 0:
                position = torch.zeros(3)
            else:
                # Find the relation word that came before this object
                rel_pos = None
                for j in range(i - 1, -1, -1):
                    if roles[j].item() == ROLE_RELATION:
                        rel_pos = j
                        break
                if rel_pos is not None:
                    position = preds["relation_pred"][sample_idx, rel_pos].cpu()
                else:
                    position = torch.zeros(3)

            # Build transformed template
            template = template_lib.get(obj_name)
            if template is not None:
                transformed = GaussianTemplate(
                    means=template.means * current_scale + position.unsqueeze(0),
                    scales=template.scales + torch.log(torch.tensor(current_scale)).item(),
                    opacities=template.opacities.clone(),
                )
            else:
                transformed = None

            objects.append({
                "type_name": obj_name,
                "type_idx": obj_idx,
                "color": current_color.clone(),
                "scale": current_scale,
                "position": position.clone(),
                "template": transformed,
            })

            # Reset for next object
            current_color = torch.tensor([0.5, 0.5, 0.5])
            current_scale = 1.0

    return objects
