"""
Gemma multimodal spike: IMAGE -> Raum scene tree (the real reason to use Gemma).

Planck/Hertz are text-only and structurally CANNOT do this. Gemma 4 E4B is
Any-to-Any; feed it a reference photo/sketch of a building and have it emit the
same structure-only decomposition tree the text decomposer produces, so Raum can
"build THIS". This grounds meaning in an image at INFERENCE time (distinct from
VSP-for-LM, which baked grounding into a token table and died negative,
project_sgs_vsp_gate).

SPIKE DISCIPLINE (gate-and-kill): the ONLY question here is "can Gemma emit a
valid, in-vocab scene tree from an image?". If yes -> it's worth wiring an image
input into the Raum decomposer UI. If no -> the multimodal path is dead and we
say so. No UI work until this gate passes.

Reuses the text spike's contract exactly: same EXPANDABLE_PARTS constraint, same
parse_tree + validate_skeleton, same fill so the result renders in Raum. Only the
INPUT changes (image content block instead of text-only), via AutoProcessor.

CAVEAT (verify on the box): Gemma 4 multimodal loads via AutoProcessor +
AutoModelForImageTextToText (transformers >= 4.50). If the class name differs for
the installed build, the load block is the one thing to adjust; the message
schema (image + text content blocks in apply_chat_template) is standard.

Usage (4090 box):
  python scripts/gemma_image_to_tree.py --model models/gemma-4-e4b-it `
    --image path/to/castle_photo.jpg `
    --out results/gemma_image_tree.json --dump-tree output/gemma_img_scene.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.castle_grammar import EXPANDABLE_PARTS
from scripts.generate_trees_gemma import parse_tree, validate_skeleton


# Same schema + vocab constraint as the text path, but the scene comes from the
# IMAGE, not a text description. The instruction tells Gemma to LOOK.
SYSTEM_PROMPT = """You are a 3D scene-layout engine. Look at the image and output a JSON tree that places NAMED PARTS in space to reconstruct the structure shown. You do NOT draw geometry; a separate renderer expands each part.

Rules:
1. Root is {"name": "scene", "children": [...]}.
2. Each child has: "name", "position" [x,y,z], "scale" (float). NO "gaussians", NO nested children.
3. You may ONLY use these part names (suffix with _0,_1,... for repeats, or _n/_s/_e/_w for sides):
   %s
4. Ground goes first: use "hill" or "ground" or "cliff" at y=-0.1.
5. Place parts on top of the ground (y around 0.4-0.7), scene fits in a [-2,2] cube.
6. Reconstruct the building/scene in the image from these parts as best you can (match tower count, walls, gate placement to what you see).
7. Output ONLY the JSON. No prose, no markdown fences.""" % (", ".join(EXPANDABLE_PARTS),)


def load_gemma_mm(model_path):
    """Load Gemma 4 as an image-text model. Returns (processor, model, torch)."""
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _MM
    except ImportError:
        # older/newer builds: fall back to the generic multimodal causal class
        from transformers import AutoModelForCausalLM as _MM
    print(f"[gemma-img] loading {model_path} (multimodal) ...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = _MM.from_pretrained(
        model_path, dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    return processor, model, torch


def image_to_tree(processor, model, torch, image_path, max_new=1024, temperature=0.1):
    """One image -> validated + filled scene tree (or None). Same output contract
    as GemmaDecomposer.generate_tree so it drops into the Raum render path."""
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "Reconstruct this as a Raum scene tree."},
        ]},
    ]
    inputs = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=temperature > 0,
                             temperature=max(temperature, 1e-4))
    text = processor.decode(gen[0][input_len:], skip_special_tokens=True)
    raw = parse_tree(text)
    if raw is None:
        return None, text
    clean, _ = validate_skeleton(raw)
    if clean is None:
        return None, text
    # fill so it renders in Raum (same contract as the text decomposer)
    from scripts.infer_decomposer import fill_gaussians, shift_above_ground
    fill_gaussians(clean, None)
    shift_above_ground(clean)
    return clean, text


def main():
    p = argparse.ArgumentParser(description="Gemma multimodal spike: image -> Raum scene tree")
    p.add_argument("--model", required=True, help="local Gemma 4 folder")
    p.add_argument("--image", required=True, help="reference image (path or URL)")
    p.add_argument("--out", default="results/gemma_image_tree.json")
    p.add_argument("--dump-tree", default=None,
                   help="write the filled tree for infer_decomposer.py --scene-file")
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    args = p.parse_args()

    processor, model, torch = load_gemma_mm(args.model)
    tree, raw = image_to_tree(processor, model, torch, args.image,
                              max_new=args.max_new, temperature=args.temperature)

    print("\n[gemma-img] GATE (multimodal) =====================")
    if tree is None:
        print(f"  FAIL: no valid in-vocab tree from the image.")
        print(f"  raw output (first 400 chars):\n{raw[:400]}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"image": args.image, "parsed": False, "raw": raw[:2000]},
                  open(args.out, "w"), indent=2)
        print(f"  saved -> {args.out}")
        print("  KILL: multimodal image->tree does not work with this build/prompt.")
        sys.exit(1)

    leaves = tree.get("children", [])
    names = [c.get("name") for c in leaves]
    print(f"  PASS: valid tree, {len(leaves)} parts: {names}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"image": args.image, "parsed": True, "tree": tree}, open(args.out, "w"), indent=2)
    print(f"  saved -> {args.out}")
    if args.dump_tree:
        Path(args.dump_tree).parent.mkdir(parents=True, exist_ok=True)
        json.dump(tree, open(args.dump_tree, "w"), indent=2)
        print(f"  filled tree -> {args.dump_tree}")
        print(f"  render it: python scripts/infer_decomposer.py --scene-file {args.dump_tree} --no-snap --serve --port 8003")
    print("  PASS -> worth wiring an image-upload input into the Raum decomposer UI.")


if __name__ == "__main__":
    main()
