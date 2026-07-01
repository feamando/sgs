"""
Raum Path A: generate decomposition-tree training data with a LOCAL Gemma 4.

The existing generate_decomposition_trees.py calls the Claude API and emits DEEP
gaussian-bearing trees. Path A needs the opposite: SHALLOW skeletons (named
parts as leaves, position+scale only) of NEW scene types beyond the castle, so
the decomposer learns to COMPOSE arbitrary scenes and the learned fill
(train_fill.py) renders each part. And it should run locally + free on the box,
not via a paid API.

Gemma 4 (huggingface.co/collections/google/gemma-4) generates these cheaply.
The model is constrained to the part vocabulary the fill stage can render --
Gemma may NAME a "lighthouse" scene but must build it from known parts
(tower + ...), because fill richness caps usable data richness (a part the fill
can't render is a wasted label).

Output matches the castle_16 / build_stage3_dataset format:
  {"prompt": "...", "tree": {"name":"scene","children":[
      {"name":"hill","position":[..],"scale":..},
      {"name":"tower_0","position":[..],"scale":..}, ...]}}

Usage (4090 / a venv with transformers + the downloaded model):
  python scripts/generate_trees_gemma.py \
    --model models/gemma-4-e4b-it \
    --prompts data/gemma/scene_prompts.txt \
    --out data/decomposition_trees/gemma_train.json --n 200
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.castle_grammar import _part_kind, EXPANDABLE_PARTS


SYSTEM_PROMPT = """You are a 3D scene-layout engine. Given a scene description, output a JSON tree that places NAMED PARTS in space. You do NOT draw geometry; a separate renderer expands each part.

Rules:
1. Root is {"name": "scene", "children": [...]}.
2. Each child has: "name", "position" [x,y,z], "scale" (float). NO "gaussians", NO nested children.
3. You may ONLY use these part names (suffix with _0,_1,... for repeats, or _n/_s/_e/_w for sides):
   %s
4. Ground goes first: use "hill" or "ground" or "cliff" at y=-0.1.
5. Place parts on top of the ground (y around 0.4-0.7), scene fits in a [-2,2] cube.
6. Compose the requested scene from these parts as best you can (a lighthouse = a tall tower; a fort = towers + walls + keep).
7. Output ONLY the JSON. No prose, no markdown fences.

Scene: "%s"
""" % (", ".join(EXPANDABLE_PARTS), "%s")


def build_prompt(scene):
    return SYSTEM_PROMPT % scene


def parse_tree(text):
    """Extract the JSON object from the model's output, tolerant of fences."""
    text = text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    # grab the outermost {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def validate_skeleton(tree):
    """Shallow, parts-only, all names renderable by the fill stage. Returns
    (clean_tree, n_dropped) -- drops unknown-name children rather than failing."""
    if not isinstance(tree, dict) or tree.get("name") != "scene":
        return None, 0
    kids = tree.get("children", [])
    if not isinstance(kids, list) or not kids:
        return None, 0
    clean, dropped = [], 0
    GROUND = {"hill", "ground", "cliff"}
    for c in kids:
        if not isinstance(c, dict) or "name" not in c:
            dropped += 1
            continue
        nm = c["name"].lower()
        ok = _part_kind(nm) is not None or any(nm == g or nm.startswith(g) for g in GROUND)
        if not ok or c.get("gaussians") or c.get("children"):
            # not a known part, or not shallow -> drop (fill can't render it)
            if not ok:
                dropped += 1
                continue
            c.pop("gaussians", None)
            c.pop("children", None)
        c.setdefault("position", [0, 0, 0])
        c.setdefault("scale", 1.0)
        clean.append(c)
    if not clean:
        return None, dropped
    return {"name": "scene", "children": clean}, dropped


def main():
    p = argparse.ArgumentParser(description="Generate shallow scene skeletons with local Gemma 4")
    p.add_argument("--model", required=True, help="local path to a downloaded Gemma 4 model")
    p.add_argument("--prompts", required=True, help="text file, one scene prompt per line")
    p.add_argument("--out", default="data/decomposition_trees/gemma_train.json")
    p.add_argument("--n", type=int, default=200, help="max prompts to process")
    p.add_argument("--repeat", type=int, default=1, help="samples per prompt (temperature varies them)")
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gemma] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()

    prompts = [l.strip() for l in open(args.prompts) if l.strip()][:args.n]
    print(f"[gemma] {len(prompts)} prompts x{args.repeat}")

    records, n_drop, n_fail = [], 0, 0
    for i, scene in enumerate(prompts):
        for _ in range(args.repeat):
            msgs = [{"role": "user", "content": build_prompt(scene)}]
            # Newer transformers returns a BatchEncoding (dict-like) from
            # apply_chat_template; generate() wants **inputs, not a bare tensor.
            inputs = tok.apply_chat_template(
                msgs, add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to(model.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new,
                                     do_sample=args.temperature > 0,
                                     temperature=max(args.temperature, 1e-4))
            text = tok.decode(gen[0][input_len:], skip_special_tokens=True)
            tree = parse_tree(text)
            if tree is None:
                n_fail += 1
                continue
            clean, dropped = validate_skeleton(tree)
            n_drop += dropped
            if clean is None:
                n_fail += 1
                continue
            records.append({"prompt": scene, "tree": clean})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(prompts)}] kept {len(records)}  fail {n_fail}  dropped-leaves {n_drop}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=2)
    print(f"[gemma] saved {len(records)} records -> {args.out} "
          f"(fail {n_fail}, dropped-leaves {n_drop})")
    print(f"[gemma] next: mix into train_decomposer.py --data (with castle_16 + stage3)")


if __name__ == "__main__":
    main()
