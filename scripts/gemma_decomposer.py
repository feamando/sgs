"""
Raum Path A hotswap: Gemma 4 as a DECOMPOSER (prompt -> structure-only tree).

Gate 0 of SETUP_072026_gemma.md -- the cheapest possible Gemma decomposer:
few-shot prompt the instruction-tuned base with a handful of prompt->tree
exemplars and parse the JSON it returns. NO training. If this holds the schema
reliably, we skip LoRA (SETUP_072026_gemma.md #2) and wire it straight into the
Raum decomposer app.

`GemmaDecomposer.generate_tree(prompt) -> dict` mirrors the public method of the
SGS `Decomposer` in scripts/infer_decomposer.py, so the HF backend (#3) is a
drop-in: the fill + render pipeline downstream consumes the same tree dict
(CompositionNode: name/position/scale/(rot)/(color)/children, NO gaussians).

Reuses the proven load + chat-template + JSON-extract path from
scripts/generate_trees_gemma.py (the data generator that already emits this
exact tree format), so few-shot and the later LoRA run stay format-consistent.

Usage (4090 box, main .venv with transformers>=4.50):
  python scripts/gemma_decomposer.py --model models/gemma-4-e4b-it \
    --eval scripts/assets/gemma_scene_prompts.txt \
    --out results/gemma_decomp_fewshot.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.castle_grammar import _part_kind, EXPANDABLE_PARTS
from scripts.generate_trees_gemma import parse_tree, validate_skeleton


# System instructions: same rules as the data generator's SYSTEM_PROMPT, minus
# the trailing per-scene interpolation (the scene arrives as a user turn so we
# can prepend few-shot exemplars in between).
SYSTEM_PROMPT = """You are a 3D scene-layout engine. Given a scene description, output a JSON tree that places NAMED PARTS in space. You do NOT draw geometry; a separate renderer expands each part.

Rules:
1. Root is {"name": "scene", "children": [...]}.
2. Each child has: "name", "position" [x,y,z], "scale" (float). NO "gaussians", NO nested children.
3. You may ONLY use these part names (suffix with _0,_1,... for repeats, or _n/_s/_e/_w for sides):
   %s
4. Ground goes first: use "hill" or "ground" or "cliff" at y=-0.1.
5. Place parts on top of the ground (y around 0.4-0.7), scene fits in a [-2,2] cube.
6. Compose the requested scene from these parts as best you can (a lighthouse = a tall tower; a fort = towers + walls + keep).
7. Output ONLY the JSON. No prose, no markdown fences.""" % (", ".join(EXPANDABLE_PARTS),)

USER_TMPL = "DECOMPOSE: {prompt}"

# Image variant: same schema/vocab, but the scene comes from the IMAGE. Used by
# GemmaMMDecomposer.generate_tree_from_image (gate PASSED 2026-07-27, Neuschwanstein).
IMAGE_SYSTEM_PROMPT = """You are a 3D scene-layout engine. Look at the image and output a JSON tree that places NAMED PARTS in space to reconstruct the structure shown. You do NOT draw geometry; a separate renderer expands each part.

Rules:
1. Root is {"name": "scene", "children": [...]}.
2. Each child has: "name", "position" [x,y,z], "scale" (float). NO "gaussians", NO nested children.
3. You may ONLY use these part names (suffix with _0,_1,... for repeats, or _n/_s/_e/_w for sides):
   %s
4. Ground goes first: use "hill" or "ground" or "cliff" at y=-0.1.
5. Place parts on top of the ground (y around 0.4-0.7), scene fits in a [-2,2] cube.
6. Reconstruct the building/scene in the image from these parts as best you can (match tower count, walls, gate placement to what you see).
7. Output ONLY the JSON. No prose, no markdown fences.""" % (", ".join(EXPANDABLE_PARTS),)


# PARAMETRIC prompt (Phase 1, SETUP §7): Gemma composes ANY object from a few
# primitives with explicit dimensions, instead of naming a fixed grammar part.
# This removes both the vocabulary ceiling (14 castle parts) and the geometry
# ceiling (hand-authored shape per name). Gemma acts as a blockout artist.
PARAMETRIC_SYSTEM_PROMPT = """You are a 3D blockout artist. Given a description, output a JSON tree that reconstructs the object as a composition of PARAMETRIC PRIMITIVES placed in space. A renderer rasterizes each primitive from its dimensions -- you supply the shapes, sizes, positions and colors.

Rules:
1. Root is {"name": "scene", "children": [...]}.
2. Each child is one primitive: {"shape", "position":[x,y,z], "size":[w,h,d], "color":[r,g,b]}, optional "taper" (0..1, shrinks the top).
3. "shape" is ONE of: box, cylinder, cone, sphere, dome, wedge, plane. NO other shapes, NO nested children, NO "gaussians".
4. size is the FULL extent in world units on each axis. position is the CENTER of the primitive. Do NOT also set "scale".
5. Scene fits in a [-2,2] cube. Ground/water = a wide flat box or plane at y=-0.1. Build upward (y up).
6. Compose faithfully: a ship = tapered box hull + cylinder mast + plane sail; a lighthouse = tapered cylinder tower + box lantern room + cone cap; a pagoda = stacked tapered boxes with wedge roofs. Match counts and proportions.
7. Colors are realistic RGB in 0..1.
8. Output ONLY the JSON. No prose, no markdown fences."""


def _n_leaves(tree):
    if not isinstance(tree, dict):
        return 0
    return len(tree.get("children", []) or [])


def _n_unknown_leaves(tree):
    """Leaves whose name the fill stage can't render (same test as validate_skeleton)."""
    GROUND = {"hill", "ground", "cliff"}
    bad = 0
    for c in tree.get("children", []) or []:
        if not isinstance(c, dict) or "name" not in c:
            bad += 1
            continue
        nm = c["name"].lower()
        ok = _part_kind(nm) is not None or any(nm == g or nm.startswith(g) for g in GROUND)
        if not ok:
            bad += 1
    return bad


_PARAM_SHAPES = {"box", "cylinder", "cone", "sphere", "dome", "wedge", "plane",
                 "tube", "column", "pole", "pillar", "mast", "spire", "pyramid",
                 "steeple", "ball", "orb", "slab", "cube", "prism", "gable"}


def validate_parametric(tree):
    """Validate a PARAMETRIC tree (shape+size+color primitives). Keeps only
    children with a known shape; drops others. Returns (clean_tree, n_dropped).
    Distinct from validate_skeleton (which gates on named grammar parts)."""
    if not isinstance(tree, dict) or tree.get("name") != "scene":
        return None, 0
    kids = tree.get("children", [])
    if not isinstance(kids, list) or not kids:
        return None, 0
    clean, dropped = [], 0
    for c in kids:
        if not isinstance(c, dict) or "shape" not in c:
            dropped += 1
            continue
        if str(c["shape"]).lower() not in _PARAM_SHAPES:
            dropped += 1
            continue
        c.setdefault("position", [0, 0, 0])
        c.setdefault("size", [0.3, 0.3, 0.3])
        c.setdefault("color", [0.6, 0.6, 0.6])
        # CompositionNode.from_dict (used by fill/shift/tree_to_tensors) requires
        # a "name"; parametric nodes only carry "shape", so derive one. Keep scale
        # at 1.0 -- size is already baked into the primitive's local coords, so a
        # node scale != 1 would double-apply.
        c.setdefault("name", str(c["shape"]).lower())
        c["scale"] = 1.0
        c.pop("children", None)
        c.pop("gaussians", None)
        clean.append(c)
    if not clean:
        return None, dropped
    return {"name": "scene", "children": clean}, dropped


def load_exemplars(path, n_shot):
    """Draw n_shot diverse prompt->tree exemplars from a decomposer dataset
    (path1_train.json format: [{"prompt","tree"}]). Prefers a spread: one
    castle, one non-castle, then whatever else, so the shots don't all look
    the same."""
    if not path or n_shot <= 0:
        return []
    recs = json.load(open(path))
    castle = [r for r in recs if "castle" in r["prompt"].lower()]
    other = [r for r in recs if "castle" not in r["prompt"].lower()]
    picked, seen = [], set()
    for pool in (castle, other, recs):
        for r in pool:
            if len(picked) >= n_shot:
                break
            key = r["prompt"].lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(r)
    return picked[:n_shot]


class GemmaDecomposer:
    """Few-shot Gemma 4 decomposer. Same interface as the SGS Decomposer."""

    def __init__(self, model_path, exemplars_path=None, n_shot=3,
                 max_new=1024, temperature=0.1, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new = max_new
        self.temperature = temperature
        self.exemplars = load_exemplars(exemplars_path, n_shot)
        # Interface parity with the SGS Decomposer (infer_decomposer.py): the
        # serve/fill path reads these. Gemma has no SentencePiece / SGS model,
        # so vocab_size + scan_library are None and the parse-failure debug path
        # (which pokes .sp/.model) is guarded by backend in infer_decomposer.py.
        self.last_raw = ""
        self.scan_library = None
        self.vocab_size = None

        print(f"[gemma-decomp] loading {model_path} ...")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None)
        if adapter:
            from peft import PeftModel
            print(f"[gemma-decomp] loading LoRA adapter {adapter} ...")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        print(f"[gemma-decomp] ready ({len(self.exemplars)} few-shot exemplars)")

    def _messages(self, prompt):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in self.exemplars:
            tree_json = json.dumps(ex["tree"], separators=(",", ":"))
            msgs.append({"role": "user", "content": USER_TMPL.format(prompt=ex["prompt"])})
            msgs.append({"role": "assistant", "content": tree_json})
        msgs.append({"role": "user", "content": USER_TMPL.format(prompt=prompt)})
        return msgs

    def _raw_generate(self, prompt, max_new=None, temperature=None):
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        inputs = self.tok.apply_chat_template(
            self._messages(prompt), add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4))
        text = self.tok.decode(gen[0][input_len:], skip_special_tokens=True)
        self.last_raw = text
        return text

    def generate_tree(self, prompt: str, max_new: int = None,
                      temperature: float = None, top_k: int = None,
                      retries: int = 3) -> dict | None:
        """Prompt -> validated shallow skeleton dict (or None if unparseable).

        Signature-compatible with the SGS Decomposer.generate_tree so the
        infer_decomposer.py serve handler calls it unchanged. top_k is accepted
        for parity but unused (HF generate uses temperature sampling here).
        Attempt 0 is near-greedy (most reliable for JSON); retries re-sample.
        """
        text = self._raw_generate(prompt, max_new=max_new, temperature=temperature)
        tree = parse_tree(text)
        if tree is not None:
            clean, _ = validate_skeleton(tree)
            if clean is not None:
                return self._fill(clean)
        for _ in range(max(0, retries - 1)):
            t = self.temperature if temperature is None else temperature
            text = self._raw_generate(prompt, max_new=max_new, temperature=max(t, 0.4))
            tree = parse_tree(text)
            if tree is not None:
                clean, _ = validate_skeleton(tree)
                if clean is not None:
                    return self._fill(clean)
        return None

    def _fill(self, tree: dict) -> dict:
        """Turn the validated shallow skeleton into a renderable gaussian-bearing
        tree, EXACTLY as the SGS Decomposer does (fill_gaussians + shift_above_
        ground). Without this the tree reaches tree_to_tensors with 0 gaussians
        -> empty cloud -> 'Mean of empty slice' and nothing renders. Imported
        lazily to avoid a circular import (infer_decomposer imports this module)."""
        from scripts.infer_decomposer import fill_gaussians, shift_above_ground
        fill_gaussians(tree, getattr(self, "scan_library", None))
        shift_above_ground(tree)
        return tree

    def generate_tree_verbose(self, prompt: str) -> dict:
        """Same, but returns metrics for the Gate-0 measurement."""
        text = self._raw_generate(prompt)
        raw = parse_tree(text)
        if raw is None:
            return {"prompt": prompt, "parsed": False, "tree": None,
                    "n_leaves": 0, "n_unknown": 0}
        n_leaves = _n_leaves(raw)
        n_unknown = _n_unknown_leaves(raw)
        clean, _ = validate_skeleton(raw)
        return {"prompt": prompt, "parsed": clean is not None, "tree": clean,
                "n_leaves": n_leaves, "n_unknown": n_unknown}


class GemmaMMDecomposer:
    """Unified MULTIMODAL Gemma decomposer for the Raum app: ONE model load
    serves BOTH text (generate_tree) and image (generate_tree_from_image).

    Why one class: Gemma 4 E4B is ~8B (~16GB bf16); a 4090 (24GB) cannot hold two
    copies, so the text-only GemmaDecomposer and the image path must share a
    single AutoModelForImageTextToText load. This is the class the served Raum app
    uses; standalone GemmaDecomposer stays for the Gate-0 text CLI.

    Interface-compatible with the SGS Decomposer: generate_tree(prompt, ...) plus
    .last_raw / .scan_library / .vocab_size, so infer_decomposer.py's render path
    is backend-agnostic. Adds generate_tree_from_image(path) for the multimodal
    input Planck/Hertz cannot do (gate PASSED 2026-07-27).
    """

    def __init__(self, model_path, exemplars_path=None, n_shot=3,
                 max_new=1024, temperature=0.1):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _MM
            self.model_class = "AutoModelForImageTextToText"
        except ImportError:
            from transformers import AutoModelForCausalLM as _MM
            self.model_class = "AutoModelForCausalLM (FALLBACK, image may be ignored)"

        self.torch = torch
        self.max_new = max_new
        self.temperature = temperature
        self.exemplars = load_exemplars(exemplars_path, n_shot)
        self.last_raw = ""
        self.scan_library = None
        self.vocab_size = None
        self.supports_image = True

        print(f"[gemma-mm] loading {model_path} via {self.model_class} ...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = _MM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()
        print(f"[gemma-mm] ready ({len(self.exemplars)} few-shot exemplars)")

    # ---- text turn (system + exemplars + user prompt) ----
    def _text_messages(self, prompt):
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        for ex in self.exemplars:
            tree_json = json.dumps(ex["tree"], separators=(",", ":"))
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": USER_TMPL.format(prompt=ex["prompt"])}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": tree_json}]})
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": USER_TMPL.format(prompt=prompt)}]})
        return msgs

    # ---- image turn (system + text exemplars + image) ----
    def _image_messages(self, image_path):
        msgs = [{"role": "system", "content": [{"type": "text", "text": IMAGE_SYSTEM_PROMPT}]}]
        for ex in self.exemplars:
            tree_json = json.dumps(ex["tree"], separators=(",", ":"))
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": USER_TMPL.format(prompt=ex["prompt"])}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": tree_json}]})
        msgs.append({"role": "user", "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": "Reconstruct the building in this image as a Raum scene tree."}]})
        return msgs

    def _run(self, messages, max_new=None, temperature=None):
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-4))
        text = self.processor.decode(gen[0][input_len:], skip_special_tokens=True)
        self.last_raw = text
        return text

    def _parse_fill(self, text):
        tree = parse_tree(text)
        if tree is None:
            return None
        clean, _ = validate_skeleton(tree)
        if clean is None:
            return None
        from scripts.infer_decomposer import fill_gaussians, shift_above_ground
        fill_gaussians(clean, getattr(self, "scan_library", None))
        shift_above_ground(clean)
        return clean

    def generate_tree(self, prompt: str, max_new: int = None,
                      temperature: float = None, top_k: int = None,
                      retries: int = 3) -> dict | None:
        """Text prompt -> filled tree. Signature matches the SGS Decomposer."""
        tree = self._parse_fill(self._run(self._text_messages(prompt), max_new, temperature))
        if tree is not None:
            return tree
        t = self.temperature if temperature is None else temperature
        for _ in range(max(0, retries - 1)):
            tree = self._parse_fill(self._run(self._text_messages(prompt), max_new, max(t, 0.4)))
            if tree is not None:
                return tree
        return None

    def generate_tree_from_image(self, image_path: str, max_new: int = None,
                                 temperature: float = None) -> dict | None:
        """Reference image -> filled tree. The capability Planck/Hertz cannot do."""
        return self._parse_fill(self._run(self._image_messages(image_path), max_new, temperature))

    # ---- PARAMETRIC (Phase 1): Gemma composes geometry from primitives ----
    def _parametric_messages(self, prompt, image_path=None):
        msgs = [{"role": "system", "content": [{"type": "text", "text": PARAMETRIC_SYSTEM_PROMPT}]}]
        user = []
        if image_path:
            user.append({"type": "image", "url": image_path})
            user.append({"type": "text", "text": "Reconstruct the object in this image as parametric primitives."})
        else:
            user.append({"type": "text", "text": f"Object: {prompt}"})
        msgs.append({"role": "user", "content": user})
        return msgs

    def _parse_fill_parametric(self, text):
        tree = parse_tree(text)
        if tree is None:
            return None
        clean, _ = validate_parametric(tree)
        if clean is None:
            return None
        from scripts.infer_decomposer import fill_gaussians, shift_above_ground
        fill_gaussians(clean, None)   # scans don't apply to parametric primitives
        shift_above_ground(clean)
        return clean

    def generate_parametric(self, prompt: str = None, image_path: str = None,
                            max_new: int = None, temperature: float = None,
                            retries: int = 3) -> dict | None:
        """Prompt OR image -> filled tree of parametric primitives (box/cylinder/
        cone/... with explicit sizes). The geometry lever: any object, not just
        the 14 grammar parts."""
        msgs = self._parametric_messages(prompt, image_path)
        tree = self._parse_fill_parametric(self._run(msgs, max_new, temperature))
        if tree is not None:
            return tree
        t = self.temperature if temperature is None else temperature
        for _ in range(max(0, retries - 1)):
            tree = self._parse_fill_parametric(self._run(msgs, max_new, max(t, 0.4)))
            if tree is not None:
                return tree
        return None


def main():
    p = argparse.ArgumentParser(description="Gate 0: few-shot Gemma 4 decomposer")
    p.add_argument("--model", required=True, help="local Gemma 4 path, e.g. models/gemma-4-e4b-it")
    p.add_argument("--eval", required=True, help="text file, one eval prompt per line")
    p.add_argument("--out", default="results/gemma_decomp_fewshot.json")
    p.add_argument("--exemplars", default="data/decomposition_trees/path1_train.json",
                   help="decomposer dataset to draw few-shot exemplars from")
    p.add_argument("--n-shot", type=int, default=3)
    p.add_argument("--max-new", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--dump-tree", default=None,
                   help="also write the FIRST valid tree to this path, ready for "
                        "infer_decomposer.py --scene-file (visual spot-check)")
    args = p.parse_args()

    dec = GemmaDecomposer(args.model, exemplars_path=args.exemplars,
                          n_shot=args.n_shot, max_new=args.max_new,
                          temperature=args.temperature)

    prompts = [l.strip() for l in open(args.eval) if l.strip()]
    print(f"[gemma-decomp] {len(prompts)} eval prompts")

    results, n_valid, tot_leaves, tot_unknown = [], 0, 0, 0
    for i, prompt in enumerate(prompts):
        r = dec.generate_tree_verbose(prompt)
        results.append(r)
        n_valid += int(r["parsed"])
        tot_leaves += r["n_leaves"]
        tot_unknown += r["n_unknown"]
        flag = "ok " if r["parsed"] else "FAIL"
        print(f"  [{i+1}/{len(prompts)}] {flag} {r['n_leaves']} leaves "
              f"({r['n_unknown']} unknown)  {prompt[:48]}")

    n = len(prompts)
    valid_rate = n_valid / n if n else 0.0
    vocab_rate = 1.0 - (tot_unknown / tot_leaves) if tot_leaves else 0.0
    summary = {
        "model": args.model, "n_shot": args.n_shot, "n_prompts": n,
        "json_valid_rate": round(valid_rate, 4),
        "part_vocab_rate": round(vocab_rate, 4),
        "n_valid": n_valid, "total_leaves": tot_leaves, "total_unknown": tot_unknown,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(args.out, "w"), indent=2)

    if args.dump_tree:
        first = next((r["tree"] for r in results if r["parsed"] and r["tree"]), None)
        if first is not None:
            # Fill before dumping so infer_decomposer.py --scene-file renders it
            # (scene-file loads the tree as-is; a bare skeleton renders empty).
            from scripts.infer_decomposer import fill_gaussians, shift_above_ground
            fill_gaussians(first, None)
            shift_above_ground(first)
            Path(args.dump_tree).parent.mkdir(parents=True, exist_ok=True)
            json.dump(first, open(args.dump_tree, "w"), indent=2)
            print(f"[gemma-decomp] first valid tree (filled) -> {args.dump_tree}")

    print("\n[gemma-decomp] GATE 0 ==================================")
    print(f"  JSON-valid rate : {valid_rate:.1%}  ({n_valid}/{n})")
    print(f"  part-vocab rate : {vocab_rate:.1%}  ({tot_leaves - tot_unknown}/{tot_leaves} leaves renderable)")
    print(f"  saved -> {args.out}")
    print("  PASS (>=~90% valid + coherent): skip LoRA, wire into infer_decomposer.py --backend hf")
    print("  PARTIAL: proceed to train_decomposer_gemma.py (SETUP_072026_gemma.md #2)")
    print("  recall/structure are in the summary above (leaves + unknown per prompt).")
    print("  visual spot-check one tree: extract results[i].tree to a file, then")
    print("    python scripts/infer_decomposer.py --scene-file <that_tree.json> --no-snap")


if __name__ == "__main__":
    main()
