"""
Raum 1.3 Phase 3: Inference pipeline (prompt -> tree -> render).

Takes a text prompt, feeds it to the fine-tuned decomposer, parses
the output as a composition tree JSON, and serves it via the scene
viewer.

Usage:
    python scripts/infer_decomposer.py `
      --checkpoint checkpoints/planck_decomposer/best.pt `
      --tokenizer data/wikipedia/tokenizer.model `
      --prompt "a castle on a hill"

    python scripts/infer_decomposer.py `
      --checkpoint checkpoints/planck_decomposer/best.pt `
      --tokenizer data/wikipedia/tokenizer.model `
      --serve --port 8003
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.raum.decomposition import CompositionNode, tree_to_tensors, print_tree, save_tree


def parse_args():
    p = argparse.ArgumentParser(description="Raum 1.3/1.4 inference: prompt -> tree -> render")
    p.add_argument("--checkpoint", help="Fine-tuned decomposer checkpoint (not needed with --scene-file)")
    p.add_argument("--tokenizer", help="SentencePiece model (not needed with --scene-file)")
    p.add_argument("--scene-file", type=str, default=None,
                   help="Raum 0.5: render a fixed pre-built scene tree (JSON) with no model in the loop")
    p.add_argument("--prompt", type=str, default=None, help="Single prompt to decompose")
    p.add_argument("--serve", action="store_true", help="Launch web UI for interactive prompts")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-new", type=int, default=4096, help="Max tokens to generate for tree")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Near-greedy for structured JSON; higher temps derail mid-tree")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--fidelity", choices=["low", "high"], default="low",
                   help="low=Raum 1.3 skeleton, high=subdivision+densification+refinement")
    p.add_argument("--refine-mode", choices=["sgs", "multiview", "none"], default="sgs",
                   help="Refinement mode for high fidelity (default: sgs)")
    p.add_argument("--templates", default="data/architecture_gs",
                   help="Template library path for SGS refinement")
    return p.parse_args()


class Decomposer:
    """Wraps the fine-tuned Planck model for tree generation."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: torch.device):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.device = device

        from src.sgs_lm import SGSLanguageModel, migrate_state_dict
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        state = migrate_state_dict(state)
        vocab_size = state["tok_mu.weight"].shape[0]

        self.model = SGSLanguageModel(vocab_size=vocab_size)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def generate_tree(self, prompt: str, max_new: int = 4096,
                      temperature: float = 0.1, top_k: int = 3,
                      retries: int = 3) -> dict | None:
        """Generate + parse a tree, retrying on parse failure.

        Structured JSON is near-deterministic, so attempt 1 is GREEDY (the most
        reliable). If the model derails into garbage and parsing fails, retry
        with the requested sampling (which varies the path and usually recovers
        on the next try). Returns the parsed tree, or None after `retries`.
        """
        attempt0 = self._generate_once(prompt, max_new, temperature=0.0, top_k=1)
        if attempt0 is not None:
            return attempt0
        for _ in range(max(0, retries - 1)):
            tree = self._generate_once(prompt, max_new,
                                       temperature=max(temperature, 0.2), top_k=max(top_k, 5))
            if tree is not None:
                return tree
        return None

    @torch.no_grad()
    def _generate_once(self, prompt: str, max_new: int = 4096,
                       temperature: float = 0.1, top_k: int = 3) -> dict | None:
        """
        Generate a composition tree from a text prompt (single attempt).

        The model generates a STRUCTURE-ONLY tree (names, positions, scales,
        children) without gaussians. Gaussians are filled in procedurally
        at render time based on node names. This keeps output within the
        512-token context window.

        Returns the parsed tree dict, or None if parsing fails.
        """
        input_text = f"DECOMPOSE: {prompt}\nTREE: "
        input_ids = self.sp.encode(input_text, out_type=int)

        ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = []

        for step in range(max_new):
            if ids_t.shape[1] > 512:
                ids_t = ids_t[:, -512:]

            logits = self.model(ids_t)
            raw_logits = logits[0, -1, :]

            if temperature <= 0.0:
                # greedy: most reliable for structured JSON
                next_id = int(raw_logits.argmax().item())
            else:
                next_logits = raw_logits / temperature
                if top_k > 0:
                    topk_vals, topk_idx = next_logits.topk(top_k)
                    mask = torch.full_like(next_logits, float("-inf"))
                    mask.scatter_(0, topk_idx, topk_vals)
                    next_logits = mask
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            if next_id == self.sp.eos_id():
                break

            generated.append(next_id)
            ids_t = torch.cat([ids_t, torch.tensor([[next_id]], device=self.device)], dim=1)

            # Early stop: the model often does not emit EOS and would run to
            # max_new, appending garbage after a valid tree that then truncates
            # unparseably. Once the first top-level JSON object is balanced
            # (brace depth back to 0 after the opening brace), stop. Check
            # periodically since decode is O(n).
            if step > 8 and step % 16 == 0:
                txt = self.sp.decode(generated)
                s = txt.find("{")
                if s != -1:
                    depth = 0
                    for ch in txt[s:]:
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                break
                    if depth == 0:
                        break

        # Decode and parse JSON
        output_text = self.sp.decode(generated)
        self.last_raw = output_text

        # Debug: always print raw output
        print(f"  Raw ({len(generated)} tokens): {output_text[:300]}...")

        # Try to extract JSON from the output
        tree = self._parse_tree_json(output_text)

        # If we got a tree, fill in gaussians procedurally for leaf nodes
        if tree:
            self._fill_gaussians(tree)
            self._shift_above_ground(tree)

        return tree

    def _shift_above_ground(self, tree: dict):
        """Shift the entire scene so all Gaussians sit above Y=0."""
        from src.raum.decomposition import CompositionNode, tree_to_tensors
        node = CompositionNode.from_dict(tree)
        tensors = tree_to_tensors(node)
        if tensors["means"].shape[0] == 0:
            return
        min_y = tensors["means"][:, 1].min().item()
        if min_y < 0:
            # Shift root position up
            pos = tree.get("position", [0, 0, 0])
            tree["position"] = [pos[0], pos[1] - min_y + 0.1, pos[2]]

    def _fill_gaussians(self, node: dict):
        """Recursively fill leaf nodes with procedural Gaussians based on name."""
        import math, random

        if "children" in node and node["children"]:
            for child in node["children"]:
                self._fill_gaussians(child)
            return

        # Raum 1.5: expand a shallow grammar PART leaf (tower/wall/keep/tree/
        # gate) into its atomic compound, so a model that emits a shallow
        # skeleton renders identically to the 0.5 grammar. Falls through to the
        # generic per-name fill below for non-part leaves.
        if "gaussians" not in node or not node.get("gaussians"):
            try:
                from scripts.castle_grammar import expand_part
                part = expand_part(node.get("name", ""), color=node.get("color"))
            except Exception:
                part = None
            if part is not None:
                pd = part.to_dict()
                # Lift the expanded compound's sub-parts directly under this
                # node (avoid a redundant tower_NE -> tower_NE nesting).
                if pd.get("children"):
                    node["children"] = pd["children"]
                else:
                    node["gaussians"] = pd.get("gaussians", [])
                for child in node.get("children", []):
                    self._fill_gaussians(child)
                return

        if "gaussians" not in node or not node.get("gaussians"):
            # Leaf node without gaussians: generate procedurally
            name = node.get("name", "").lower()
            color = node.get("color", [0.6, 0.6, 0.6])

            # Pick primitive type from name.
            n = 60
            gaussians = []

            # Suffix-aware routing for grammar compound names (Raum 0.5/1.5).
            # The semantic part is the trailing token: tower_NE_roof -> "roof",
            # keep_body -> "body" (prefix "keep"). This must run BEFORE the
            # substring chain, or "tower"/"keep" in the prefix would wrongly
            # win (e.g. tower_NE_roof matching "tower" -> cylinder).
            toks = name.replace("-", "_").split("_")
            suffix = toks[-1] if toks else name
            prefix = toks[0] if toks else name
            grammar_shape = None
            if suffix in ("roof", "canopy", "cap", "cone"):
                grammar_shape = "cone"
            elif suffix in ("trunk",):
                grammar_shape = "cylinder"
            elif suffix in ("crenellation", "battlement", "merlon", "face",
                            "arch", "brick", "stone"):
                grammar_shape = "box"
            elif suffix == "body":
                # tower bodies are round; keep / building bodies are boxes
                grammar_shape = "cylinder" if prefix in ("tower", "turret", "spire") else "box"

            if grammar_shape == "cone":
                for i in range(n):
                    t = i/n
                    theta = 2*math.pi*(i*7)/n
                    r = 0.3*(1-t)
                    gaussians.append({"position": [r*math.cos(theta), t*0.5, r*math.sin(theta)],
                                      "scale": [-3.2, -3.2, -3.2], "opacity": 2.0, "color": color})
                node["gaussians"] = gaussians
                return
            elif grammar_shape == "cylinder":
                for i in range(n):
                    theta = 2*math.pi*i/n
                    y = (i/n) * 1.0 - 0.5
                    gaussians.append({"position": [0.25*math.cos(theta), y, 0.25*math.sin(theta)],
                                      "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})
                node["gaussians"] = gaussians
                return
            elif grammar_shape == "box":
                side = max(3, round(n ** (1.0/3.0)))
                for ix in range(side):
                    for iy in range(side):
                        for iz in range(side):
                            x = (ix/(side-1) - 0.5) * 0.8
                            y = (iy/(side-1) - 0.5) * 0.8
                            z = (iz/(side-1) - 0.5) * 0.8
                            gaussians.append({"position": [x, y, z],
                                              "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})
                node["gaussians"] = gaussians
                return

            if any(w in name for w in ["ground", "floor", "plane", "field", "road",
                                       "water", "sand", "snow",
                                       # open spaces read as a flat slab
                                       "courtyard", "yard", "plaza", "terrace",
                                       "patio", "square", "base", "foundation",
                                       "platform"]):
                # Flat plane
                side = int(math.sqrt(n))
                for i in range(side):
                    for j in range(side):
                        x = (i/side - 0.5) * 2.0
                        z = (j/side - 0.5) * 2.0
                        gaussians.append({"position": [x, random.uniform(-0.02, 0.02), z],
                                          "scale": [-2.5, -2.5, -2.5], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["hill", "mound", "knoll", "dune", "mountain",
                                         "dome", "rise"]):
                # Dome / mound (hemisphere bulging up from y=0)
                golden = (1+math.sqrt(5))/2
                for i in range(n):
                    theta = 2*math.pi*i/golden
                    # upper hemisphere only
                    phi = math.acos(1 - (i+0.5)/n)
                    r = 0.8
                    gaussians.append({"position": [r*math.sin(phi)*math.cos(theta),
                                                   r*math.cos(phi) - 0.1,
                                                   r*math.sin(phi)*math.sin(theta)],
                                      "scale": [-2.6, -2.6, -2.6], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["tower", "trunk", "pole", "post", "column",
                                         "pillar", "mast", "chimney", "turret",
                                         "spire", "keep", "minaret"]):
                # Cylinder
                for i in range(n):
                    theta = 2*math.pi*i/n
                    y = (i/n) * 1.0 - 0.5
                    gaussians.append({"position": [0.25*math.cos(theta), y, 0.25*math.sin(theta)],
                                      "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["roof", "cone", "top", "canopy", "cap"]):
                # Cone
                for i in range(n):
                    t = i/n
                    theta = 2*math.pi*(i*7)/n
                    r = 0.3*(1-t)
                    gaussians.append({"position": [r*math.cos(theta), t*0.5, r*math.sin(theta)],
                                      "scale": [-3.2, -3.2, -3.2], "opacity": 2.0, "color": color})
            elif any(w in name for w in ["wall", "fence", "gate", "door", "box",
                                         "body", "hull", "block", "brick",
                                         # building bodies are solid boxes
                                         "building", "wing", "hall", "house",
                                         "palace", "castle", "manor", "mansion",
                                         "structure", "room", "hut", "cabin",
                                         "barn", "shed", "fort", "annex", "main"]):
                # Solid box (filled grid, not a shell, so it reads as mass)
                side = max(3, round(n ** (1.0/3.0)))
                for ix in range(side):
                    for iy in range(side):
                        for iz in range(side):
                            x = (ix/(side-1) - 0.5) * 0.8
                            y = (iy/(side-1) - 0.5) * 0.8
                            z = (iz/(side-1) - 0.5) * 0.8
                            gaussians.append({"position": [x, y, z],
                                              "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})
            else:
                # Default: sphere
                golden = (1+math.sqrt(5))/2
                for i in range(n):
                    theta = 2*math.pi*i/golden
                    phi = math.acos(1-2*(i+0.5)/n)
                    r = 0.3
                    gaussians.append({"position": [r*math.sin(phi)*math.cos(theta), r*math.sin(phi)*math.sin(theta), r*math.cos(phi)],
                                      "scale": [-3.0, -3.0, -3.0], "opacity": 2.0, "color": color})

            node["gaussians"] = gaussians

    @staticmethod
    def _sanitize_json(s: str) -> str:
        """Repair common model JSON malformations before parsing.

        At low temperature the model occasionally emits trailing commas
        (`[0,0,]`, `{...,}`) or bare/partial numbers (`0.`, `-.`) that
        json.loads rejects even when the structure is otherwise complete.
        """
        import re
        # bare decimal point: 0. -> 0.0,  .5 -> 0.5,  -. -> 0
        s = re.sub(r"(?<![\d.])-?\.(?![\d])", "0", s)   # lone "." or "-."
        s = re.sub(r"(\d)\.(?=[,\]\}\s])", r"\1.0", s)   # "0." -> "0.0"
        s = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", s)      # ".5" -> "0.5"
        # non-standard literals
        s = re.sub(r"\bNaN\b|\bInfinity\b|\b-Infinity\b", "0", s)
        # collapse repeated commas (`,,` -> `,`)
        s = re.sub(r",\s*,+", ",", s)
        # leading commas right after an opening bracket
        s = re.sub(r"([\[{])\s*,", r"\1", s)
        # trailing commas before a closing ] or } (after the above)
        s = re.sub(r",\s*([\]}])", r"\1", s)
        return s

    @staticmethod
    def _recover_json(text: str) -> dict | None:
        """Structural recovery: scan the text as JSON, tracking bracket depth
        and string state, and close everything at the last position where the
        structure was valid. Robust to truncation anywhere (mid-array,
        mid-value, mid-key) regardless of the specific malformation.
        """
        start = text.find("{")
        if start == -1:
            return None
        s = text
        # Walk char-by-char tracking string state and the bracket stack. Record
        # a "safe prefix" snapshot every time we are cleanly positioned AFTER a
        # complete element and BEFORE the next key/value -- i.e. right after a
        # closing } or ], or right after a separating comma. At those points the
        # text so far, plus the closers for the open brackets, is valid JSON.
        # Truncation anywhere later (mid-key, mid-string, mid-number) just falls
        # back to the last safe snapshot.
        stack, in_str, esc = [], False, False
        best = None  # (cut_index, closers_string)
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                    # right after an opening bracket, an empty container is a
                    # valid (if minimal) cut point
                    best = (i + 1, "".join(reversed(stack)))
                elif ch in "}]":
                    if stack:
                        stack.pop()
                    # after a complete value/container -> safe boundary
                    best = (i + 1, "".join(reversed(stack)))
                elif ch == ",":
                    # comma at array level (next char is a value, not a key) is
                    # a safe cut: drop the comma and close. Only when the
                    # enclosing container is an array.
                    if stack and stack[-1] == "]":
                        best = (i, "".join(reversed(stack)))
        if best is None:
            return None
        cut, closers = best
        closed = s[start:cut] + closers
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            return None

    def _parse_tree_json(self, text: str) -> dict | None:
        """Attempt to parse a composition tree from generated text."""
        # Try direct parse, then a sanitized parse
        for candidate in (text, self._sanitize_json(text)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in the text
        start = text.find("{")
        if start == -1:
            return None

        # Find matching closing brace; on parse failure, fall through to repair
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    sub = text[start:i+1]
                    for candidate in (sub, self._sanitize_json(sub)):
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                    break  # balanced but unparseable -> try the repair path below

        # JSON was truncated (model hit max tokens). Try to repair by
        # closing open brackets/braces.
        truncated = self._sanitize_json(text[start:])
        # Count unclosed brackets
        open_braces = truncated.count("{") - truncated.count("}")
        open_brackets = truncated.count("[") - truncated.count("]")

        # Truncate at last complete object/array entry
        # Find the last comma or complete value
        repair = truncated.rstrip()
        if repair.endswith(","):
            repair = repair[:-1]

        # Close all open brackets
        repair += "]" * open_brackets + "}" * open_braces

        try:
            return json.loads(repair)
        except json.JSONDecodeError:
            # More aggressive: truncate to last complete gaussians entry
            # Find last complete "}" before the truncation
            last_close = repair.rfind("}")
            if last_close > 0:
                # Try progressively shorter substrings
                for cut in range(last_close, max(last_close - 200, 0), -1):
                    if repair[cut] == "}":
                        attempt = repair[:cut+1]
                        ob = attempt.count("{") - attempt.count("}")
                        obrk = attempt.count("[") - attempt.count("]")
                        attempt += "]" * obrk + "}" * ob
                        try:
                            return json.loads(self._sanitize_json(attempt))
                        except json.JSONDecodeError:
                            continue
            # Last resort: structural recovery on the sanitized text.
            return self._recover_json(self._sanitize_json(text))


def validate_tree(tree_dict: dict) -> tuple[dict, dict]:
    """Raum 1.6 grammar-validated decoding.

    Walk the generated tree and ensure every leaf name is renderable: it must
    either be an expandable grammar part (tower/wall/gate/...) or hit a known
    generic-fill keyword. Leaves that match nothing would silently become a
    default sphere, so we drop them and report. Returns (clean_tree, report).
    """
    try:
        from scripts.castle_grammar import _part_kind
    except Exception:
        _part_kind = lambda n: None

    GENERIC = ("ground", "floor", "plane", "field", "road", "water", "sand",
               "snow", "courtyard", "yard", "plaza", "terrace", "patio",
               "square", "base", "foundation", "platform", "hill", "mound",
               "knoll", "dune", "mountain", "dome", "rise", "tower", "trunk",
               "pole", "post", "column", "pillar", "mast", "chimney", "turret",
               "spire", "keep", "minaret", "roof", "cone", "top", "canopy",
               "cap", "wall", "fence", "gate", "door", "box", "body", "hull",
               "block", "brick", "building", "wing", "hall", "house", "palace",
               "castle", "scene")
    report = {"dropped": [], "kept_leaves": 0, "total_nodes": 0}

    def known(name: str) -> bool:
        n = (name or "").lower()
        if _part_kind(n) is not None:
            return True
        return any(w in n for w in GENERIC)

    def clean(node: dict) -> dict | None:
        report["total_nodes"] += 1
        kids = node.get("children")
        if kids:
            cleaned = [c for c in (clean(k) for k in kids) if c is not None]
            node["children"] = cleaned
            # an internal node with all children dropped becomes a bare leaf;
            # keep it only if its own name is renderable
            if not cleaned and not known(node.get("name", "")):
                report["dropped"].append(node.get("name", "?"))
                return None
            return node
        # leaf
        if known(node.get("name", "")) or node.get("gaussians"):
            report["kept_leaves"] += 1
            return node
        report["dropped"].append(node.get("name", "?"))
        return None

    clean_tree = clean(tree_dict) or {"name": "scene", "children": []}
    snap_layout(clean_tree, report)
    return clean_tree, report


# Canonical grammar layout (mirrors castle_grammar.build_castle_on_hill).
_RING = 0.7
_CASTLE_LAYOUT = {
    "tower_sw": [-_RING, 0, -_RING], "tower_se": [_RING, 0, -_RING],
    "tower_ne": [_RING, 0, _RING], "tower_nw": [-_RING, 0, _RING],
    "wall_s": [0, 0, -_RING], "wall_n": [0, 0, _RING],
    "wall_e": [_RING, 0, 0], "wall_w": [-_RING, 0, 0],
    "keep": [0, 0, 0],
}
_CASTLE_ROT = {"wall_e": [0.7071, 0, 0.7071, 0], "wall_w": [0.7071, 0, 0.7071, 0]}


def snap_layout(tree: dict, report: dict):
    """Raum 1.6: snap named parts to their canonical grammar positions.

    The model is reliable about WHICH elements exist but noisy about WHERE.
    Inside a castle node, snap corner towers to the wall-ring corners, walls to
    the face midpoints, and the keep to center -- removing the float/scatter so
    the layout matches the deterministic 0.5 grammar. Standalone scenes are left
    as the model placed them.
    """
    snapped = [0]

    def walk(node: dict):
        if node.get("name", "").lower() == "castle" and node.get("children"):
            for c in node["children"]:
                key = c.get("name", "").lower()
                if key in _CASTLE_LAYOUT:
                    c["position"] = list(_CASTLE_LAYOUT[key])
                    if key in _CASTLE_ROT:
                        c["rotation"] = list(_CASTLE_ROT[key])
                    snapped[0] += 1
        for c in node.get("children", []) or []:
            walk(c)

    walk(tree)
    report["snapped"] = snapped[0]


def apply_high_fidelity(tree, refine_mode="sgs", templates_dir="data/architecture_gs",
                        prebuilt=False):
    """Apply subdivision + densification + optional refinement to a composition tree.

    prebuilt=True (Raum 0.5): the tree already has dense ATOMIC leaves from the
    grammar (stones, crenellations), so subdivision is skipped. Subdividing
    atomic parts only inflates them into redundant 64-point blobs. Densify then
    only fills genuine gaps.
    """
    from scripts.subdivide_scene import subdivide_tree, set_templates_dir
    from src.raum.densify import DensifyConfig, densify_loop

    # Step 1: Subdivide (60 -> ~5K-13K) using templates when available.
    # Skipped for prebuilt grammar scenes whose leaves are already atomic.
    if prebuilt:
        tensors = tree_to_tensors(tree)
    else:
        tpl_path = Path(templates_dir) if templates_dir else None
        set_templates_dir(tpl_path)
        tree = subdivide_tree(tree, n_children=12)
        tensors = tree_to_tensors(tree)
    n_sub = tensors["means"].shape[0]

    # Step 2: Densify (5K -> ~50K)
    config = DensifyConfig(grad_threshold=0.0002, max_gaussians=60000)
    tensors = densify_loop(tensors, n_iterations=30, config=config)
    n_dense = tensors["means"].shape[0]

    # Step 3: Refine (optional)
    if refine_mode == "sgs":
        from scripts.refine_scene import refine_sgs
        tensors = refine_sgs(tensors, Path(templates_dir), n_iterations=50, lr=5e-4)
    elif refine_mode == "multiview":
        from scripts.refine_scene import refine_multiview
        tensors = refine_multiview(tensors, n_iterations=50, n_views=8, lr=1e-4)

    return tensors, {"n_subdivided": n_sub, "n_densified": n_dense, "n_refined": tensors["means"].shape[0]}


VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"/><title>Raum 1.3: Recursive Decomposition</title>
<style>
* { box-sizing: border-box; margin: 0; }
body { background: #0a0a0f; color: #f5f1e8; font-family: Inter, sans-serif;
  display: grid; grid-template-rows: auto 1fr; height: 100vh; overflow: hidden; }
header { padding: 12px 20px; border-bottom: 1px solid #1f1f2a;
  display: flex; align-items: center; justify-content: space-between; }
header h1 { font-size: 15px; font-weight: 700; }
header .status { font-size: 11px; color: #8a8598; font-family: monospace; }
main { display: grid; grid-template-columns: 320px 1fr; overflow: hidden; }
.sidebar { padding: 16px; border-right: 1px solid #1f1f2a; overflow-y: auto; }
.sidebar label { font-size: 11px; color: #8a8598; display: block; margin-bottom: 6px; }
.sidebar textarea { width: 100%; height: 60px; background: #12121a; border: 1px solid #1f1f2a;
  color: #f5f1e8; border-radius: 6px; padding: 8px; font-size: 13px; resize: vertical; font-family: inherit; }
.sidebar textarea:focus { border-color: #ffb347; outline: none; }
.sidebar button { width: 100%; margin-top: 10px; padding: 10px; font-size: 13px; font-weight: 600;
  background: #ffb347; color: #0a0a0f; border: none; border-radius: 6px; cursor: pointer; }
.sidebar button:hover { filter: brightness(1.1); }
.sidebar button:disabled { background: #3a3a48; color: #8a8598; cursor: wait; }
.tree-output { margin-top: 16px; padding-top: 12px; border-top: 1px solid #1f1f2a; }
.tree-output pre { font-size: 10px; color: #8a8598; white-space: pre-wrap; max-height: 300px; overflow-y: auto;
  background: #12121a; padding: 8px; border-radius: 6px; }
.tree-output .stats { font-size: 11px; color: #ffb347; margin-bottom: 8px; }
#viewer { position: relative; }
#info { position: absolute; top: 10px; right: 10px; font-size: 11px; color: #8a8598;
  background: rgba(18,18,26,0.9); padding: 8px 12px; border-radius: 6px; }
</style></head><body>
<header>
  <h1>Raum 1.4 / High-Fidelity Scene Generation</h1>
  <span class="status" id="status">ready</span>
</header>
<main>
  <div class="sidebar">
    <label>Scene prompt</label>
    <textarea id="prompt" placeholder="a castle on a hill">a castle on a hill</textarea>
    <label style="margin-top:10px">Fidelity</label>
    <select id="fidelity" style="width:100%;padding:6px;background:#12121a;border:1px solid #1f1f2a;color:#f5f1e8;border-radius:6px;font-size:12px">
      <option value="low">Low (skeleton, ~60 splats, instant)</option>
      <option value="high" selected>High (subdivide+densify+refine, ~50K splats, ~10s)</option>
    </select>
    <label style="margin-top:8px">Refinement mode</label>
    <select id="refine-mode" style="width:100%;padding:6px;background:#12121a;border:1px solid #1f1f2a;color:#f5f1e8;border-radius:6px;font-size:12px">
      <option value="sgs" selected>SGS Native (template Chamfer)</option>
      <option value="multiview">Multi-view consistency</option>
      <option value="none">None (densify only)</option>
    </select>
    <button id="generate">Decompose + Render</button>
    <button id="export-btn" style="margin-top:6px;background:#1f1f2a;color:#ffb347;border:1px solid #ffb347" disabled>Export .ply</button>
    <div class="tree-output" id="tree-panel">
      <div class="stats" id="stats"></div>
      <pre id="tree-text"></pre>
    </div>
  </div>
  <div id="viewer">
    <div id="info"></div>
  </div>
</main>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const viewer = document.getElementById('viewer');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, viewer.clientWidth/viewer.clientHeight, 0.1, 100);
camera.position.set(3, 3, 5);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewer.clientWidth, viewer.clientHeight);
renderer.setClearColor(0x0a0a0f);
viewer.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.GridHelper(10, 20, 0x1f1f2a, 0x1f1f2a));
scene.add(new THREE.AxesHelper(2));

// Lighting for the lit-ellipsoid renderer (Raum 0.6): a key sun, soft fill,
// and sky/ground hemisphere so geometry reads with shading and fake AO.
const sun = new THREE.DirectionalLight(0xfff4e6, 2.0);
sun.position.set(4, 8, 5);
scene.add(sun);
scene.add(new THREE.DirectionalLight(0xaecbff, 0.5).position.set(-5, 2, -4));
scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x2a2418, 0.7));
scene.add(new THREE.AmbientLight(0x404040, 0.4));

let points = null;

// Raum 0.6: render each Gaussian as a lit, ORIENTED ELLIPSOID instance, not a
// screen-aligned round disc. Per-instance matrix = translate * quaternion *
// per-axis scale, so flat slab-stones read as masonry and the scene has true
// 3D shading + fake AO from the lights. InstancedMesh keeps ~50-100K cheap.
const _unitSphere = new THREE.SphereGeometry(1.0, 6, 4);  // low-poly; scaled per instance
const _m4 = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _t = new THREE.Vector3();
const _s = new THREE.Vector3();
const _col = new THREE.Color();

function renderSplats(data) {
  if (points) { scene.remove(points); points.geometry?.dispose?.(); points.material?.dispose?.(); }
  const { means, colors, scales, rotations, n_splats } = data;

  // NB: do NOT set vertexColors -- InstancedMesh per-instance color comes from
  // setColorAt/instanceColor, which THREE injects automatically. vertexColors
  // would make it look for a (nonexistent) per-vertex color attribute -> black.
  const mat = new THREE.MeshLambertMaterial();
  const mesh = new THREE.InstancedMesh(_unitSphere, mat, n_splats);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  for (let i = 0; i < n_splats; i++) {
    _t.set(means[i][0], means[i][1], means[i][2]);
    // log-scale -> world radius per axis; 1.6x so neighbours overlap into a
    // continuous surface, clamped so no single splat balloons.
    const sl = scales[i];
    _s.set(
      Math.min(Math.exp(sl[0]) * 1.6, 0.25),
      Math.min(Math.exp(sl[1]) * 1.6, 0.25),
      Math.min(Math.exp(sl[2]) * 1.6, 0.25)
    );
    // quaternion stored [w,x,y,z]; THREE wants (x,y,z,w)
    if (rotations && rotations[i]) {
      const r = rotations[i];
      _q.set(r[1], r[2], r[3], r[0]);
    } else { _q.set(0, 0, 0, 1); }
    _m4.compose(_t, _q, _s);
    mesh.setMatrixAt(i, _m4);
    _col.setRGB(colors[i][0], colors[i][1], colors[i][2]);
    mesh.setColorAt(i, _col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  points = mesh;
  scene.add(points);
  document.getElementById('info').textContent = `${n_splats} gaussians`;
}

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();

const btn = document.getElementById('generate');
const exportBtn = document.getElementById('export-btn');
const promptEl = document.getElementById('prompt');
const fidelityEl = document.getElementById('fidelity');
const refineModeEl = document.getElementById('refine-mode');
const statusEl = document.getElementById('status');
const statsEl = document.getElementById('stats');
const treeEl = document.getElementById('tree-text');

btn.addEventListener('click', async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  btn.disabled = true;
  exportBtn.disabled = true;
  const fidelity = fidelityEl.value;
  const refineMode = refineModeEl.value;
  statusEl.textContent = fidelity === 'high' ? 'generating (high fidelity, ~10s)...' : 'decomposing...';
  try {
    const r = await fetch('/decompose', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({prompt, fidelity, refine_mode: refineMode}),
    });
    const data = await r.json();
    if (data.error) {
      statusEl.textContent = 'error: ' + data.error;
      treeEl.textContent = data.raw_output || '';
      return;
    }
    renderSplats(data.splats);
    let stats = `${data.n_gaussians} gaussians`;
    if (data.pipeline) stats += ` | ${data.pipeline}`;
    else stats += ` | depth ${data.depth} | ${data.n_children} top-level parts`;
    statsEl.textContent = stats;
    treeEl.textContent = JSON.stringify(data.tree, null, 2);
    statusEl.textContent = 'ready';
    exportBtn.disabled = false;
  } catch(e) {
    statusEl.textContent = 'error: ' + e.message;
  } finally { btn.disabled = false; }
});

exportBtn.addEventListener('click', async () => {
  statusEl.textContent = 'exporting .ply...';
  try {
    const r = await fetch('/export_ply');
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'scene.ply'; a.click();
    URL.revokeObjectURL(url);
    statusEl.textContent = 'ready';
  } catch(e) { statusEl.textContent = 'export error: ' + e.message; }
});

window.addEventListener('resize', () => {
  camera.aspect = viewer.clientWidth/viewer.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(viewer.clientWidth, viewer.clientHeight);
});
</script></body></html>"""


def _load_scene_tree(path):
    """Raum 0.5: load a pre-built composition tree from JSON (no model)."""
    with open(path) as f:
        return CompositionNode.from_dict(json.load(f))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Raum 0.5: fixed-scene mode. Skip the model entirely; the tree is the
    # grammar's output, fed straight into fill -> densify -> refine -> render.
    scene_mode = args.scene_file is not None
    decomposer = None
    if scene_mode:
        print(f"Fixed-scene mode (no model): {args.scene_file}")
    else:
        if not args.checkpoint or not args.tokenizer:
            raise SystemExit("--checkpoint and --tokenizer are required unless --scene-file is given")
        print(f"Loading decomposer: {args.checkpoint}")
        decomposer = Decomposer(args.checkpoint, args.tokenizer, device)
        print(f"  Vocab: {decomposer.vocab_size}, ready.")

    # Fixed-scene single render: load, run pipeline, save, exit.
    if scene_mode and not args.serve:
        tree = _load_scene_tree(args.scene_file)
        print_tree(tree)
        if args.fidelity == "high":
            print(f"\n  High fidelity: densify -> refine ({args.refine_mode}), subdivision skipped (atomic parts)...")
            tensors, info = apply_high_fidelity(
                tree, refine_mode=args.refine_mode, templates_dir=args.templates,
                prebuilt=True)
            print(f"  Pipeline: parts:{info['n_subdivided']} -> dense:{info['n_densified']} -> refine:{info['n_refined']}")
        else:
            tensors = tree_to_tensors(tree)
        out = Path("data/scenes/generated_scene.json")
        save_tree(tree, out)
        print(f"  Gaussians: {tensors['means'].shape[0]}  |  saved tree: {out}")
        return

    if args.prompt and not args.serve:
        # Single inference
        print(f"\nPrompt: {args.prompt}")
        print("Generating tree...")
        tree_dict = decomposer.generate_tree(
            args.prompt, max_new=args.max_new,
            temperature=args.temperature, top_k=args.top_k,
        )
        if tree_dict is None:
            # Debug: show what was generated
            input_text = f"DECOMPOSE: {args.prompt}\nTREE: "
            input_ids = decomposer.sp.encode(input_text, out_type=int)
            ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            generated = []
            with torch.no_grad():
                for _ in range(500):
                    if ids_t.shape[1] > 512:
                        ids_t = ids_t[:, -512:]
                    logits = decomposer.model(ids_t)
                    next_logits = logits[0, -1, :] / args.temperature
                    topk_vals, topk_idx = next_logits.topk(args.top_k)
                    mask_t = torch.full_like(next_logits, float("-inf"))
                    mask_t.scatter_(0, topk_idx, topk_vals)
                    probs = torch.softmax(mask_t, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    if next_id == decomposer.sp.eos_id():
                        break
                    generated.append(next_id)
                    ids_t = torch.cat([ids_t, torch.tensor([[next_id]], device=device)], dim=1)
            raw = decomposer.sp.decode(generated)
            print(f"ERROR: failed to generate valid JSON tree")
            print(f"\nRaw output (first 500 chars):")
            print(raw[:500])
            sys.exit(1)

        tree = CompositionNode.from_dict(tree_dict)
        print(f"\n=== Composition Tree ===")
        print_tree(tree)

        if args.fidelity == "high":
            print(f"\n  High fidelity: subdivide -> densify -> refine ({args.refine_mode})...")
            tensors, info = apply_high_fidelity(
                tree, refine_mode=args.refine_mode,
                templates_dir=args.templates,
            )
            print(f"  Pipeline: sub:{info['n_subdivided']} -> dense:{info['n_densified']} -> refine:{info['n_refined']}")
        else:
            tensors = tree_to_tensors(tree)
        print(f"\n  Gaussians: {tensors['means'].shape[0]}")

        # Save
        out_path = Path("data/scenes/generated_scene.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(tree_dict, f, indent=2)
        print(f"  Saved: {out_path}")
        print(f"\nView: python scripts/render_scene.py --scene {out_path}")

    elif args.serve:
        # Web UI mode
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, Response
        from pydantic import BaseModel
        import tempfile

        app = FastAPI()
        last_tensors = {}  # store for export

        class DecomposeRequest(BaseModel):
            prompt: str
            fidelity: str = "low"
            refine_mode: str = "sgs"

        @app.get("/", response_class=HTMLResponse)
        def index():
            return VIEWER_HTML

        @app.post("/decompose")
        def decompose(req: DecomposeRequest):
            nonlocal last_tensors

            # Raum 0.5 fixed-scene mode: ignore the prompt, render the grammar
            # tree. Lets the existing UI button render the fixed scene.
            if scene_mode:
                tree = _load_scene_tree(args.scene_file)
                tree_dict = tree.to_dict()
            else:
                prompt = req.prompt.strip()
                if not prompt:
                    return JSONResponse({"error": "empty prompt"})

                tree_dict = decomposer.generate_tree(
                    prompt, max_new=args.max_new,
                    temperature=args.temperature, top_k=args.top_k,
                )

                if tree_dict is None:
                    # dump the raw model output so the exact malformation is
                    # inspectable instead of lost
                    try:
                        Path("data/scenes").mkdir(parents=True, exist_ok=True)
                        dbg = Path("data/scenes/last_parse_failure.txt")
                        dbg.write_text(decomposer.last_raw or "", encoding="utf-8")
                        print(f"  parse FAILED; raw dumped to {dbg}", file=sys.stderr)
                    except Exception:
                        pass
                    return JSONResponse({"error": "failed to parse tree JSON", "raw_output": ""})

                # Raum 1.6: grammar-validated decoding -- drop unrenderable leaves.
                tree_dict, vreport = validate_tree(tree_dict)
                if vreport["dropped"]:
                    print(f"  validate: dropped {len(vreport['dropped'])} unknown leaves: "
                          f"{vreport['dropped'][:8]}", file=sys.stderr)
                if vreport.get("snapped"):
                    print(f"  validate: snapped {vreport['snapped']} castle parts to "
                          f"canonical layout", file=sys.stderr)

                try:
                    tree = CompositionNode.from_dict(tree_dict)
                except Exception as e:
                    return JSONResponse({"error": f"tree parse error: {e}"})

            fidelity = req.fidelity if req.fidelity else args.fidelity
            refine_mode = req.refine_mode if req.refine_mode else args.refine_mode

            pipeline_info = None
            if fidelity == "high":
                try:
                    # Both paths arrive here with an ALREADY-ATOMIC, filled tree:
                    # scene-file loads the grammar scene; the model path ran
                    # generate_tree -> _fill_gaussians which expanded the shallow
                    # skeleton into atomic compounds. So always prebuilt -- never
                    # re-subdivide atomic stones (that inflates ~2.5K parts into
                    # ~160K redundant blobs and blows up the pipeline).
                    tensors, info = apply_high_fidelity(
                        tree, refine_mode=refine_mode,
                        templates_dir=args.templates,
                        prebuilt=True,
                    )
                    pipeline_info = f"sub:{info['n_subdivided']} -> dense:{info['n_densified']} -> refine:{info['n_refined']}"
                except Exception as e:
                    import traceback
                    print("\n=== high-fidelity pipeline error ===", file=sys.stderr)
                    traceback.print_exc()
                    print("=====================================\n", file=sys.stderr)
                    return JSONResponse({"error": f"high-fidelity pipeline error: {type(e).__name__}: {e}"})
            else:
                tensors = tree_to_tensors(tree)

            last_tensors = tensors
            n = tensors["means"].shape[0]

            # Subsample for JSON transfer if too many points
            max_json_splats = 100000
            if n > max_json_splats:
                idx = torch.randperm(n)[:max_json_splats]
                means_out = tensors["means"][idx].tolist()
                colors_out = tensors["colors"][idx].tolist()
                scales_out = tensors["scales_log"][idx].tolist()
                rots_out = tensors["rotations"][idx].tolist()
                opacities_out = torch.sigmoid(tensors["opacities"][idx]).tolist()
                n_out = max_json_splats
            else:
                means_out = tensors["means"].tolist()
                colors_out = tensors["colors"].tolist()
                scales_out = tensors["scales_log"].tolist()
                rots_out = tensors["rotations"].tolist()
                opacities_out = torch.sigmoid(tensors["opacities"]).tolist()
                n_out = n

            response = {
                "tree": tree_dict,
                "splats": {
                    "means": means_out,
                    "scales": scales_out,
                    "rotations": rots_out,
                    "opacities": opacities_out,
                    "colors": colors_out,
                    "n_splats": n_out,
                },
                "n_gaussians": n,
                "depth": tree.depth,
                "n_children": len(tree.children),
            }
            if pipeline_info:
                response["pipeline"] = pipeline_info

            return JSONResponse(response)

        @app.get("/export_ply")
        def export_ply():
            if not last_tensors:
                return JSONResponse({"error": "no scene generated yet"}, status_code=400)
            from src.export.ply import write_ply
            tmp = tempfile.mktemp(suffix=".ply")
            write_ply(last_tensors, tmp)
            with open(tmp, "rb") as f:
                data = f.read()
            Path(tmp).unlink(missing_ok=True)
            return Response(content=data, media_type="application/octet-stream",
                           headers={"Content-Disposition": "attachment; filename=scene.ply"})

        import uvicorn
        print(f"\nServing at http://{args.host}:{args.port}")
        print(f"Fidelity: {args.fidelity} | Refine mode: {args.refine_mode}")
        print(f"Templates: {args.templates}")
        print("Enter a prompt in the UI to decompose + render.")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

    else:
        print("Specify --prompt or --serve")


if __name__ == "__main__":
    main()
