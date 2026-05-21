"""
Generate decomposition tree training data via Claude API.

Calls Claude to decompose scene prompts into composition trees in the
format expected by src/raum/decomposition.py. Outputs JSON training
pairs: [{prompt, tree}, ...].

Usage:
    python scripts/generate_decomposition_trees.py `
      --output data/decomposition_trees/train.json `
      --n-scenes 100

Requires ANTHROPIC_API_KEY environment variable.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SCENE_PROMPTS = [
    "a castle on a hill",
    "a red barn in a green field",
    "a lighthouse on rocky cliffs",
    "a medieval village with a church",
    "a pirate ship on the ocean",
    "a treehouse in a tall oak",
    "a space station orbiting earth",
    "a windmill in a tulip field",
    "a bridge over a river",
    "a campfire in a forest clearing",
    "a snowman in a winter garden",
    "a robot in a factory",
    "a dragon on a mountain peak",
    "a submarine under the sea",
    "a hot air balloon over a valley",
    "a train crossing a bridge",
    "a cottage with a chimney",
    "a fountain in a courtyard",
    "a rocket on a launch pad",
    "a sailboat near an island",
    "a temple in the jungle",
    "a ferris wheel at a fair",
    "a helicopter on a rooftop",
    "a greenhouse full of plants",
    "a tower with a clock",
    "a log cabin by a lake",
    "a cathedral with stained glass",
    "a crane at a construction site",
    "an igloo in the arctic",
    "a pagoda in a garden",
    "a waterfall into a pool",
    "a volcano erupting",
    "a city skyline at sunset",
    "a farm with animals",
    "a playground with swings",
    "a library with bookshelves",
    "a kitchen with a stove",
    "a car on a road",
    "a bicycle leaning on a wall",
    "a piano in a concert hall",
    "a telescope on a hilltop",
    "an aquarium with fish",
    "a throne in a great hall",
    "a well in a village square",
    "a tent in a desert",
    "a birdhouse on a pole",
    "a mailbox at a house",
    "a lantern on a post",
    "a bookshelf against a wall",
    "a swing hanging from a tree",
    "a mushroom in the forest",
    "a cactus in the desert",
    "a snowflake falling",
    "a crown on a pillow",
    "a sword in a stone",
    "a key in a lock",
    "a candle on a table",
    "a globe on a desk",
    "a compass pointing north",
    "a hourglass on a shelf",
    "a chess board with pieces",
    "a guitar leaning on an amp",
    "a microscope on a bench",
    "a trophy on a mantle",
    "a ship in a bottle",
    "a bird in a cage",
    "a fish in a bowl",
    "a flower in a vase",
    "a flag on a pole",
    "a bell in a tower",
    "a wheel on an axle",
    "a lamp on a nightstand",
    "a mirror on a wall",
    "a painting in a frame",
    "a window with curtains",
    "a door with a knocker",
    "a gate with iron bars",
    "a fence around a garden",
    "a path through the woods",
    "a bridge over a stream",
    "a dock on a lake",
    "a bench in a park",
    "a statue in a plaza",
    "a monument on a hill",
    "a pyramid in the sand",
    "a dome on a building",
    "an arch over a walkway",
    "a column supporting a roof",
    "a staircase spiraling up",
    "a balcony overlooking a garden",
    "a chimney with smoke",
    "a weathervane on a roof",
    "a sundial in a garden",
    "an anchor on a chain",
    "a lifeboat on a ship",
    "a periscope above water",
    "a drawbridge over a moat",
    "a turret on a castle wall",
    "a portcullis in an archway",
    "a catapult on a battlefield",
]

DECOMPOSITION_PROMPT = """You are a 3D scene decomposition engine. Given a text prompt describing a scene, produce a JSON composition tree that recursively breaks the scene into sub-parts down to geometric primitives.

Rules:
1. The root node is always "scene" with the full prompt
2. Each node has: name, position [x,y,z] (relative to parent), scale (float, multiplier)
3. Internal nodes have "children" (list of sub-nodes)
4. Leaf nodes have "gaussians" (list of terminal primitives)
5. Each gaussian has: position [x,y,z], scale [sx,sy,sz] (log-scale), opacity (float, logit), color [r,g,b] (0-1)
6. Use these primitive shapes for leaves:
   - Cylinder: points along a vertical axis
   - Cone: points in a cone shape
   - Box: points on 6 faces
   - Sphere/dome: points on a spherical surface
   - Flat plane: points on a horizontal surface
7. Typical leaf has 30-80 gaussians
8. Tree depth should be 2-4 levels
9. Keep positions reasonable: scene fits in a [-3, 3] cube
10. Color should be realistic for the object

Respond with ONLY the JSON tree. No explanation, no markdown fences.

Prompt: "{prompt}"
"""


def generate_tree(prompt: str, client) -> dict | None:
    """Call Claude to generate a decomposition tree for a prompt."""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": DECOMPOSITION_PROMPT.format(prompt=prompt),
            }],
        )
        text = message.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
        tree = json.loads(text)
        return tree
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ERROR: {e}")
        return None


def validate_tree(tree: dict) -> bool:
    """Basic validation that the tree has the expected structure."""
    if not isinstance(tree, dict):
        return False
    if "name" not in tree:
        return False
    has_children = "children" in tree and len(tree["children"]) > 0
    has_gaussians = "gaussians" in tree and len(tree["gaussians"]) > 0
    if not has_children and not has_gaussians:
        # Internal node must have children or be a leaf with gaussians
        # Root with children is fine
        return "children" in tree or "gaussians" in tree
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Generate decomposition trees via Claude API")
    p.add_argument("--output", default="data/decomposition_trees/train.json")
    p.add_argument("--n-scenes", type=int, default=100,
                   help="Number of scenes to generate")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Start from this prompt index (for resuming)")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between API calls (rate limiting)")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if resuming
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing trees from {output_path}")

    prompts = SCENE_PROMPTS[:args.n_scenes]
    if args.start_idx > 0:
        prompts = prompts[args.start_idx:]

    print(f"Generating {len(prompts)} decomposition trees...")
    print(f"Output: {output_path}")
    print()

    results = list(existing)
    n_success = 0
    n_fail = 0

    for i, prompt in enumerate(prompts):
        # Skip if already generated
        if any(r["prompt"] == prompt for r in results):
            print(f"  [{i+1}/{len(prompts)}] SKIP (already exists): {prompt}")
            continue

        print(f"  [{i+1}/{len(prompts)}] Generating: {prompt}...", end=" ", flush=True)
        tree = generate_tree(prompt, client)

        if tree and validate_tree(tree):
            results.append({"prompt": prompt, "tree": tree})
            n_success += 1
            print("OK")
        else:
            n_fail += 1
            print("FAIL")

        # Save after each successful generation (crash-safe)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(args.delay)

    print(f"\nDone. {n_success} generated, {n_fail} failed.")
    print(f"Total: {len(results)} trees at {output_path}")


if __name__ == "__main__":
    main()
