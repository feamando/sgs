"""
Generate diverse composition trees for decomposer training using Claude Haiku.

Produces 1000+ scene decomposition trees as JSON, covering diverse prompts
(castles, ships, villages, forests, mountains, cities, etc.) with proper
semantic node names, positions, and scales.

Cost: ~$0.55 for 1000 trees via Haiku.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_training_trees_llm.py --n-trees 1000 --output data/training_trees
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SCENE_PROMPTS = [
    # Architecture
    "a castle on a hill",
    "a medieval fortress with towers",
    "a gothic cathedral",
    "a stone bridge over a river",
    "a lighthouse on rocky cliffs",
    "a windmill in a field",
    "a ruined abbey",
    "a watchtower on a mountain pass",
    "an ancient temple with columns",
    "a wooden cabin in the forest",
    "a stone cottage with a garden",
    "a church with a bell tower",
    "a city gate with walls",
    "a palace with courtyards",
    "a viking longhouse",
    "a japanese pagoda",
    "a mosque with minarets",
    "a roman colosseum",
    "an egyptian pyramid",
    "a medieval market square",
    # Nature
    "a mountain lake with pine trees",
    "a waterfall in a jungle",
    "a desert oasis with palm trees",
    "a volcanic island",
    "a frozen lake with ice formations",
    "a canyon with layered rock",
    "a forest clearing with wildflowers",
    "a coastal cliff with waves",
    "a river delta with sandbars",
    "a snowy mountain peak",
    "rolling hills with sheep",
    "a bamboo forest",
    "a coral reef above water",
    "a swamp with dead trees",
    "a rocky beach with tide pools",
    # Vehicles and ships
    "a pirate ship on the ocean",
    "a viking longship",
    "a steamship in a harbor",
    "a fishing boat at a dock",
    "a hot air balloon over fields",
    "a train crossing a bridge",
    "a carriage on a road",
    "a space station orbiting earth",
    # Fantasy and fiction
    "a dragon perched on a tower",
    "a wizard's tower with a garden",
    "a floating island with waterfalls",
    "a dwarven mine entrance",
    "an elven tree city",
    "a haunted mansion on a cliff",
    "a crystal cave",
    "a giant's castle in the clouds",
    # Urban and modern
    "a town square with a fountain",
    "a harbor with boats and a lighthouse",
    "a farm with a barn and silo",
    "a village with a church and houses",
    "a campsite by a lake",
    "an amphitheater on a hillside",
    "a graveyard with a chapel",
    "a market with stalls and canopies",
    # Combinations
    "a castle overlooking a village",
    "a bridge connecting two cliffs",
    "a shipwreck on a beach",
    "a treehouse in an ancient oak",
    "a cave entrance in a mountain",
    "a garden with a stone fountain",
    "ruins overgrown with vines",
    "a dock with fishing boats",
    "a well in a village square",
    "a stone archway in a wall",
]

SYSTEM_PROMPT = """You are a 3D scene decomposition engine. Given a text prompt, output a JSON composition tree that breaks the scene into semantic sub-objects with spatial positions.

Rules:
- Output ONLY valid JSON (no markdown, no explanation)
- Root node is always {"name": "scene", "position": [0, 0, 0], "children": [...]}
- Each child has: name (descriptive string), position ([x, y, z] relative to parent), scale (float, default 1.0)
- Leaf nodes have no children (they are the terminal objects)
- Use descriptive names: "tower", "wall", "gate", "hill", "tree", "rock", not "object1" or "ball"
- Position Y is up. Ground is Y=0. Spread objects in X and Z.
- Scale represents relative size (1.0 = normal, 0.5 = half, 2.0 = double)
- Aim for 5-15 nodes total (not too simple, not too complex)
- Vary positions realistically (towers are above walls, rivers are below bridges)

Example for "a castle on a hill":
{"name":"scene","position":[0,0,0],"children":[{"name":"hill","position":[0,-0.3,0],"scale":2.5,"children":[{"name":"slope_front","position":[0,0,1],"scale":1.5},{"name":"slope_back","position":[0,0,-1],"scale":1.2}]},{"name":"castle","position":[0,1.2,0],"scale":1.5,"children":[{"name":"keep","position":[0,0.5,0],"scale":1.0},{"name":"tower_nw","position":[-0.8,0.3,0.8],"scale":0.6},{"name":"tower_ne","position":[0.8,0.3,0.8],"scale":0.6},{"name":"tower_sw","position":[-0.8,0.3,-0.8],"scale":0.6},{"name":"tower_se","position":[0.8,0.3,-0.8],"scale":0.6},{"name":"wall_north","position":[0,0,1],"scale":0.8},{"name":"wall_south","position":[0,0,-1],"scale":0.8},{"name":"gate","position":[0,-0.1,1.2],"scale":0.5}]}]}"""


def generate_tree_anthropic(prompt: str, client) -> dict | None:
    """Generate one composition tree via Anthropic direct API."""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Decompose: {prompt}"}],
        )
        text = response.content[0].text.strip()
        return _parse_tree_text(text)
    except Exception as e:
        print(f"  API error: {e}")
        return None


def generate_tree_bedrock(prompt: str, client, model_id: str) -> dict | None:
    """Generate one composition tree via AWS Bedrock."""
    import json as json_mod
    try:
        body = json_mod.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Decompose: {prompt}"}],
        })
        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json_mod.loads(response["body"].read())
        text = result["content"][0]["text"].strip()
        return _parse_tree_text(text)
    except Exception as e:
        print(f"  Bedrock error: {e}")
        return None


def _parse_tree_text(text: str) -> dict | None:
    """Parse tree JSON from model output (handles markdown fencing)."""
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        tree = json.loads(text)
        return tree
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                pass
        return None


def count_nodes(tree: dict) -> int:
    """Count total nodes in a tree."""
    count = 1
    for child in tree.get("children", []):
        count += count_nodes(child)
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate training trees via Claude Haiku")
    parser.add_argument("--n-trees", type=int, default=10000)
    parser.add_argument("--output", default="data/training_trees",
                        help="Output directory for tree JSONs")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Save progress every N trees")
    parser.add_argument("--backend", choices=["bedrock", "anthropic"], default="bedrock",
                        help="API backend (default: bedrock)")
    parser.add_argument("--region", default="us-east-1",
                        help="AWS region for Bedrock (default: us-east-1)")
    parser.add_argument("--model-id", default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        help="Bedrock model ID")
    args = parser.parse_args()

    if args.backend == "bedrock":
        try:
            import boto3
        except ImportError:
            print("pip install boto3")
            sys.exit(1)
        session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "bedrock"))
        client = session.client("bedrock-runtime", region_name=args.region)
        generate_fn = lambda prompt: generate_tree_bedrock(prompt, client, args.model_id)
        print(f"Backend: AWS Bedrock ({args.region})")
        print(f"Model: {args.model_id}")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        generate_fn = lambda prompt: generate_tree_anthropic(prompt, client)
        print(f"Backend: Anthropic direct API")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate prompts: cycle through base prompts with variations
    all_prompts = []
    variations = [
        "", " at dawn", " at night", " in winter", " in autumn",
        " from the east", " with a moat", " surrounded by trees",
        " on a cliff", " by the sea", " in the desert", " in the mountains",
        " in the rain", " covered in snow", " at sunset", " with fog",
        " from above", " in spring", " abandoned", " newly built",
        " small", " massive", " ancient", " futuristic",
    ]

    scale_mods = [
        "", " (3 objects)", " (5-7 objects)", " (10+ objects, complex)",
        " (simple, 3 parts)", " (detailed, many sub-parts)",
    ]

    for prompt in SCENE_PROMPTS:
        for var in variations:
            for scale in scale_mods:
                all_prompts.append(prompt + var + scale)
                if len(all_prompts) >= args.n_trees:
                    break
            if len(all_prompts) >= args.n_trees:
                break
        if len(all_prompts) >= args.n_trees:
            break

    # If still need more, repeat with shuffled combinations
    import random
    random.seed(42)
    while len(all_prompts) < args.n_trees:
        base = random.choice(SCENE_PROMPTS)
        var = random.choice(variations)
        scale = random.choice(scale_mods)
        all_prompts.append(base + var + scale)
    all_prompts = all_prompts[:args.n_trees]

    print(f"Generating {args.n_trees} trees via Claude Haiku...")
    print(f"Output: {output_dir}")
    print(f"Unique prompts: {len(set(all_prompts))}")

    generated = 0
    failed = 0
    total_nodes = 0

    for i, prompt in enumerate(all_prompts):
        tree = generate_fn(prompt)

        if tree and "name" in tree:
            n_nodes = count_nodes(tree)
            if n_nodes >= 3:  # skip trivially simple trees
                out_path = output_dir / f"tree_{generated:04d}.json"
                with open(out_path, "w") as f:
                    json.dump(tree, f)
                generated += 1
                total_nodes += n_nodes
            else:
                failed += 1
        else:
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_trees}] generated={generated} failed={failed} avg_nodes={total_nodes/max(generated,1):.1f}")

        if (i + 1) % args.batch_size == 0:
            # Save manifest
            manifest = {
                "generated": generated,
                "failed": failed,
                "avg_nodes": total_nodes / max(generated, 1),
                "prompts_processed": i + 1,
            }
            with open(output_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

        # Light rate limiting (Haiku is fast, but be polite)
        time.sleep(0.1)

    # Final manifest
    manifest = {
        "generated": generated,
        "failed": failed,
        "total_nodes": total_nodes,
        "avg_nodes": total_nodes / max(generated, 1),
        "prompts_processed": len(all_prompts),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE. Generated {generated} trees ({failed} failed)")
    print(f"Avg nodes/tree: {total_nodes/max(generated,1):.1f}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    print(f"\nNext: retrain the decomposer:")
    print(f"  python scripts\\train_decomposer.py --data {output_dir} --output checkpoints\\decomposer\\best.pt")


if __name__ == "__main__":
    main()
