"""
Download architectural 3D assets for SGS template training.

Sources:
1. Sketchfab API: CC-licensed photogrammetry scans of architectural elements
2. OpenGameArt: CC0/CC-BY modular castle kits (direct download)
3. Kenney.nl: CC0 castle kit (direct download)

Usage:
    # Sketchfab (requires API token)
    python scripts/download_architecture_assets.py --source sketchfab --token YOUR_TOKEN --output data/architecture

    # OpenGameArt (no auth needed)
    python scripts/download_architecture_assets.py --source opengameart --output data/architecture

    # All sources
    python scripts/download_architecture_assets.py --source all --token YOUR_TOKEN --output data/architecture

Environment variable alternative:
    set SKETCHFAB_API_TOKEN=YOUR_TOKEN
    python scripts/download_architecture_assets.py --source all --output data/architecture
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Sketchfab search queries: architecture + nature/landscape
SKETCHFAB_QUERIES = {
    # Architecture
    "tower": "castle tower medieval photogrammetry",
    "wall": "stone wall medieval photogrammetry",
    "arch": "stone arch gothic photogrammetry",
    "column": "column capital classical photogrammetry",
    "gate": "medieval gate door photogrammetry",
    "roof": "roof tiles medieval scan",
    "battlement": "battlement merlon castle scan",
    "stairs": "stone stairs medieval scan",
    "brick": "brick wall texture scan photogrammetry",
    "window": "gothic window arch stone scan",
    "floor": "stone floor cobblestone scan",
    # Nature and landscape (for "castle on a hill" surroundings)
    "rock": "rock boulder scan photogrammetry",
    "cliff": "cliff rock face terrain photogrammetry",
    "hill": "hillside terrain grass photogrammetry",
    "tree": "tree trunk bark photogrammetry scan",
    "bush": "bush shrub vegetation photogrammetry",
    "grass": "grass ground terrain photogrammetry",
    "forest": "forest floor moss roots photogrammetry",
    "path": "dirt path trail ground photogrammetry",
    "moss": "moss covered stone rock photogrammetry",
    "soil": "soil earth ground terrain scan",
}

# OpenGameArt direct download URLs (modular castle/medieval kits)
OPENGAMEART_URLS = [
    # These are example patterns; actual URLs need verification
    ("castle_kit", "https://opengameart.org/sites/default/files/Medieval_Building_Kit.zip"),
    ("dungeon_kit", "https://opengameart.org/sites/default/files/Dungeon_Kit.zip"),
]

# Kenney.nl castle kit (CC0)
KENNEY_URLS = [
    ("castle_kit_kenney", "https://kenney.nl/media/pages/assets/castle-kit/1/kenney_castle-kit.zip"),
]


def download_file(url: str, dest: Path, headers: dict = None) -> bool:
    """Download a file from URL to destination path."""
    try:
        req = Request(url)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        with urlopen(req, timeout=60) as response:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except (HTTPError, Exception) as e:
        print(f"  Failed: {e}")
        return False


def sketchfab_search(query: str, token: str, max_results: int = 10) -> list[dict]:
    """Search Sketchfab for downloadable models."""
    from urllib.parse import urlencode

    params = urlencode({
        "type": "models",
        "q": query,
        "downloadable": "true",
        "sort_by": "-likeCount",
        "count": str(max_results),
    })
    url = f"https://api.sketchfab.com/v3/search?{params}"
    headers = {"Authorization": f"Token {token}"}

    try:
        req = Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            # Filter for CC licenses client-side
            results = []
            for r in data.get("results", []):
                lic = r.get("license", {}).get("label", "")
                if "CC" in lic or "Creative Commons" in lic:
                    results.append(r)
            return results if results else data.get("results", [])[:max_results]
    except Exception as e:
        print(f"  Search failed for '{query}': {e}")
        return []


def sketchfab_download(uid: str, token: str, dest_dir: Path) -> bool:
    """Download a model from Sketchfab by UID."""
    # Step 1: Request download URL
    url = f"https://api.sketchfab.com/v3/models/{uid}/download"
    headers = {"Authorization": f"Token {token}"}

    try:
        req = Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        # Get the glTF or OBJ download link
        download_url = None
        for fmt in ["gltf", "obj", "source"]:
            if fmt in data:
                download_url = data[fmt].get("url")
                if download_url:
                    break

        if not download_url:
            print(f"  No downloadable format for {uid}")
            return False

        # Step 2: Download the zip
        zip_path = dest_dir / f"{uid}.zip"
        if download_file(download_url, zip_path):
            # Step 3: Extract
            extract_dir = dest_dir / uid
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                zip_path.unlink()
                return True
            except zipfile.BadZipFile:
                print(f"  Bad zip for {uid}")
                zip_path.unlink(missing_ok=True)
                return False
        return False

    except HTTPError as e:
        if e.code == 429:
            print("  Rate limited. Waiting 60s...")
            time.sleep(60)
            return sketchfab_download(uid, token, dest_dir)
        print(f"  Download failed for {uid}: {e}")
        return False


def download_sketchfab(token: str, output_dir: Path, max_per_category: int = 10):
    """Download architectural models from Sketchfab."""
    print("=== Sketchfab Downloads ===")

    total_downloaded = 0
    for category, query in SKETCHFAB_QUERIES.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{category}] Searching: '{query}'")

        results = sketchfab_search(query, token, max_results=max_per_category)
        print(f"  Found {len(results)} results")

        for i, model in enumerate(results):
            uid = model["uid"]
            name = model.get("name", uid)[:50]

            # Skip if already downloaded
            if (cat_dir / uid).exists():
                print(f"  ({i+1}/{len(results)}) {name}... EXISTS (skip)")
                continue

            print(f"  ({i+1}/{len(results)}) {name}...", end=" ")

            if sketchfab_download(uid, token, cat_dir):
                print("OK")
                total_downloaded += 1
            else:
                print("SKIP")

            # Rate limit: 1 request per second
            time.sleep(1.5)

    print(f"\nSketchfab: downloaded {total_downloaded} models")
    return total_downloaded


def download_opengameart(output_dir: Path):
    """Download modular castle kits from OpenGameArt."""
    print("\n=== OpenGameArt Downloads ===")

    downloaded = 0
    for name, url in OPENGAMEART_URLS:
        dest = output_dir / "game_kits" / f"{name}.zip"
        print(f"  Downloading {name}...", end=" ")
        if download_file(url, dest):
            # Extract
            extract_dir = output_dir / "game_kits" / name
            try:
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(extract_dir)
                dest.unlink()
                print("OK")
                downloaded += 1
            except (zipfile.BadZipFile, Exception) as e:
                print(f"Bad zip: {e}")
        else:
            print("FAILED")

    print(f"OpenGameArt: downloaded {downloaded} kits")
    return downloaded


def download_kenney(output_dir: Path):
    """Download CC0 castle kit from Kenney.nl."""
    print("\n=== Kenney.nl Downloads ===")

    downloaded = 0
    for name, url in KENNEY_URLS:
        dest = output_dir / "kenney" / f"{name}.zip"
        print(f"  Downloading {name}...", end=" ")
        if download_file(url, dest):
            extract_dir = output_dir / "kenney" / name
            try:
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(extract_dir)
                dest.unlink()
                print("OK")
                downloaded += 1
            except (zipfile.BadZipFile, Exception) as e:
                print(f"Bad zip: {e}")
        else:
            print("FAILED")

    print(f"Kenney: downloaded {downloaded} kits")
    return downloaded


def post_process(output_dir: Path):
    """
    After download: find all .obj/.stl/.glb files and organize by category.

    Creates a summary of what was downloaded and where.
    """
    print("\n=== Post-processing ===")

    mesh_files = []
    for ext in ["*.obj", "*.stl", "*.gltf", "*.glb", "*.fbx"]:
        mesh_files.extend(output_dir.rglob(ext))

    print(f"Found {len(mesh_files)} mesh files total")

    # Write manifest
    manifest = {
        "total_meshes": len(mesh_files),
        "by_extension": {},
        "files": [],
    }
    for f in mesh_files:
        ext = f.suffix.lower()
        manifest["by_extension"][ext] = manifest["by_extension"].get(ext, 0) + 1
        manifest["files"].append(str(f.relative_to(output_dir)))

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")
    print(f"Extensions: {manifest['by_extension']}")

    return len(mesh_files)


def main():
    parser = argparse.ArgumentParser(description="Download architectural 3D assets")
    parser.add_argument("--source", choices=["sketchfab", "opengameart", "kenney", "all"],
                        default="all", help="Download source (default: all)")
    parser.add_argument("--token", default=None,
                        help="Sketchfab API token (or set SKETCHFAB_API_TOKEN env var)")
    parser.add_argument("--output", default="data/architecture",
                        help="Output directory (default: data/architecture)")
    parser.add_argument("--max-per-category", type=int, default=10,
                        help="Max models per category from Sketchfab (default: 10)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("SKETCHFAB_API_TOKEN")

    if args.source in ("sketchfab", "all"):
        if not token:
            print("Sketchfab requires an API token.")
            print("Get one at: https://sketchfab.com/settings/password")
            print("Then: --token YOUR_TOKEN or set SKETCHFAB_API_TOKEN env var")
            if args.source == "sketchfab":
                sys.exit(1)
            print("Skipping Sketchfab, continuing with other sources...\n")
        else:
            download_sketchfab(token, output_dir, args.max_per_category)

    if args.source in ("opengameart", "all"):
        download_opengameart(output_dir)

    if args.source in ("kenney", "all"):
        download_kenney(output_dir)

    n_meshes = post_process(output_dir)

    print(f"\n{'='*60}")
    print(f"DONE. {n_meshes} mesh files in {output_dir}")
    print(f"{'='*60}")
    print(f"\nNext: convert to GS training data:")
    print(f"  python scripts\\build_objaverse_gs.py --objaverse-dir {output_dir} --output data\\architecture_gs")


if __name__ == "__main__":
    main()
