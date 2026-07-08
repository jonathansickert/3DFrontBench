"""Remove a random subset of the visible furniture from a single scene.

Reads the "visible_furniture" list from metadata.json (also mirrored by the
per-object GLBs under the scene's visible_furniture/ folder), drops a random
n% of those objects from scene.glb / scene_bbox.glb, deletes their GLBs, and
writes the result to a new scene directory. The original pos/rot/scale of
removed objects is preserved under metadata["removed_furniture"] so they can
still be used as ground truth (e.g. for a "put the object back" benchmark).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import trimesh


def _geom_names_for_nodes(scene: trimesh.Scene, node_names: list[str]) -> set[str]:
    geom_names = set()
    for node_name in node_names:
        _, geom_name = scene.graph[node_name]
        geom_names.add(geom_name)
    return geom_names


def _strip_objects_from_glb(glb_path: Path, node_names: list[str]) -> None:
    if not glb_path.exists():
        return

    scene = trimesh.load(glb_path, process=False)
    scene.delete_geometry(_geom_names_for_nodes(scene, node_names))
    scene.export(glb_path)


def remove_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float,
    seed: int | None = None,
) -> list[str]:
    """Copy a scene to output_dir with n% of its visible objects removed.

    Args:
        scene_dir: Source scene directory (contains scene.glb, metadata.json, visible_furniture/).
        output_dir: Destination directory to write the modified scene to. Must not already exist.
        percent: Percentage (0-100) of visible objects to remove.
        seed: Optional RNG seed for reproducible selection.

    Returns:
        Names of the furniture objects that were removed.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")

    shutil.copytree(scene_dir, output_dir)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    visible_names = list(metadata["visible_furniture"])
    n_remove = round(len(visible_names) * percent / 100)

    rng = random.Random(seed)
    removed_names = set(rng.sample(visible_names, n_remove))

    if not removed_names:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return []

    _strip_objects_from_glb(output_dir / "scene.glb", list(removed_names))

    removed_furniture = [item for item in metadata["furniture"] if item["name"] in removed_names]
    metadata["furniture"] = [item for item in metadata["furniture"] if item["name"] not in removed_names]
    metadata["visible_furniture"] = [name for name in visible_names if name not in removed_names]
    metadata["removed_furniture"] = removed_furniture

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    visible_furniture_dir = output_dir / "visible_furniture"
    for name in removed_names:
        glb_path = visible_furniture_dir / f"{name}.glb"
        glb_path.unlink(missing_ok=True)

    return sorted(removed_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument("percent", type=float, help="Percentage (0-100) of visible objects to remove")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    args = parser.parse_args()

    removed = remove_random_visible_objects(args.scene_dir, args.output_dir, args.percent, seed=args.seed)

    print(f"Removed {len(removed)} object(s) from {args.output_dir}:")
    for name in removed:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
