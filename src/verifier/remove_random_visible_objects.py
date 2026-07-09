"""Remove a random subset of the visible furniture from a single scene.

Reads the "visible_furniture" list from metadata.json, drops a random n% of
those objects from scene.glb, and writes the result (scene.glb + camera.json
only) to a new output directory.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.util import (
    load_metadata,
    prepare_permuted_scene_dir,
    remove_nodes,
    select_random_visible_furniture,
    write_json,
)


def remove_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float,
    seed: int | None = None,
) -> dict[str, bool]:
    """Remove n% of a scene's visible objects, writing scene.glb + camera.json to output_dir.

    Args:
        scene_dir: Source scene directory (contains scene.glb, camera.json, metadata.json).
        output_dir: Destination directory for the modified scene. Overwritten if it exists.
        percent: Percentage (0-100) of visible objects to remove.
        seed: Optional RNG seed for reproducible selection.

    Returns:
        removed_objects: mapping of removed furniture name -> True.
    """
    metadata = load_metadata(scene_dir)
    rng = random.Random(seed)
    selected = select_random_visible_furniture(metadata, percent, rng)

    scene_glb_path = prepare_permuted_scene_dir(scene_dir, output_dir)
    if selected:
        remove_nodes(scene_glb_path, selected)

    removed_objects = {name: True for name in selected}
    write_json(output_dir / "removed_objects.json", removed_objects)

    return removed_objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument("percent", type=float, help="Percentage (0-100) of visible objects to remove")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    args = parser.parse_args()

    removed_objects = remove_random_visible_objects(args.scene_dir, args.output_dir, args.percent, seed=args.seed)

    print(f"Removed {len(removed_objects)} object(s) in {args.output_dir}:")
    for name in removed_objects:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
