"""Randomly translate a subset of the visible furniture in the horizontal plane.

Reads the "visible_furniture" list from metadata.json, picks a random n% of
those objects, and nudges each one by a random distance in a random direction
within the horizontal (x/z, since the GLB is Y-up) plane in scene.glb. The
vertical (y) position, orientation, and scale are unchanged, so objects stay
grounded. Writes the result (scene.glb + camera.json only) to a new output
directory.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from src.util import (
    furniture_by_name,
    load_metadata,
    make_transform,
    prepare_permuted_scene_dir,
    resolve_percent,
    select_random_visible_furniture,
    update_node_transforms,
    write_json,
)


def translate_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float | None = None,
    seed: int | None = None,
    min_distance: float = 0.3,
    max_distance: float = 1.0,
) -> dict[str, list[float]]:
    """Randomly translate n% of a scene's visible objects within the horizontal plane.

    Each selected object is offset by a random distance, in a random direction,
    within the horizontal (x/z) plane. Vertical position, orientation, and
    scale are left unchanged.

    Args:
        scene_dir: Source scene directory (contains scene.glb, camera.json, metadata.json).
        output_dir: Destination directory for the modified scene. Overwritten if it exists.
        percent: Percentage (0-100) of visible objects to translate. If omitted, sampled
            randomly from {0, 10, ..., 100} and recorded in percent.json.
        seed: Optional RNG seed for reproducible selection, directions, and distances.
        min_distance: Minimum translation distance, in scene units (meters).
        max_distance: Maximum translation distance, in scene units (meters).

    Returns:
        translations: mapping of translated furniture name -> applied [dx, dy, dz] offset.
    """
    metadata = load_metadata(scene_dir)
    furniture = furniture_by_name(metadata)

    rng = random.Random(seed)
    percent = resolve_percent(percent, rng)
    selected = select_random_visible_furniture(metadata, percent, rng)

    translations = {}
    for name in selected:
        distance = rng.uniform(min_distance, max_distance)
        angle = rng.uniform(0, 2 * math.pi)
        translations[name] = [distance * math.cos(angle), 0.0, distance * math.sin(angle)]

    scene_glb_path = prepare_permuted_scene_dir(scene_dir, output_dir)

    if translations:
        new_transforms = {}
        for name, offset in translations.items():
            item = furniture[name]
            new_pos = [item["pos"][i] + offset[i] for i in range(3)]
            new_transforms[name] = make_transform(new_pos, item["rot"], item["scale"])
        update_node_transforms(scene_glb_path, new_transforms)

    write_json(output_dir / "translations.json", translations)
    write_json(output_dir / "percent.json", {"percent": percent})

    return translations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument(
        "percent",
        type=float,
        nargs="?",
        default=None,
        help="Percentage (0-100) of visible objects to translate. If omitted, sampled randomly from 0,10,...,100",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    parser.add_argument("--min-distance", type=float, default=0.3, help="Minimum translation distance")
    parser.add_argument("--max-distance", type=float, default=1.0, help="Maximum translation distance")
    args = parser.parse_args()

    translations = translate_random_visible_objects(
        args.scene_dir,
        args.output_dir,
        args.percent,
        seed=args.seed,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )

    print(f"Translated {len(translations)} object(s) in {args.output_dir}:")
    for name, offset in translations.items():
        print(f"  - {name}: [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}]")


if __name__ == "__main__":
    main()
