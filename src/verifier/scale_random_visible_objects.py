"""Randomly scale a subset of the visible furniture, uniformly per object.

Reads the "visible_furniture" list from metadata.json, picks a random n% of
those objects, and multiplies each one's scale by a random uniform factor
(same factor on x/y/z, so the object keeps its proportions) in scene.glb.
Position and orientation are unchanged. Writes the result (scene.glb +
camera.json only) to a new output directory.
"""

from __future__ import annotations

import argparse
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


def scale_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float | None = None,
    seed: int | None = None,
    min_factor: float = 0.5,
    max_factor: float = 1.5,
) -> dict[str, float]:
    """Uniformly rescale n% of a scene's visible objects, in place.

    Args:
        scene_dir: Source scene directory (contains scene.glb, camera.json, metadata.json).
        output_dir: Destination directory for the modified scene. Overwritten if it exists.
        percent: Percentage (0-100) of visible objects to scale. If omitted, sampled
            randomly from {0, 10, ..., 100} and recorded in percent.json.
        seed: Optional RNG seed for reproducible selection and factors.
        min_factor: Minimum scale factor.
        max_factor: Maximum scale factor.

    Returns:
        scaling: mapping of scaled furniture name -> applied scale factor.
    """
    metadata = load_metadata(scene_dir)
    furniture = furniture_by_name(metadata)

    rng = random.Random(seed)
    percent = resolve_percent(percent, rng)
    selected = select_random_visible_furniture(metadata, percent, rng)
    scaling = {name: rng.uniform(min_factor, max_factor) for name in selected}

    scene_glb_path = prepare_permuted_scene_dir(scene_dir, output_dir)

    if scaling:
        new_transforms = {}
        for name, factor in scaling.items():
            item = furniture[name]
            new_scale = [s * factor for s in item["scale"]]
            new_transforms[name] = make_transform(item["pos"], item["rot"], new_scale)
        update_node_transforms(scene_glb_path, new_transforms)

    write_json(output_dir / "scaling.json", scaling)
    write_json(output_dir / "percent.json", {"percent": percent})

    return scaling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument(
        "percent",
        type=float,
        nargs="?",
        default=None,
        help="Percentage (0-100) of visible objects to scale. If omitted, sampled randomly from 0,10,...,100",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    parser.add_argument("--min-factor", type=float, default=0.5, help="Minimum scale factor")
    parser.add_argument("--max-factor", type=float, default=1.5, help="Maximum scale factor")
    args = parser.parse_args()

    scaling = scale_random_visible_objects(
        args.scene_dir,
        args.output_dir,
        args.percent,
        seed=args.seed,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )

    print(f"Scaled {len(scaling)} object(s) in {args.output_dir}:")
    for name, factor in scaling.items():
        print(f"  - {name}: {factor:.2f}x")


if __name__ == "__main__":
    main()
