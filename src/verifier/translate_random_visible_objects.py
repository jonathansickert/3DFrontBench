"""Randomly translate a subset of the visible furniture in the horizontal plane.

Reads the "visible_furniture" list from metadata.json, picks a random n% of
those objects, and nudges each one by a random distance in a random direction
within the horizontal (x/z, since the GLB is Y-up) plane in scene.glb /
scene_bbox.glb. The vertical (y) position, orientation, and scale are
unchanged, so objects stay grounded. The furniture's "pos" field in
metadata.json is updated to match, and the original position plus the
applied offset is recorded under metadata["translated_furniture"] for
ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import trimesh

from src.util import make_transform


def _translate_objects_in_glb(glb_path: Path, transforms: dict[str, list]) -> None:
    if not glb_path.exists():
        return

    scene = trimesh.load(glb_path, process=False)
    for node_name, new_transform in transforms.items():
        scene.graph.update(frame_to=node_name, matrix=new_transform)
    scene.export(glb_path)


def translate_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float,
    seed: int | None = None,
    min_distance: float = 0.3,
    max_distance: float = 1.0,
) -> dict[str, float]:
    """Copy a scene to output_dir with n% of its visible objects randomly translated.

    Each selected object is offset by a random distance, in a random direction,
    within the horizontal (x/z) plane. Vertical position, orientation, and
    scale are left unchanged.

    Args:
        scene_dir: Source scene directory (contains scene.glb, metadata.json, visible_furniture/).
        output_dir: Destination directory to write the modified scene to. Must not already exist.
        percent: Percentage (0-100) of visible objects to translate.
        seed: Optional RNG seed for reproducible selection, directions, and distances.
        min_distance: Minimum translation distance, in scene units (meters).
        max_distance: Maximum translation distance, in scene units (meters).

    Returns:
        Mapping of translated object name to the applied distance.
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
    n_translate = round(len(visible_names) * percent / 100)

    rng = random.Random(seed)
    translated_names = rng.sample(visible_names, n_translate)

    offsets = {}
    for name in translated_names:
        distance = rng.uniform(min_distance, max_distance)
        angle = rng.uniform(0, 2 * math.pi)
        offsets[name] = (distance * math.cos(angle), distance * math.sin(angle), distance)

    if not offsets:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return {}

    new_transforms = {}
    translated_furniture = []
    for item in metadata["furniture"]:
        name = item["name"]
        if name not in offsets:
            continue
        dx, dz, distance = offsets[name]
        original_pos = item["pos"]
        item["pos"] = [original_pos[0] + dx, original_pos[1], original_pos[2] + dz]
        new_transforms[name] = make_transform(item["pos"], item["rot"], item["scale"]).tolist()
        translated_furniture.append(
            {
                "name": name,
                "distance": distance,
                "offset": [dx, 0.0, dz],
                "original_pos": original_pos,
                "new_pos": item["pos"],
            }
        )
    metadata["translated_furniture"] = translated_furniture

    _translate_objects_in_glb(output_dir / "scene.glb", new_transforms)
    _translate_objects_in_glb(output_dir / "scene_bbox.glb", new_transforms)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {name: distance for name, (_, _, distance) in offsets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument("percent", type=float, help="Percentage (0-100) of visible objects to translate")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    parser.add_argument("--min-distance", type=float, default=0.3, help="Minimum translation distance")
    parser.add_argument("--max-distance", type=float, default=1.0, help="Maximum translation distance")
    args = parser.parse_args()

    distances = translate_random_visible_objects(
        args.scene_dir,
        args.output_dir,
        args.percent,
        seed=args.seed,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
    )

    print(f"Translated {len(distances)} object(s) in {args.output_dir}:")
    for name, distance in distances.items():
        print(f"  - {name}: {distance:.2f}m")


if __name__ == "__main__":
    main()
