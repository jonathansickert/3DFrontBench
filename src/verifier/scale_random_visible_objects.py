"""Randomly scale a subset of the visible furniture, uniformly per object.

Reads the "visible_furniture" list from metadata.json, picks a random n% of
those objects, and multiplies each one's scale by a random uniform factor
(same factor on x/y/z, so the object keeps its proportions) in scene.glb /
scene_bbox.glb. Position and orientation are unchanged. The furniture's
"scale" field in metadata.json is updated to match, and the original scale
plus the applied factor is recorded under metadata["scaled_furniture"] for
ground truth.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import trimesh

from src.util import make_transform


def _scale_objects_in_glb(glb_path: Path, transforms: dict[str, list]) -> None:
    if not glb_path.exists():
        return

    scene = trimesh.load(glb_path, process=False)
    for node_name, new_transform in transforms.items():
        scene.graph.update(frame_to=node_name, matrix=new_transform)
    scene.export(glb_path)


def scale_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float,
    seed: int | None = None,
    min_factor: float = 0.5,
    max_factor: float = 1.5,
) -> dict[str, float]:
    """Copy a scene to output_dir with n% of its visible objects uniformly rescaled.

    Args:
        scene_dir: Source scene directory (contains scene.glb, metadata.json, visible_furniture/).
        output_dir: Destination directory to write the modified scene to. Must not already exist.
        percent: Percentage (0-100) of visible objects to scale.
        seed: Optional RNG seed for reproducible selection and factors.
        min_factor: Minimum scale factor.
        max_factor: Maximum scale factor.

    Returns:
        Mapping of scaled object name to the applied scale factor.
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
    n_scale = round(len(visible_names) * percent / 100)

    rng = random.Random(seed)
    scaled_names = rng.sample(visible_names, n_scale)
    factors = {name: rng.uniform(min_factor, max_factor) for name in scaled_names}

    if not factors:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return {}

    new_transforms = {}
    scaled_furniture = []
    for item in metadata["furniture"]:
        name = item["name"]
        if name not in factors:
            continue
        factor = factors[name]
        original_scale = item["scale"]
        item["scale"] = [s * factor for s in original_scale]
        new_transforms[name] = make_transform(item["pos"], item["rot"], item["scale"]).tolist()
        scaled_furniture.append(
            {
                "name": name,
                "factor": factor,
                "original_scale": original_scale,
                "new_scale": item["scale"],
            }
        )
    metadata["scaled_furniture"] = scaled_furniture

    _scale_objects_in_glb(output_dir / "scene.glb", new_transforms)
    _scale_objects_in_glb(output_dir / "scene_bbox.glb", new_transforms)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return factors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument("percent", type=float, help="Percentage (0-100) of visible objects to scale")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    parser.add_argument("--min-factor", type=float, default=0.5, help="Minimum scale factor")
    parser.add_argument("--max-factor", type=float, default=1.5, help="Maximum scale factor")
    args = parser.parse_args()

    factors = scale_random_visible_objects(
        args.scene_dir,
        args.output_dir,
        args.percent,
        seed=args.seed,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
    )

    print(f"Scaled {len(factors)} object(s) in {args.output_dir}:")
    for name, factor in factors.items():
        print(f"  - {name}: {factor:.2f}x")


if __name__ == "__main__":
    main()
