"""Randomly rotate a subset of the visible furniture about the vertical axis.

Reads the "visible_furniture" list from metadata.json, picks a random n% of
those objects, and spins each in place (position unchanged) by a random angle
about the vertical (world Y, since the GLB is Y-up) axis in scene.glb. Writes
the result (scene.glb + camera.json only) to a new output directory.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import trimesh.transformations as tf

from src.util import (
    furniture_by_name,
    load_metadata,
    make_transform,
    prepare_permuted_scene_dir,
    select_random_visible_furniture,
    update_node_transforms,
    write_json,
)


def _rotate_quaternion(rot: list[float], angle_rad: float) -> list[float]:
    """Apply a Y-axis rotation on top of a [qx, qy, qz, qw] quaternion, same convention as make_transform."""
    qx, qy, qz, qw = rot
    delta = tf.quaternion_about_axis(angle_rad, [0, 1, 0])  # [w, x, y, z]
    new_q = tf.quaternion_multiply(delta, [qw, qx, qy, qz])  # [w, x, y, z]
    w, x, y, z = new_q
    return [x, y, z, w]


def rotate_random_visible_objects(
    scene_dir: Path,
    output_dir: Path,
    percent: float,
    seed: int | None = None,
    min_degrees: float = 0.0,
    max_degrees: float = 360.0,
) -> dict[str, float]:
    """Rotate n% of a scene's visible objects about the vertical axis, in place.

    Args:
        scene_dir: Source scene directory (contains scene.glb, camera.json, metadata.json).
        output_dir: Destination directory for the modified scene. Overwritten if it exists.
        percent: Percentage (0-100) of visible objects to rotate.
        seed: Optional RNG seed for reproducible selection and angles.
        min_degrees: Minimum rotation magnitude, in degrees.
        max_degrees: Maximum rotation magnitude, in degrees.

    Returns:
        rotations: mapping of rotated furniture name -> applied angle, in degrees.
    """
    metadata = load_metadata(scene_dir)
    furniture = furniture_by_name(metadata)

    rng = random.Random(seed)
    selected = select_random_visible_furniture(metadata, percent, rng)
    rotations = {name: rng.uniform(min_degrees, max_degrees) for name in selected}

    scene_glb_path = prepare_permuted_scene_dir(scene_dir, output_dir)

    if rotations:
        new_transforms = {}
        for name, angle_deg in rotations.items():
            item = furniture[name]
            new_rot = _rotate_quaternion(item["rot"], np.deg2rad(angle_deg))
            new_transforms[name] = make_transform(item["pos"], new_rot, item["scale"])
        update_node_transforms(scene_glb_path, new_transforms)

    write_json(output_dir / "rotations.json", rotations)

    return rotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument("percent", type=float, help="Percentage (0-100) of visible objects to rotate")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    parser.add_argument("--min-degrees", type=float, default=0.0, help="Minimum rotation magnitude")
    parser.add_argument("--max-degrees", type=float, default=360.0, help="Maximum rotation magnitude")
    args = parser.parse_args()

    rotations = rotate_random_visible_objects(
        args.scene_dir,
        args.output_dir,
        args.percent,
        seed=args.seed,
        min_degrees=args.min_degrees,
        max_degrees=args.max_degrees,
    )

    print(f"Rotated {len(rotations)} object(s) in {args.output_dir}:")
    for name, angle in rotations.items():
        print(f"  - {name}: {angle:.1f} deg")


if __name__ == "__main__":
    main()
