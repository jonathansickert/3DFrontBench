"""Randomly rotate a subset of the visible furniture about the vertical axis.

Reads the "visible_furniture" list from metadata.json, picks a random n% of
those objects, and spins each in place (position unchanged) by a random angle
about the vertical (world Y, since the GLB is Y-up) axis in scene.glb /
scene_bbox.glb. The furniture's "rot" quaternion in metadata.json is updated
to match, and the original rotation plus the applied delta angle is recorded
under metadata["rotated_furniture"] for ground truth.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tf


def _rotate_transform_in_place(transform: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a node's world transform about the vertical (Y) axis through its own position."""
    ry3 = tf.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
    new_transform = transform.copy()
    new_transform[:3, :3] = ry3 @ transform[:3, :3]
    return new_transform


def _rotate_objects_in_glb(glb_path: Path, angles_rad: dict[str, float]) -> None:
    if not glb_path.exists():
        return

    scene = trimesh.load(glb_path, process=False)
    for node_name, angle_rad in angles_rad.items():
        transform, _ = scene.graph[node_name]
        new_transform = _rotate_transform_in_place(transform, angle_rad)
        scene.graph.update(frame_to=node_name, matrix=new_transform)
    scene.export(glb_path)


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
    """Copy a scene to output_dir with n% of its visible objects rotated about the vertical axis.

    Args:
        scene_dir: Source scene directory (contains scene.glb, metadata.json, visible_furniture/).
        output_dir: Destination directory to write the modified scene to. Must not already exist.
        percent: Percentage (0-100) of visible objects to rotate.
        seed: Optional RNG seed for reproducible selection and angles.
        min_degrees: Minimum rotation magnitude, in degrees.
        max_degrees: Maximum rotation magnitude, in degrees.

    Returns:
        Mapping of rotated object name to the applied angle, in degrees.
    """
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")

    if output_dir.exists():
       shutil.rmtree(output_dir)
    
    shutil.copytree(scene_dir, output_dir)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    visible_names = list(metadata["visible_furniture"])
    n_rotate = round(len(visible_names) * percent / 100)

    rng = random.Random(seed)
    rotated_names = rng.sample(visible_names, n_rotate)
    angles_deg = {name: rng.uniform(min_degrees, max_degrees) for name in rotated_names}

    if not angles_deg:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return {}

    angles_rad = {name: np.deg2rad(angle) for name, angle in angles_deg.items()}
    _rotate_objects_in_glb(output_dir / "scene.glb", angles_rad)

    rotated_furniture = []
    for item in metadata["furniture"]:
        name = item["name"]
        if name not in angles_deg:
            continue
        original_rot = item["rot"]
        item["rot"] = _rotate_quaternion(original_rot, angles_rad[name])
        rotated_furniture.append(
            {
                "name": name,
                "angle_degrees": angles_deg[name],
                "original_rot": original_rot,
                "new_rot": item["rot"],
            }
        )
    metadata["rotated_furniture"] = rotated_furniture

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return angles_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory for the modified scene")
    parser.add_argument("percent", type=float, help="Percentage (0-100) of visible objects to rotate")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible selection")
    parser.add_argument("--min-degrees", type=float, default=0.0, help="Minimum rotation magnitude")
    parser.add_argument("--max-degrees", type=float, default=360.0, help="Maximum rotation magnitude")
    args = parser.parse_args()

    angles = rotate_random_visible_objects(
        args.scene_dir,
        args.output_dir,
        args.percent,
        seed=args.seed,
        min_degrees=args.min_degrees,
        max_degrees=args.max_degrees,
    )

    print(f"Rotated {len(angles)} object(s) in {args.output_dir}:")
    for name, angle in angles.items():
        print(f"  - {name}: {angle:.1f} deg")


if __name__ == "__main__":
    main()
