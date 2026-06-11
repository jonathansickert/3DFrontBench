"""Extract furniture objects visible from the camera in a scene.

Visibility: at least one vertex of the object's mesh, after applying its world
transform, projects inside the camera image bounds and lies within [znear, zfar].

Camera convention: camera.json stores `c2w_blender` in Blender's Z-up frame.
The GLB is Y-up (Blender applies Rx(-90) on export). We reproduce the same
axis correction used in sample_visible_pointcloud.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from dlinvc.util import blender_c2w_to_opencv, c2w_to_w2c, visible_vertex_mask, load_scene


def get_visible_objects(
    scene: trimesh.Scene,
    cam: dict,
    metadata: dict,
) -> list[tuple[dict, trimesh.Trimesh]]:
    """Return (furniture_item, geometry) for each visible object."""
    K = np.array(cam["K"], dtype=np.float64)

    c2w = blender_c2w_to_opencv(np.array(cam["c2w_blender"], dtype=np.float64))
    w2c = c2w_to_w2c(c2w)
    width: int = cam["width"]
    height: int = cam["height"]
    znear: float = cam["znear"]
    zfar: float = cam["zfar"]

    results = []
    for item in metadata["furniture"]:
        node_name = item["name"]
        _, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        vertices = np.asarray(geom.vertices, dtype=np.float64)
        if visible_vertex_mask(vertices, w2c, K, width, height, znear, zfar).any():
            results.append((item, geom))

    return results


def extract_visible_objects(scene_dir: Path) -> list[dict]:
    """Return metadata entries for furniture visible from the camera.

    Args:
        scene_dir: Path to a scene subdirectory containing scene.glb,
                   metadata.json and camera.json.

    Returns:
        List of furniture dicts (from metadata["furniture"]) that are visible.
    """

    scene = load_scene(scene_dir / "scene.glb")
    with open(scene_dir / "camera.json") as f:
        cam = json.load(f)

    with open(scene_dir / "metadata.json") as f:
        metadata = json.load(f)

    visible_objects = []

    for visible_item, geom in get_visible_objects(scene, cam, metadata):
        name = visible_item["name"]
        label = visible_item["label"]
        mesh = geom.copy()

        visible_objects.append(
            {
                "name": name,
                "label": label,
                "mesh": mesh,
            }
        )

    return visible_objects


if __name__ == "__main__":
    visible_objects = extract_visible_objects(
        Path(
            "/home/jonathansickert/git/DLinVC/dataset_huggingface/0f661df2-0f41-47a4-830c-7444f4a33a03_LivingDiningRoom-12554/"
        )
    )

    for visible_object in visible_objects:
        print(visible_object["name"])
