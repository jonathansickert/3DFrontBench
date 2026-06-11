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

def _match_geometries(scene: trimesh.Scene, metadata: dict) -> list[trimesh.Trimesh]:
    """Match each metadata furniture entry to its world-space geometry.

    The GLB bakes all node transforms into vertex positions (every scene graph
    transform is identity). When the same furniture name appears multiple times,
    the GLTF has only one named node — other instances are stored as anonymous
    meshes. Matching by centroid-to-pos distance recovers the correct geometry
    for every instance.

    Returns a list parallel to metadata["furniture"].
    """
    geom_centroids = {
        name: np.asarray(g.vertices, dtype=np.float64).mean(axis=0)
        for name, g in scene.geometry.items()
    }
    centroid_arr = np.stack(list(geom_centroids.values()))  # (G, 3)
    geom_names = list(geom_centroids.keys())

    matched = []
    for item in metadata["furniture"]:
        pos = np.array(item["pos"], dtype=np.float64)
        dists = np.linalg.norm(centroid_arr - pos, axis=1)
        best = geom_names[int(dists.argmin())]
        matched.append(scene.geometry[best])
    return matched


def get_visible_objects(
    scene: trimesh.Scene,
    metadata: dict,
    cam: dict,
) -> list[tuple[dict, trimesh.Trimesh]]:
    """Return (furniture_item, geometry) for each visible object."""
    K = np.array(cam["K"], dtype=np.float64)

    c2w = blender_c2w_to_opencv(np.array(cam["c2w_blender"], dtype=np.float64))
    w2c = c2w_to_w2c(c2w)
    width: int = cam["width"]
    height: int = cam["height"]
    znear: float = cam["znear"]
    zfar: float = cam["zfar"]

    geometries = _match_geometries(scene, metadata)

    results = []
    for item, geom in zip(metadata["furniture"], geometries):
        # Vertices are already in world space (transforms baked on GLB export)
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
    with open(scene_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(scene_dir / "camera.json") as f:
        cam = json.load(f)

    visible_objects = []

    for visible_item, geom in get_visible_objects(scene, metadata, cam):
        name = visible_item["name"]
        label = visible_item["label"]
        mesh = geom.copy()

        visible_objects.append({
            "name" : name,
            "label" : label,
            "mesh" : mesh,
        })

    return visible_objects


if __name__ == "__main__":
    visible_objects = extract_visible_objects(
        Path("/Users/jonathansickert/git/DLinVC/dataset_huggingface/0f661df2-0f41-47a4-830c-7444f4a33a03_LivingDiningRoom-12554")
    )

    for visible_object in visible_objects:
        print(visible_object["name"])
