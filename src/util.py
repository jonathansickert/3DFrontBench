import json
import random
import shutil

import numpy as np
import trimesh
from pathlib import Path
import trimesh.transformations as tf
import pyrender


def blender_c2w_to_opencv(c2w_blender: np.ndarray) -> np.ndarray:
    """Convert Blender camera-to-world matrix to OpenCV convention."""
    # c2w_blender lives in Blender's Z-up world, but the GLB mesh is Y-up (Blender rotates
    # vertices by Rx(-90) on export). Pre-multiply to bring the camera into Y-up world space,
    # then post-multiply to convert Blender camera axes (Y-up, -Z forward) to OpenCV (Y-down, +Z forward).
    Rx_neg90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = Rx_neg90 @ c2w_blender @ flip_yz
    return c2w


def blender_c2w_to_pyrender(c2w_blender: np.ndarray) -> np.ndarray:
    Rx_neg90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    c2w = Rx_neg90 @ c2w_blender
    return c2w


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert camera-to-world matrix to get world-to-camera."""
    return np.linalg.inv(c2w)


def remove_textures(scene: trimesh.Scene):
    for _, geom in scene.geometry.items():
        if hasattr(geom.visual, "to_color"):
            geom.visual = geom.visual.to_color()


def load_scene(scene_path: Path):
    loaded = trimesh.load(scene_path, process=False)
    remove_textures(loaded)

    return loaded


def visible_vertex_mask(
    vertices: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    znear: float,
    zfar: float,
) -> np.ndarray:
    """Boolean mask: True where a vertex projects inside the camera frame."""
    ones = np.ones((len(vertices), 1))
    verts_h = np.hstack([vertices, ones])  # (N, 4)
    verts_cam = (w2c @ verts_h.T).T  # (N, 4)

    Z = verts_cam[:, 2]
    # avoid division by zero for vertices behind the camera
    valid_z = Z > 0

    u = np.where(valid_z, K[0, 0] * verts_cam[:, 0] / np.where(valid_z, Z, 1) + K[0, 2], -1)
    v = np.where(valid_z, K[1, 1] * verts_cam[:, 1] / np.where(valid_z, Z, 1) + K[1, 2], -1)

    return valid_z & (Z >= znear) & (Z <= zfar) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def make_transform(pos: list, rot: list, scale: list) -> np.ndarray:
    sx, sy, sz = scale
    S = np.diag([sx, sy, sz, 1.0])

    qx, qy, qz, qw = rot
    R = tf.quaternion_matrix([qw, qx, qy, qz])

    T = tf.translation_matrix(pos)

    return T @ R @ S


def load_metadata(scene_dir: Path) -> dict:
    with open(scene_dir / "metadata.json") as f:
        return json.load(f)


def furniture_by_name(metadata: dict) -> dict[str, dict]:
    return {item["name"]: item for item in metadata["furniture"]}


def resolve_percent(percent: float | None, rng: random.Random) -> float:
    """Resolve an optional percent argument.

    If percent is None, sample one of 0, 10, ..., 100 at random.
    """
    if percent is None:
        percent = rng.randint(0, 10) * 10

    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")

    return percent


def select_random_visible_furniture(metadata: dict, percent: float, rng: random.Random) -> list[str]:
    """Randomly sample n% of a scene's visible furniture names."""
    if not 0 <= percent <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")

    visible_names = list(metadata["visible_furniture"])
    n = round(len(visible_names) * percent / 100)

    return rng.sample(visible_names, n)


def compute_perturbation_score(
    magnitudes: dict[str, float], total_visible: int, max_magnitude: float = 1.0
) -> float:
    """A single scene-level score in [0, 1] for how much a scene got perturbed.

    Each per-object perturbation magnitude (e.g. degrees rotated, translation
    distance, |scale factor - 1|, or 1.0 per removed object) is divided by
    max_magnitude -- the largest magnitude that perturbation could have
    produced -- and clipped to 1.0, so the result is always in [0, 1]
    regardless of the raw units involved. The normalized magnitudes are then
    averaged over every visible object in the scene, not just the perturbed
    ones, so untouched objects implicitly contribute 0. This makes the score
    reflect both how many objects were touched and how strongly.
    """
    if total_visible == 0 or max_magnitude <= 0:
        return 0.0

    normalized = sum(min(magnitude / max_magnitude, 1.0) for magnitude in magnitudes.values())
    return normalized / total_visible


def circular_angle_distance(angle_deg: float) -> float:
    """Smallest-magnitude visual displacement of a rotation, in degrees, within [0, 180].

    Rotation wraps: spinning an object by 350 deg looks almost identical to
    spinning it by -10 deg, and 180 deg is the point of maximum visual
    difference (any further rotation starts looking more similar to the
    original again). This maps a raw sampled angle (of any magnitude, e.g.
    from resampling in [0, 360)) onto that perceptual scale.
    """
    remainder = angle_deg % 360.0
    return min(remainder, 360.0 - remainder)


def prepare_permuted_scene_dir(scene_dir: Path, output_dir: Path) -> Path:
    """Set up output_dir with just scene.glb and camera.json copied from scene_dir.

    Returns:
        Path to the copied scene.glb, ready to be edited in place.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shutil.copy2(scene_dir / "scene.glb", output_dir / "scene.glb")
    shutil.copy2(scene_dir / "camera.json", output_dir / "camera.json")

    return output_dir / "scene.glb"


def update_node_transforms(glb_path: Path, transforms: dict[str, np.ndarray]) -> None:
    """Overwrite the world transform of one or more nodes in a GLB scene, in place."""
    scene = trimesh.load(glb_path, process=False)
    for node_name, transform in transforms.items():
        scene.graph.update(frame_to=node_name, matrix=transform)
    scene.export(glb_path)


def remove_nodes(glb_path: Path, node_names: list[str]) -> None:
    """Delete one or more named nodes (and their geometry) from a GLB scene, in place."""
    scene = trimesh.load(glb_path, process=False)
    geom_names = {scene.graph[name][1] for name in node_names}
    scene.delete_geometry(geom_names)
    scene.export(glb_path)


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def norm_depth(depth):
    depth_vis = depth.copy()
    mask = depth_vis > 0
    depth_vis[mask] = (depth_vis[mask] - depth_vis[mask].min()) / (depth_vis[mask].max() - depth_vis[mask].min())
    depth_vis[~mask] = depth_vis[mask].max()
    depth_vis = (depth_vis * 255).astype(np.uint8)

    return depth_vis


def get_pyrender_cam(cam_params: dict):
    cam = pyrender.IntrinsicsCamera(
        fx=cam_params["fx"],
        fy=cam_params["fy"],
        cx=cam_params["cx"],
        cy=cam_params["cy"],
        znear=cam_params["znear"],
        zfar=cam_params["zfar"],
    )

    c2w_blender = np.array(cam_params["c2w_blender"])
    c2w = blender_c2w_to_pyrender(c2w_blender)

    return cam, c2w, cam_params["width"], cam_params["height"]


def get_light_positions(scene: trimesh.Scene) -> list[np.ndarray]:
    transforms = []
    for node in scene.graph.nodes_geometry:
        if "light" in node.lower():
            mat, geom_name = scene.graph[node]
            centroid_local = scene.geometry[geom_name].centroid
            pos = (mat @ np.append(centroid_local, 1))[:3]
            T = np.eye(4)
            T[:3, 3] = pos
            transforms.append(T)
    return transforms


def render_trimesh_scene(
    scene: trimesh.Scene,
    cam: pyrender.IntrinsicsCamera,
    c2w: np.array,
    width: int,
    height: int,
):
    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.3, 0.3, 0.3])
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    pyrender_scene.add(cam, pose=c2w)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    pyrender_scene.add(light, pose=c2w)

    point_light = pyrender.PointLight(color=np.ones(3), intensity=300.0)
    for T in get_light_positions(scene):
        pyrender_scene.add(point_light, pose=T)

    color, depth = renderer.render(pyrender_scene)
    depth = norm_depth(depth)
    return color, depth
