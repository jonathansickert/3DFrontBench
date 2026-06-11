import numpy as np
import trimesh
from pathlib import Path
import trimesh.transformations as tf


def blender_c2w_to_opencv(c2w_blender: np.ndarray) -> np.ndarray:
    """Convert Blender camera-to-world matrix to OpenCV convention."""
    # c2w_blender lives in Blender's Z-up world, but the GLB mesh is Y-up (Blender rotates
    # vertices by Rx(-90) on export). Pre-multiply to bring the camera into Y-up world space,
    # then post-multiply to convert Blender camera axes (Y-up, -Z forward) to OpenCV (Y-down, +Z forward).
    Rx_neg90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = Rx_neg90 @ c2w_blender @ flip_yz
    return c2w


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Invert camera-to-world matrix to get world-to-camera."""
    return np.linalg.inv(c2w)


def load_scene(scene_path: Path):
    loaded = trimesh.load(scene_path, process=False)
    for geom_name, geom in loaded.geometry.items():
        if hasattr(geom.visual, "to_color"):
            geom.visual = geom.visual.to_color()

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
