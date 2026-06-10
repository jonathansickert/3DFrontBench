"""Sample a point cloud from the visible part of a scene mesh.

Visibility is defined as all mesh vertices that project within the camera frame
(image bounds) and lie between znear and zfar. Points are then sampled uniformly
from the sub-mesh formed by faces whose vertices are all visible.
"""

from dlinvc.dataset import Eval3DFrontDataset

import numpy as np
import trimesh


def _visible_vertex_mask(
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
    verts_h = np.hstack([vertices, ones])          # (N, 4)
    verts_cam = (w2c @ verts_h.T).T                # (N, 4)

    Z = verts_cam[:, 2]
    # avoid division by zero for vertices behind the camera
    valid_z = Z > 0

    u = np.where(valid_z, K[0, 0] * verts_cam[:, 0] / np.where(valid_z, Z, 1) + K[0, 2], -1)
    v = np.where(valid_z, K[1, 1] * verts_cam[:, 1] / np.where(valid_z, Z, 1) + K[1, 2], -1)

    return valid_z & (Z >= znear) & (Z <= zfar) & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def sample_visible_pointcloud(
    cam: dict,
    mesh: trimesh.Trimesh,
    n_samples: int = 10_000,
) -> np.ndarray:
    """Sample a point cloud from the camera-visible part of a mesh.

    Args:
        cam: Dictionary containing camera parameters.
        mesh: Trimesh object representing the scene geometry.
        n_samples: Number of surface points to sample from visible geometry.

    Returns:
        Point cloud as (M, 3) float32 array in world coordinates.
        M may be less than n_samples if the visible surface is small.
    """

    K = np.array(cam["K"], dtype=np.float64)
    # Blender camera: X right, Y up, -Z forward.
    # Flip Y and Z to get OpenCV convention (X right, Y down, +Z forward).
    c2w_blender = np.array(cam["c2w_blender"], dtype=np.float64)
    # c2w_blender lives in Blender's Z-up world, but the GLB mesh is Y-up (Blender rotates
    # vertices by Rx(-90) on export). Pre-multiply to bring the camera into Y-up world space,
    # then post-multiply to convert Blender camera axes (Y-up, -Z forward) to OpenCV (Y-down, +Z forward).
    Rx_neg90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = Rx_neg90 @ c2w_blender @ flip_yz

    w2c = np.linalg.inv(c2w)
    width: int = cam["width"]
    height: int = cam["height"]
    znear: float = cam["znear"]
    zfar: float = cam["zfar"]

    mesh = mesh.to_geometry()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)   # (V, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64)            # (F, 3)

    visible = _visible_vertex_mask(vertices, w2c, K, width, height, znear, zfar)

    face_visible = visible[faces].all(axis=1)

    visible_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces[face_visible],
        process=False,
    )

    points, _ = trimesh.sample.sample_surface(visible_mesh, n_samples)
    return points.astype(np.float32)

if __name__ == "__main__":
    dataset = Eval3DFrontDataset("./dataset_huggingface")

    sample = dataset[1]
    print(sample["scene_id"])
    cam = sample["camera"]
    mesh = sample["scene"]
    points = sample_visible_pointcloud(cam, mesh, n_samples=100_000)
    print(points.shape)

    np.savetxt("cloud.xyz", points, fmt="%.6f")
