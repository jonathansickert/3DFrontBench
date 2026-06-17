"""Sample a point cloud from the visible part of a scene mesh.

Visibility is defined as all mesh vertices that project within the camera frame
(image bounds) and lie between znear and zfar. Points are then sampled uniformly
from the sub-mesh formed by faces whose vertices are all visible.
"""

from dataset import Eval3DFrontDataset
from util import blender_c2w_to_opencv, c2w_to_w2c, visible_vertex_mask

import numpy as np
import trimesh


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
    c2w_blender = np.array(cam["c2w_blender"], dtype=np.float64)
    c2w = blender_c2w_to_opencv(c2w_blender)
    w2c = c2w_to_w2c(c2w)

    width: int = cam["width"]
    height: int = cam["height"]
    znear: float = cam["znear"]
    zfar: float = cam["zfar"]

    mesh = mesh.to_geometry()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)  # (V, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64)  # (F, 3)

    visible = visible_vertex_mask(vertices, w2c, K, width, height, znear, zfar)
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

    np.savetxt("outputs/cloud.xyz", points, fmt="%.6f")
