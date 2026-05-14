"""Render Hypersim scene bounding boxes from metadata.

This module provides utilities to extract scene geometry and camera parameters from
Hypersim datasets and render them to images using PyRender.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyrender
import trimesh
from PIL import Image


def extract_hyperism_scene_boundary_boxes(scene_dir: str) -> tuple[np.array, np.array, np.array]:
    """Extract bounding box parameters from Hypersim scene.

    Args:
        scene_dir: Path to the Hypersim scene directory.

    Returns:
        Tuple of (extents, orientations, positions) arrays.
    """
    scene_dir: Path = Path(scene_dir)
    if not scene_dir.exists():
        raise ValueError("Scene does not exist.")

    with h5py.File(
        scene_dir / "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5"
    ) as f:
        extents = np.array(f["dataset"])

    with h5py.File(
        scene_dir / "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5"
    ) as f:
        orientations = np.array(f["dataset"])

    with h5py.File(
        scene_dir / "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5"
    ) as f:
        positions = np.array(f["dataset"])

    return extents, orientations, positions


def extract_hyperism_scene_camera_params(
    scene_dir: str, frame_id: int, hyperism_cam_metadata_file: str
) -> tuple[np.array, float, int, int]:
    """Extract camera extrinsics and intrinsics for a given frame.

    Args:
        scene_dir: Path to the Hypersim scene directory.
        frame_id: Frame index to extract camera parameters for.
        hyperism_cam_metadata_file: Path to camera metadata CSV file.

    Returns:
        Tuple of (M_c2w, fov_y, width_pixels, height_pixels).
    """

    scene_dir: Path = Path(scene_dir)
    if not scene_dir.exists():
        raise ValueError("Scene does not exist.")

    # camera extrinsics
    with h5py.File("../hyperism/ai_001_001/_detail/cam_00/camera_keyframe_orientations.hdf5", "r") as f:
        camera_orientations = np.array(f["dataset"])

    with h5py.File("../hyperism/ai_001_001/_detail/cam_00/camera_keyframe_positions.hdf5", "r") as f:
        camera_positions = np.array(f["dataset"])

    camera_position_world = camera_positions[frame_id]
    R_world_from_cam = camera_orientations[frame_id]

    M_c2w = np.eye(4)
    M_c2w[:3, :3] = R_world_from_cam
    M_c2w[:3, 3] = camera_position_world

    # camera intrinsics
    # from https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/jupyter/02_rendering_hypersim_meshes_with_pytorch3d.ipynb
    camera_metadata = pd.read_csv(hyperism_cam_metadata_file, index_col="scene_name")
    df_ = camera_metadata.loc["ai_001_001"]

    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])

    if df_["use_camera_physical"]:
        fov_x = df_["camera_physical_fov"]
    else:
        fov_x = df_["settings_camera_fov"]

    fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x / 2.0) / width_pixels)

    return M_c2w, fov_y, width_pixels, height_pixels


def render_scene(
    extents: np.array,
    orientations: np.array,
    positions: np.array,
    M_c2w: np.array,
    fov_y: float,
    width_pixels: int,
    height_pixels: int,
    render_save_path: str,
) -> tuple[trimesh.Scene, Image.Image]:
    """Render bounding boxes in a scene and save the image.

    Args:
        extents: Box dimensions for each object.
        orientations: Rotation matrices for each object.
        positions: Center positions for each object.
        M_c2w: Camera-to-world transformation matrix.
        fov_y: Vertical field of view in radians.
        width_pixels: Image width in pixels.
        height_pixels: Image height in pixels.
        render_save_path: Path to save the rendered image.

    Returns:
        Tuple of (trimesh scene, rendered PIL Image).
    """

    # build scene
    boxes = []
    for i in range(extents.shape[0]):
        transform = np.eye(4)
        transform[:3, :3] = orientations[i]
        transform[:3, 3] = positions[i]

        box = trimesh.creation.box(extents=np.array(extents[i]), transform=transform)
        boxes.append(box)

    tm_scene = trimesh.util.concatenate(boxes)

    # render scene
    pr_mesh = pyrender.Mesh.from_trimesh(tm_scene)
    pr_scene = pyrender.Scene()
    pr_scene.add(pr_mesh)

    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=width_pixels / height_pixels)
    pr_scene.add(camera, pose=M_c2w)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    pr_scene.add(light, pose=M_c2w)

    r = pyrender.OffscreenRenderer(width_pixels, height_pixels)
    color, depth = r.render(pr_scene)

    save_path = Path(render_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    render = Image.fromarray(color)
    render.save(save_path)

    return tm_scene, render


def hyperism_scene_render(
    scene_dir: str, frame_id: int, hyperism_cam_metadata_file: str, render_save_path: str
) -> tuple[trimesh.Scene, Image.Image]:
    """Render a Hypersim scene with bounding boxes for a specific frame.

    Args:
        scene_dir: Path to the Hypersim scene directory.
        frame_id: Frame index to render.
        hyperism_cam_metadata_file: Path to camera metadata CSV file.
        render_save_path: Path to save the rendered image.

    Returns:
        Tuple of (trimesh scene, rendered PIL Image).
    """
    extents, orientations, positions = extract_hyperism_scene_boundary_boxes(scene_dir=scene_dir)
    M_c2w, fov_y, width_pixels, height_pixels = extract_hyperism_scene_camera_params(
        scene_dir=scene_dir, frame_id=frame_id, hyperism_cam_metadata_file=hyperism_cam_metadata_file
    )

    scene, render = render_scene(
        extents=extents,
        orientations=orientations,
        positions=positions,
        M_c2w=M_c2w,
        fov_y=fov_y,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        render_save_path=render_save_path,
    )

    return scene, render
