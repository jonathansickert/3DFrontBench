"""Render Hypersim scene bounding boxes from metadata.

This module provides utilities to extract scene geometry and camera parameters from
Hypersim datasets and render them to images using PyRender.
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyrender
import trimesh
from PIL import Image


def download_and_extract_hyperism_scene(url: str, download_dir: str) -> None:
    """Download a Hypersim scene zip, extract it, and keep only essential directories.
    Keeps scene/_detail and scene/images/scene_cam_00_final_preview

    Args:
        url: URL to the scene zip file.
        download_dir: Directory to download and extract the scene to.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    scene_name = url.split("/")[-1].replace(".zip", "")

    if (download_dir / scene_name).is_dir():
        print(f"Scene {scene_name} already exists. Skipping...")
        return

    zip_path = download_dir / f"{scene_name}.zip"
    print(f"Downloading {scene_name}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {scene_name}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    zip_path.unlink()

    # Keep only scene/_detail and scene/images/scene_cam_00_final_preview for space reasons
    scene_dir = download_dir / scene_name
    images_dir = scene_dir / "images"
    for item in images_dir.iterdir():
        if not item.name == "scene_cam_00_final_preview":
            shutil.rmtree(item)


def download_hyperism_cam_metadata(download_dir: str) -> None:
    """Download Hypersim camera metadata CSV file.

    Args:
        download_dir: Directory to download the metadata file to.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/apple/ml-hypersim/main/contrib/mikeroberts3000/metadata_camera_parameters.csv"
    urllib.request.urlretrieve(url, download_dir / "metadata_camera_parameters.csv")


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
        scene_dir / "_detail/mesh/metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5"
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

    def is_valid_row(arr):
        if arr.ndim == 1:
            return np.isfinite(arr)
        else:
            arr_2d = arr.reshape(arr.shape[0], -1)
            return np.isfinite(arr_2d).all(axis=1)

    valid_mask = is_valid_row(extents) & is_valid_row(orientations) & is_valid_row(positions)

    extents = extents[valid_mask]
    orientations = orientations[valid_mask]
    positions = positions[valid_mask]

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
    with h5py.File(scene_dir / "_detail/cam_00/camera_keyframe_orientations.hdf5", "r") as f:
        camera_orientations = np.array(f["dataset"])

    with h5py.File(scene_dir / "_detail/cam_00/camera_keyframe_positions.hdf5", "r") as f:
        camera_positions = np.array(f["dataset"])

    camera_position_world = camera_positions[frame_id]
    R_world_from_cam = camera_orientations[frame_id]

    M_c2w = np.eye(4)
    M_c2w[:3, :3] = R_world_from_cam
    M_c2w[:3, 3] = camera_position_world

    # camera intrinsics
    # from https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/jupyter/02_rendering_hypersim_meshes_with_pytorch3d.ipynb
    camera_metadata = pd.read_csv(hyperism_cam_metadata_file, index_col="scene_name")
    df_ = camera_metadata.loc[scene_dir.name]

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
    scene_save_path: str,
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

    tm_scene.export(scene_save_path)

    return tm_scene, render


def hyperism_scene_render(
    scene_dir: str, frame_id: int, hyperism_cam_metadata_file: str, output_dir: str
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_name = Path(scene_dir).name
    render_save_path = output_dir / f"{scene_name}.png"
    scene_save_path = output_dir / f"{scene_name}.glb"

    scene, render = render_scene(
        extents=extents,
        orientations=orientations,
        positions=positions,
        M_c2w=M_c2w,
        fov_y=fov_y,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        render_save_path=str(render_save_path),
        scene_save_path=str(scene_save_path),
    )

    return scene, render


selected_hyperism_scene_urls = [
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_009_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_008_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_028_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_004_007.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_046_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_021_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_031_010.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_027_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_050_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_023_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_034_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_053_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_003_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_017_002.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_026_016.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_006_006.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_016_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_055_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_001_001.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_045_008.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_007_005.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_002_004.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_044_009.zip",
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/ai_005_010.zip",
]

if __name__ == "__main__":
    for scene_url in selected_hyperism_scene_urls:
        download_and_extract_hyperism_scene(scene_url, download_dir="./hyperism")
    download_hyperism_cam_metadata("./hyperism")