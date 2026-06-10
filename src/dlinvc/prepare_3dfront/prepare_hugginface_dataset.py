"""
Prepare 3D-FRONT dataset for Hugging Face by organizing and rendering scenes.

This script processes generated 3D scene data to create a structured Hugging Face dataset.
It performs the following steps:

1. **Dataset Organization**: Organizes scene files, bounding boxes, metadata, and camera
   parameters into a consistent directory structure
2. **Scene Rendering**: Loads 3D scenes and renders them to 2D RGB images and depth maps
   using pyrender with proper camera parameters
3. **Coordinate Conversion**: Handles Blender coordinate system (Z-up) conversion to
   pyrender's Y-up convention

Output structure: Each scene gets a dedicated directory containing:
- scene.glb: Full 3D geometry
- scene_bbox.glb: Bounding box geometry
- metadata.json: Scene and furniture information
- camera.json: Camera intrinsics and pose parameters
- color.png: Rendered RGB image
- depth.png: Rendered depth map
"""

from pathlib import Path
import json
import shutil
from PIL import Image
import pyrender
import trimesh
import numpy as np


def load_scene(scene_path: Path):
    loaded = trimesh.load(scene_path, process=False)
    for geom_name, geom in loaded.geometry.items():
        if hasattr(geom.visual, "to_color"):
            geom.visual = geom.visual.to_color()

    return loaded


def norm_depth(depth):
    depth_vis = depth.copy()
    mask = depth_vis > 0
    depth_vis[mask] = (depth_vis[mask] - depth_vis[mask].min()) / (depth_vis[mask].max() - depth_vis[mask].min())
    depth_vis[~mask] = depth_vis[mask].max()
    depth_vis = (depth_vis * 255).astype(np.uint8)

    return depth_vis


def get_camera(camera_path: Path):
    with open(camera_path, "r") as f:
        cam_params = json.load(f)

    cam = pyrender.IntrinsicsCamera(
        fx=cam_params["fx"],
        fy=cam_params["fy"],
        cx=cam_params["cx"],
        cy=cam_params["cy"],
        znear=cam_params["znear"],
        zfar=cam_params["zfar"],
    )

    # blender uses z-up => convert to y-up for pyrender
    c2w = np.array(cam_params["c2w_blender"])
    Rx_neg90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    c2w_yup = Rx_neg90 @ c2w

    return cam, c2w_yup, cam_params["width"], cam_params["height"]


if __name__ == "__main__":
    dataset_path = Path("./dataset")
    huggingface_dataset = Path("./dataset_huggingface")
    huggingface_dataset.mkdir(exist_ok=True)

    metadata_files = list(dataset_path.glob("*_metadata.json"))
    for metadata_file in metadata_files:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        scene_id = metadata["scene_id"]
        room_name = metadata["room_id"]

        dst_dir = huggingface_dataset / f"{scene_id}_{room_name}"
        dst_dir.mkdir(exist_ok=True)

        shutil.copy(dataset_path / metadata["scene_name"], dst_dir / "scene.glb")
        shutil.copy(dataset_path / metadata["scene_name_bbox"], dst_dir / "scene_bbox.glb")

        new_metadata = {
            "scene_id": scene_id,
            "room_name": room_name,
            "furniture": metadata["furniture"],
        }

        with open(dst_dir / "metadata.json", "w") as f:
            json.dump(new_metadata, f, indent=2)

        shutil.copy(dataset_path / f"{scene_id}_{room_name}_camera.json", dst_dir / "camera.json")

    for subdir in huggingface_dataset.iterdir():
        trimesh_scene = load_scene(subdir / "scene.glb")
        trimesh_bbox_scene = load_scene(subdir / "scene_bbox.glb")
        pyrender_scene = pyrender.Scene.from_trimesh_scene(trimesh_scene, ambient_light=[0.3, 0.3, 0.3])
        pyrender_bbox_scene = pyrender.Scene.from_trimesh_scene(trimesh_bbox_scene, ambient_light=[0.3, 0.3, 0.3])

        cam, c2w, width, height = get_camera(subdir / "camera.json")

        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        pyrender_scene.add(cam, pose=c2w)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        pyrender_scene.add(light, pose=c2w)

        pyrender_bbox_scene.add(cam, pose=c2w)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        pyrender_bbox_scene.add(light, pose=c2w)

        color, depth = renderer.render(pyrender_scene)
        color_bbox, depth_bbox = renderer.render(pyrender_bbox_scene)

        Image.fromarray(color).save(subdir / "color.png")
        Image.fromarray(norm_depth(depth)).save(subdir / "depth.png")
        Image.fromarray(color_bbox).save(subdir / "color_bbox.png")
        Image.fromarray(norm_depth(depth_bbox)).save(subdir / "depth_bbox.png")
