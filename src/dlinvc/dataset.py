from pathlib import Path
import json

import trimesh
import pyrender
import numpy as np


def load_scene(scene_path: Path):
    loaded = trimesh.load(scene_path, process=False)
    for geom_name, geom in loaded.geometry.items():
        if hasattr(geom.visual, "to_color"):
            geom.visual = geom.visual.to_color()

    return loaded


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


class IndoorSceneDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        metadata_files = list(self.dataset_path.glob("*_metadata.json"))
        self.scenes = []

        for metadata_file in metadata_files:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            scene_name = metadata["scene_name"]
            bbox_name = metadata["scene_name_bbox"]

            scene_path = self.dataset_path / scene_name
            bbox_path = self.dataset_path / bbox_name

            scene = load_scene(scene_path)
            bbox = load_scene(bbox_path)

            cam, c2w, width, height = get_camera(self.dataset_path / f"{scene_name[:-4]}_camera.json")

            scene = {
                "scene": scene,
                "bbox": bbox,
                "metadata": metadata,
                "camera": cam,
                "c2w": c2w,
                "width": width,
                "height": height,
            }
            self.scenes.append(scene)

        print(f"Loaded '{len(self.scenes)}' scenes.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

    def __iter__(self):
        return iter(self.scenes)
