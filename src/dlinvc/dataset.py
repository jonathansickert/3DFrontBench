from pathlib import Path
import json

from PIL import Image
import trimesh


def load_scene(scene_path: Path):
    loaded = trimesh.load(scene_path, process=False)
    for geom_name, geom in loaded.geometry.items():
        if hasattr(geom.visual, "to_color"):
            geom.visual = geom.visual.to_color()

    return loaded


class Eval3DFrontDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

        self.scenes = []
        subdirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        subdirs = [d for d in subdirs if d.name != ".cache"]

        for subdir in subdirs:
            scene_path = subdir / "scene.glb"
            bbox_path = subdir / "scene_bbox.glb"
            metadata_path = subdir / "metadata.json"
            camera_path = subdir / "camera.json"

            color_path = subdir / "color.png"
            color_bbox_path = subdir / "color_bbox.png"
            depth_path = subdir / "depth.png"
            depth_bbox_path = subdir / "depth_bbox.png"

            scene = load_scene(scene_path)
            bbox = load_scene(bbox_path)

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            with open(camera_path, "r") as f:
                cam = json.load(f)

            scene_data = {
                "scene_id" : subdir.name,
                "scene": scene,
                "bbox": bbox,
                "metadata": metadata,
                "camera": cam,
                "color": Image.open(color_path),
                "color_bbox": Image.open(color_bbox_path),
                "depth": Image.open(depth_path),
                "depth_bbox": Image.open(depth_bbox_path),
            }
            self.scenes.append(scene_data)

        print(f"Loaded '{len(self.scenes)}' scenes.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

    def __iter__(self):
        return iter(self.scenes)
