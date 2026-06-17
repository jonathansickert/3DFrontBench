"""Render a point cloud image from the dataset camera position.

Uses the same camera pose convention as prepare_hugginface_dataset.py:
  pose = Rx_neg90 @ c2w_blender  (Z-up Blender world → Y-up pyrender world)
"""

import numpy as np
import pyrender
from PIL import Image
from dataset import Eval3DFrontDataset
from sample_visible_pointcloud import sample_visible_pointcloud


def render_pointcloud(
    points: np.ndarray,
    cam: dict,
) -> Image.Image:
    """Render a point cloud from the dataset camera pose.

    Args:
        points: (N, 3) float array of world-space points (Y-up world, matching GLB).
        cam: Camera dict from camera.json.

    Returns:
        Rendered PIL image.
    """
    camera = pyrender.IntrinsicsCamera(
        fx=cam["fx"],
        fy=cam["fy"],
        cx=cam["cx"],
        cy=cam["cy"],
        znear=cam["znear"],
        zfar=cam["zfar"],
    )

    # Same convention as the dataset rendering script
    Rx_neg90 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    pose = Rx_neg90 @ np.array(cam["c2w_blender"], dtype=np.float64)

    colors = np.full((len(points), 3), 255, dtype=np.uint8)
    pc_mesh = pyrender.Mesh.from_points(points.astype(np.float32), colors=colors)

    scene = pyrender.Scene(bg_color=[20, 20, 20, 255], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pc_mesh)
    scene.add(camera, pose=pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=cam["width"], viewport_height=cam["height"])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    renderer.delete()

    return Image.fromarray(color)


if __name__ == "__main__":
    dataset = Eval3DFrontDataset("./dataset_huggingface")
    sample = dataset[0]
    print(f"scene: {sample['scene_id']}")

    points = sample_visible_pointcloud(sample["camera"], sample["scene"], n_samples=100_000)
    print(f"sampled {len(points)} points")

    render = render_pointcloud(points, sample["camera"])

    out_path = "./outputs/pointcloud_render.png"
    render.save(out_path)
    print(f"saved rendered image to {out_path}")
