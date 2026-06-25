import subprocess
import os
from pathlib import Path

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "blender" / "render_3d_front_images.py"
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"

VARIANTS = [
    ("scene.glb", "color.png"),
    ("scene_scaffold.glb", "color_scaffold.png"),
    ("scene_bbox.glb", "color_bbox.png"),
]


def render_glb_file(blender: str, scene_dir: Path, glb_name: str, out_name: str):
    glb_path = scene_dir / glb_name
    cam_path = scene_dir / "camera.json"
    out_path = scene_dir / out_name

    if not glb_path.exists() or not cam_path.exists():
        raise FileNotFoundError(f"Scene does not exist: {glb_path}")

    cmd = [
        blender,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(glb_path),
        str(cam_path),
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0
    if not ok:
        raise RuntimeError(f"Rendering failed: {result.stderr}")


def main():
    scene_dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name != ".cache")
    tasks = [(scene_dir, glb_name, out_name) for scene_dir in scene_dirs for glb_name, out_name in VARIANTS]

    for scene_dir, glb_name, out_name in tasks:
        print(f"Rendering {scene_dir.name}/{glb_name} ...")
        try:
            render_glb_file(BLENDER_PATH, scene_dir, glb_name, out_name)
            print("Rendering succesful")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Rendering failed: {e}")


if __name__ == "__main__":
    main()
