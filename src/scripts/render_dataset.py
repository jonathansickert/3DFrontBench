import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "blender" / "render_3d_front_images.py"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Source scene directory")
    args = parser.parse_args()

    for scene_dir in args.dataset_dir.iterdir():
        if scene_dir.name.startswith("."):
            continue

        glb_path = scene_dir / "scene.glb"
        cam_path = scene_dir / "camera.json"
        out_path = scene_dir / "color.png"

        cmd = [
            BLENDER_PATH,
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