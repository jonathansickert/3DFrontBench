import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "blender" / "render_3d_front_images.py"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    args = parser.parse_args()

    glb_path = args.scene_dir / "scene.glb"
    cam_path = args.scene_dir / "camera.json"
    out_path = args.scene_dir / "color.png"

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