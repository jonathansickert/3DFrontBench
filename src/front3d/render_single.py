import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path("src/blender/render_scene.py")

def render_single(scene_dir: Path):
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

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    ok = result.returncode == 0
    if not ok:
        raise RuntimeError(f"Rendering failed: {result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    args = parser.parse_args()

    render_single(args.scene_dir)

    