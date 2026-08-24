import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).parents[1] / "blender/render_scene_top_down.py"

assert RENDER_SCRIPT.exists(), RENDER_SCRIPT

def render_top_down_single(scene_dir: Path):
    glb_path = scene_dir / "scene.glb"
    metadata_path = scene_dir / "metadata.json"
    out_path = scene_dir / "color_top_down.png"

    cmd = [
        BLENDER_PATH,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(glb_path),
        str(metadata_path),
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

    render_top_down_single(args.scene_dir)
