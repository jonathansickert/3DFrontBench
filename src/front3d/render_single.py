import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).parents[1] / "blender/render_scene.py"
MATERIAL_MODES = ("full_pbr", "texture_flat_lighting", "no_texture_flat_lighting")

assert RENDER_SCRIPT.exists(), RENDER_SCRIPT

def render_single(scene_dir: Path, material_mode: str = "full_pbr"):
    if material_mode not in MATERIAL_MODES:
        raise ValueError(f"Unknown material_mode: {material_mode!r}, expected one of {MATERIAL_MODES}")

    glb_path = scene_dir / "scene.glb"
    cam_path = scene_dir / "camera.json"
    out_path = scene_dir / f"color_{material_mode}.png"

    cmd = [
        BLENDER_PATH,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(glb_path),
        str(cam_path),
        str(out_path),
        material_mode,
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
    parser.add_argument(
        "--material-mode",
        choices=MATERIAL_MODES,
        default="full_pbr",
        help="Material/lighting quality mode",
    )
    args = parser.parse_args()

    render_single(args.scene_dir, material_mode=args.material_mode)
