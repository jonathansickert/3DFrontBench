import subprocess
import os
from pathlib import Path

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).resolve().parents[2] / "src" / "blender" / "export_scaffold_blend.py"
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def export_scaffold_blendfile(blender: str, scene_dir: Path):
    cmd = [
        blender,
        "--background",
        "--python", str(RENDER_SCRIPT),
        "--",
        str(scene_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = result.returncode == 0
    if not ok:
        raise RuntimeError(f"Rendering failed: {result.stderr}")


def main():
    scene_dirs = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name != ".cache")

    for scene_dir in scene_dirs:
        print(f"Export scaffold .blend file for {scene_dir.name}")
        try:
            export_scaffold_blendfile(BLENDER_PATH, scene_dir)
        except RuntimeError as e:
            print(f"Export failed: {e}")
            


if __name__ == "__main__":
    main()
