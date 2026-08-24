import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).parent / "render_nfinite_perturbed.py"

assert RENDER_SCRIPT.exists(), RENDER_SCRIPT


def render_perturbed(blend_path: str, out_path: str, perturbations: str):
    cmd = [
        BLENDER_PATH,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(blend_path),
        str(out_path),
        "--perturbations",
        perturbations,
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Rendering failed: {result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("blend_path", type=Path, help="Source scene")
    parser.add_argument("out_path", type=Path)
    parser.add_argument(
        "--perturbations",
        type=str,
        required=True,
        help=(
            "Perturbations for this scene as a JSON-formatted string, e.g. "
            '\'{"SOCKET": {"perturbation_type": "rotation", "args": [["x"], 52]}}\''
        ),
    )

    args = parser.parse_args()

    render_perturbed(args.blend_path, args.out_path, perturbations=args.perturbations)
