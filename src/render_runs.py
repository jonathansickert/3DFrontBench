from pathlib import Path
import os
import subprocess

run_dir = Path("/Users/jonathansickert/Downloads/runs_kimi2.7")
dataset_dir = Path("/Users/jonathansickert/git/3DFrontBench/dataset")

BLENDER_CMD = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = "/Users/jonathansickert/git/3DFrontBench/src/blender/render_vlm_output.py"

OUTPUT = Path("./runs_kimi/")
OUTPUT.mkdir(exist_ok=True, parents=True)


for run in run_dir.iterdir():
    if not run.is_dir():
        continue

    referecence_image = dataset_dir / run.name / "color.png"
    cam_path = dataset_dir / run.name / "camera.json"
    outpath = OUTPUT / run.name
    outpath.mkdir(exist_ok=True, parents=True)

    for step in run.iterdir():
        if step.name == ".ipynb_checkpoints":
            continue

        blendfile = step / "state.blend"
        out_path = outpath / f"color_{step.name}"

        cmd = [
            BLENDER_CMD,
            "--background",
            "--python",
            str(RENDER_SCRIPT),
            "--",
            str(blendfile),
            str(cam_path),
            str(out_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
