"""Build a randomly-sampled perturbation-detection dataset.

For every scene in --objects-path, samples --num-variants independent perturbation specs (one
draw per object: rotation/scale/placement/count, or "none") and materializes each as a
self-contained entry directory:

<dataset_dir>/<scene>_<variant>/
    labels.json         -- digit legend for this scene, copied from <labeled_dir>/labels.json
    labeled.png         -- unperturbed reference render, copied from <labeled_dir>/
    perturbations.json  -- {object_label: {"perturbation_type": ..., "args": ...}}
    perturbed.png       -- rendered from <blend_dir>/<scene>.blend with that spec applied
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from src.nfinite.render_single_labeled import OBJECTS_JSON
from src.nfinite.render_single_perturbed import render_perturbed

REPO_ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = REPO_ROOT / "NFINITE"
LABELED_DIR = REPO_ROOT / "NFINITE_labeled"
DATASET_DIR = REPO_ROOT / "dataset"

MIN_K = 0
MAX_K = 3

MIN_ROTATION = 30
MIN_K_AXIS = 1
MAX_K_AXIS = 2

SCALE = [0.67, 0.8, 1.25, 1.5]
TRANSLATION = [0.67, 0.8, 1.25, 1.5]


def sample_perturbation(obj: dict) -> dict:
    perturbation_type = random.choice(obj["Perturbation"])

    if perturbation_type == "rotation":
        degrees = random.randint(MIN_ROTATION, obj["Rotation Limit"]) * random.choice([-1, 1])
        args = [obj["Rotation"], degrees]
    elif perturbation_type == "scale":
        args = [random.choice(SCALE)]
    elif perturbation_type == "placement":
        axes = random.sample(obj["Placement"], k=random.randint(MIN_K_AXIS, MAX_K_AXIS))
        args = []
        for axis in axes:
            args.append(axis)
            args.append(random.choice(TRANSLATION) * random.choice([-1, 1]))
    else:
        args = None

    return {"perturbation_type": perturbation_type, "args": args}


def sample_scene_perturbations(scene_objects: list[dict]) -> dict[str, dict]:
    k = random.randint(MIN_K, min(MAX_K, len(scene_objects)))
    selected = {obj["Object"] for obj in random.sample(scene_objects, k=k)}

    return {
        obj["Object"]: sample_perturbation(obj) if obj["Object"] in selected else {"perturbation_type": "none", "args": None}
        for obj in scene_objects
    }


def create_dataset(
    dataset_dir: Path,
    num_variants: int,
    blend_dir: Path = BLEND_DIR,
    labeled_dir: Path = LABELED_DIR,
    objects_path: Path = OBJECTS_JSON,
    scenes: list[str] | None = None,
) -> None:
    with open(objects_path) as f:
        objects_by_scene = json.load(f)
    with open(labeled_dir / "labels.json") as f:
        legend_by_scene = json.load(f)

    for scene in scenes or sorted(objects_by_scene):
        blend_path = blend_dir / f"{scene}.blend"
        labeled_png = labeled_dir / f"{scene}.png"
        if not blend_path.exists() or not labeled_png.exists():
            print(f"skipping {scene}: missing {blend_path} or {labeled_png}")
            continue

        for variant in range(num_variants):
            entry_dir = dataset_dir / f"{scene}_{variant}"
            entry_dir.mkdir(parents=True, exist_ok=True)
            print(f"Creating {entry_dir.name} ...")

            with open(entry_dir / "labels.json", "w") as f:
                json.dump(legend_by_scene[scene], f, indent=2)

            shutil.copy2(labeled_png, entry_dir / "labeled.png")

            perturbations = sample_scene_perturbations(objects_by_scene[scene])
            with open(entry_dir / "perturbations.json", "w") as f:
                json.dump(perturbations, f, indent=2)

            render_perturbed(blend_path, entry_dir / "perturbed.png", perturbations=json.dumps(perturbations))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Where to write dataset entries (default: %(default)s)")
    parser.add_argument(
        "--n-variants", type=int, default=5, help="Perturbation variants to sample per scene (default: %(default)s)"
    )
    parser.add_argument("--blend-dir", type=Path, default=BLEND_DIR, help="Directory of source .blend scenes (default: %(default)s)")
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=LABELED_DIR,
        help="Directory of labeled reference renders + labels.json (default: %(default)s)",
    )
    parser.add_argument(
        "--objects-path", type=Path, default=OBJECTS_JSON, help="Per-object perturbation metadata (default: %(default)s)"
    )
    parser.add_argument("--scenes", nargs="+", default=None, help="Limit to these scene names (default: every scene in --objects-path)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    create_dataset(
        args.dataset_dir,
        args.n_variants,
        blend_dir=args.blend_dir,
        labeled_dir=args.labeled_dir,
        objects_path=args.objects_path,
        scenes=args.scenes,
    )


if __name__ == "__main__":
    main()
