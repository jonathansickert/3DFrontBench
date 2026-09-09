"""Build a randomly-sampled perturbation-detection dataset.

For every scene in --objects-path, samples --num-variants independent perturbation specs (one
draw per object: rotation/scale/placement/count, or "none") and materializes each as a
self-contained entry directory:

<dataset_dir>/<scene>_<variant>/
    labels.json         -- digit legend for this scene, copied from <labeled_dir>/labels.json
    labeled.png         -- unperturbed reference render, copied from <labeled_dir>/
    perturbations.json  -- {object_label: {"perturbation_type": ..., "args": ...}}
    perturbed.png       -- rendered from <blend_dir>/<scene>.blend with that spec applied

With --cross-view, each entry additionally gets one perturbed_view<camera_index>.png per
alternative camera position listed for that scene in assets/camera_positions.json (none if the
scene has no entry there) -- the same perturbation spec re-rendered from each alternate camera,
alongside the original perturbed.png from the .blend file's authored camera. labeled.png is
unaffected either way: it's always the single reference copied from <labeled_dir>/.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from src.nfinite.render_single_labeled import CAMERA_POSITIONS_JSON, OBJECTS_JSON
from src.nfinite.render_single_perturbed import RENDER_PASSES, render_perturbed

REPO_ROOT = Path(__file__).resolve().parents[1]
BLEND_DIR = REPO_ROOT / "NFINITE"
LABELED_DIR = REPO_ROOT / "NFINITE_labeled"

MIN_ROTATION = 30

SCALE = [0.67, 0.8, 1.25, 1.5]
TRANSLATION = [0.67, 0.8, 1.25, 1.5]

TYPES = ["count", "scale", "rotation", "placement"]

# How many objects get perturbed per image. Includes 0 so that
# fully-unchanged pairs exist.
K_WEIGHTS = {0: 0.10, 1: 0.25, 2: 0.35, 3: 0.20, 4: 0.10}


# --- samplers ------------------------------------------------------------

def sample_rotation(obj: dict) -> dict:
    degrees = random.randint(MIN_ROTATION, obj["Rotation Limit"]) * random.choice([-1, 1])
    args = [obj["Rotation"], degrees]
    return {"perturbation_type" : "rotation", "args" : args}

def sample_count(obj: dict) -> dict:
    return {"perturbation_type" : "count", "args" : None}

def parse_placement_axis(spec: str) -> tuple[str, float]:
    # objects.json marks each axis an object may slide along with a "+"/"-"
    # prefix once only one direction keeps it visible to this scene's camera
    # (e.g. sliding a wall-mounted mirror "+x" tucks it out of frame behind a
    # doorway, but "-x" doesn't) -- verified by hand per scene, since that's
    # scene geometry no automated check here reliably got right. An unsigned
    # axis ("x") means both directions were checked and are fine.
    if spec[0] in "+-":
        return spec[1:], (1.0 if spec[0] == "+" else -1.0)
    return spec, random.choice([-1.0, 1.0])

def sample_placement(obj: dict) -> dict:
    axis, sign = parse_placement_axis(random.choice(obj["Placement"]))
    signed_factor = sign * random.choice(TRANSLATION)
    return {"perturbation_type" : "placement", "args" : [axis, signed_factor]}

def sample_scale(obj: dict) -> dict:
    return {"perturbation_type" : "scale", "args" : [random.choice(SCALE)]}

SAMPLERS = {
    "count"     : sample_count,
    "scale"     : sample_scale,
    "rotation"  : sample_rotation,
    "placement" : sample_placement,
}


# --- type choice: quota-driven instead of per-object uniform -----------

def sample_perturbation(obj: dict, quota: dict) -> dict:
    """Pick the legal type that is furthest behind its quota.

    random.choice(obj["Perturbation"]) skews the marginals: rows that
    forbid rotation drag rotation's share down and nothing compensates.
    Objects that *do* allow rotation now absorb that shortfall.
    """
    legal = [t for t in obj["Perturbation"] if t in quota]
    if not legal:
        return None

    best = max(quota[t] for t in legal)
    perturbation_type = random.choice([t for t in legal if quota[t] == best])

    quota[perturbation_type] -= 1
    return SAMPLERS[perturbation_type](obj)


def sample_image(objects: list, quota: dict) -> list:
    """One image: draw k, then perturb k distinct objects."""
    k = random.choices(list(K_WEIGHTS), weights=list(K_WEIGHTS.values()))[0]
    k = min(k, len(objects))

    out = []
    for obj in random.sample(objects, k=k):
        pert = sample_perturbation(obj, quota)
        if pert is not None:
            pert["object_label"] = obj["Object"]
            out.append(pert)
    return out


def build_manifest(objects_by_scene: dict, n_images_per_scene: int) -> list:
    n_images = n_images_per_scene * len(objects_by_scene)
    n_pert = int(sum(k * w for k, w in K_WEIGHTS.items()) * n_images)
    quota = {t: n_pert // len(TYPES) for t in TYPES}

    manifest = []
    for i in range(n_images_per_scene):
        for scene, objects in objects_by_scene.items():
            manifest.append({
                "image_id"      : f"{scene}__{i:03d}",
                "scene"         : scene,
                "perturbations" : sample_image(objects, quota),
            })
    return manifest

def create_dataset(
    dataset_dir: Path,
    num_variants: int,
    blend_dir: Path = BLEND_DIR,
    labeled_dir: Path = LABELED_DIR,
    objects_path: Path = OBJECTS_JSON,
    render_pass: str | None = None,
    cross_view: bool = False,
) -> None:
    with open(objects_path) as f:
        objects_by_scene = json.load(f)
    with open(labeled_dir / "labels.json") as f:
        legend_by_scene = json.load(f)

    camera_positions_by_scene = {}
    if cross_view:
        with open(CAMERA_POSITIONS_JSON) as f:
            camera_positions_by_scene = json.load(f)

    scenes = {}
    for scene, scene_objects in objects_by_scene.items():
        blend_path = blend_dir / f"{scene}.blend"
        labeled_png = labeled_dir / f"{scene}.png"
        if not blend_path.exists() or not labeled_png.exists():
            print(f"skipping {scene}: missing {blend_path} or {labeled_png}")
            continue
        scenes[scene] = scene_objects

    variant_by_scene = {scene: 0 for scene in scenes}
    for manifest_entry in build_manifest(scenes, num_variants):
        scene = manifest_entry["scene"]
        variant = variant_by_scene[scene]
        variant_by_scene[scene] += 1

        entry_dir = dataset_dir / f"{scene}_{variant}"
        entry_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating {entry_dir.name} ...")

        with open(entry_dir / "labels.json", "w") as f:
            json.dump(legend_by_scene[scene], f, indent=2)

        shutil.copy2(labeled_dir / f"{scene}.png", entry_dir / "labeled.png")

        perturbations = {obj["Object"]: {"perturbation_type": "none", "args": None} for obj in scenes[scene]}
        for pert in manifest_entry["perturbations"]:
            perturbations[pert.pop("object_label")] = pert
        with open(entry_dir / "perturbations.json", "w") as f:
            json.dump(perturbations, f, indent=2)

        perturbations_json = json.dumps(perturbations)
        render_perturbed(
            blend_dir / f"{scene}.blend",
            entry_dir / "perturbed.png",
            perturbations=perturbations_json,
            render_pass=render_pass,
        )

        for camera_position in range(len(camera_positions_by_scene.get(scene, []))):
            render_perturbed(
                blend_dir / f"{scene}.blend",
                entry_dir / f"perturbed_view{camera_position}.png",
                perturbations=perturbations_json,
                render_pass=render_pass,
                camera_position=camera_position,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--n_variants", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--labeled-dir", type=Path, default=LABELED_DIR, help="default: %(default)s")
    parser.add_argument(
        "--render-pass",
        choices=RENDER_PASSES,
        default=None,
        help="Lighting-independent render pass/visualization mode for perturbed.png (default: beauty)",
    )
    parser.add_argument(
        "--cross-view",
        action="store_true",
        help=(
            "Also render perturbed_view<n>.png from each alternative camera position listed for "
            f"the scene in {CAMERA_POSITIONS_JSON}, alongside the default perturbed.png "
            "(labeled.png is unaffected)"
        ),
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    create_dataset(
        args.dataset_dir,
        args.n_variants,
        labeled_dir=args.labeled_dir,
        render_pass=args.render_pass,
        cross_view=args.cross_view,
    )


if __name__ == "__main__":
    main()
