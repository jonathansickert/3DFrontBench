"""Re-render an existing perturbation-detection dataset under a fixed set of rendering
parameters, without resampling which objects are perturbed.

For every <source_dataset_dir>/<entry>/ directory, copies labels.json and perturbations.json
unchanged -- the ground truth stays exactly as originally sampled -- and re-renders labeled.png
and perturbed.png, plus any perturbed_view<n>.png the source entry already has (from the same
camera indices), into <dest_dataset_dir>/<entry>/, all under the given rendering parameters.
Entries the source doesn't have cross-view renders for don't get any created here either --
this only reproduces what's already there, styled differently.

Typical use: take the winning trial from an optimize_rendering.py study (tuned against only a
small --n-entries subset) and rebuild the *full* dataset it was tuned against under those exact
settings, e.g. for trial params like
{'material_mode': 'flat_color', 'view_transform': 'AgX', 'shadow_softness': 2.39,
 'gi_bounces': 2, 'exposure': -0.019, 'samples': 27, 'denoising': True, 'resolution': 'hd'}:

    python src/rerender_dataset.py NFINITE100-CROSS NFINITE100-CROSS-tuned \\
        --material-mode flat_color --view-transform AgX --shadow-softness 2.39 \\
        --gi-bounces 2 --exposure -0.019 --samples 27 --denoising true --resolution hd

Writes to a new dest_dataset_dir rather than overwriting the source in place, since the source
is typically still needed as a baseline to compare the re-rendered version against. Both
labeled.png and perturbed.png/perturbed_view*.png always share the same parameters within an
entry, for the same reason optimize_rendering.py does: scoring them under different rendering
styles would let a VLM key off the rendering difference instead of the actual perturbation.
"""

import argparse
import json
import shutil
from pathlib import Path

from src.create_perturbation_dataset import BLEND_DIR
from src.nfinite.render_single_labeled import render_labeled
from src.nfinite.render_single_perturbed import render_perturbed

# What each parameter does when left un-overridden (i.e. not passed on the CLI, so None flows
# through to render_labeled/render_perturbed). material_mode and samples have a real, fixed
# Python-level default; everything else is never touched at all in that case, so whatever the
# source .blend was authored with wins -- there's no single fixed value to report for those.
DEFAULT_DESCRIPTIONS = {
    "material_mode": "full_pbr",
    "view_transform": "unchanged (as authored in the source .blend)",
    "shadow_softness": "unchanged (as authored in the source .blend)",
    "gi_bounces": "unchanged (as authored in the source .blend)",
    "exposure": "unchanged (as authored in the source .blend, i.e. 0 stops)",
    "samples": 32,
    "denoising": "unchanged (as authored in the source .blend)",
    "resolution": "unchanged (native resolution, i.e. 'full_hd')",
}


def _scene_name(entry_name: str) -> str:
    # Entry dirs are named "<scene>_<variant>" by create_perturbation_dataset.py, and scene
    # names themselves can contain underscores -- only the trailing variant index is reliably
    # a plain integer, so split off exactly that.
    scene, _, variant = entry_name.rpartition("_")
    if not scene or not variant.isdigit():
        raise ValueError(f"{entry_name!r} doesn't look like a '<scene>_<variant>' entry directory name")
    return scene


def rerender_entry(source_entry_dir: Path, dest_entry_dir: Path, blend_dir: Path, render_params: dict) -> None:
    blend_path = blend_dir / f"{_scene_name(source_entry_dir.name)}.blend"

    dest_entry_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_entry_dir / "labels.json", dest_entry_dir / "labels.json")
    shutil.copy2(source_entry_dir / "perturbations.json", dest_entry_dir / "perturbations.json")

    with open(source_entry_dir / "perturbations.json") as f:
        perturbations_json = json.dumps(json.load(f))

    render_labeled(blend_path, dest_entry_dir / "labeled.png", labels=True, **render_params)
    render_perturbed(blend_path, dest_entry_dir / "perturbed.png", perturbations=perturbations_json, **render_params)

    for view_path in sorted(source_entry_dir.glob("perturbed_view*.png")):
        camera_position = int(view_path.stem.removeprefix("perturbed_view"))
        render_perturbed(
            blend_path,
            dest_entry_dir / view_path.name,
            perturbations=perturbations_json,
            camera_position=camera_position,
            **render_params,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dataset_dir", type=Path, help="Existing dataset to reuse ground truth from")
    parser.add_argument("dest_dataset_dir", type=Path, help="Where to write the re-rendered dataset")
    parser.add_argument("--blend-dir", type=Path, default=BLEND_DIR, help="default: %(default)s")
    parser.add_argument("--material-mode", default=None, choices=["full_pbr", "flat_color", "clay"])
    parser.add_argument("--view-transform", default=None, choices=["Standard", "Filmic", "AgX"])
    parser.add_argument("--shadow-softness", type=float, default=None, help="Degrees, 0.5-20")
    parser.add_argument("--gi-bounces", type=int, default=None)
    parser.add_argument("--exposure", type=float, default=None, help="Stops (EV)")
    parser.add_argument("--samples", type=int, default=None, help="Cycles sample count")
    parser.add_argument(
        "--denoising", type=lambda v: v.lower() in ("1", "true", "yes", "on"), default=None,
    )
    parser.add_argument("--resolution", default=None, choices=["512", "hd", "full_hd", "4k"])
    args = parser.parse_args()

    requested = {
        "material_mode": args.material_mode,
        "view_transform": args.view_transform,
        "shadow_softness": args.shadow_softness,
        "gi_bounces": args.gi_bounces,
        "exposure": args.exposure,
        "samples": args.samples,
        "denoising": args.denoising,
        "resolution": args.resolution,
    }
    render_params = {k: v for k, v in requested.items() if v is not None}
    if not render_params:
        raise SystemExit("Pass at least one rendering parameter, or there's nothing to re-render differently.")

    entry_dirs = sorted(p for p in args.source_dataset_dir.iterdir() if p.is_dir())
    if not entry_dirs:
        raise SystemExit(f"No dataset entries found in {args.source_dataset_dir}")

    print(f"Re-rendering {len(entry_dirs)} entries from {args.source_dataset_dir} into {args.dest_dataset_dir}\n")
    print("Rendering parameters (* = default, not overridden on the CLI):")
    for name, default in DEFAULT_DESCRIPTIONS.items():
        value = requested[name]
        print(f"  {name:16s} {value}" if value is not None else f"  {name:16s} {default} *")
    print()

    for entry_dir in entry_dirs:
        print(f"Re-rendering {entry_dir.name} ...")
        rerender_entry(entry_dir, args.dest_dataset_dir / entry_dir.name, args.blend_dir, render_params)


if __name__ == "__main__":
    main()
