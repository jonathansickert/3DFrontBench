"""Optimize NFINITE rendering parameters for same-view perturbation-detection score, via Optuna.

Fixes a small, deterministic subset of entries from an existing dataset (built by
create_perturbation_dataset.py) -- reusing their already-sampled perturbations.json and
labels.json so only rendering *style* varies between trials, never which objects are perturbed
or how the digit legend is assigned (both are derived purely from objects.json's scene order,
independent of any rendering parameter -- see _build_label_entries in render_nfinite_labeled.py).

For each Optuna trial:

1. sample a candidate rendering-parameter set (material_mode, view_transform, shadow_softness,
   gi_bounces, exposure, samples, denoising, resolution)
2. re-render labeled.png and perturbed.png for every fixed entry with those parameters -- both
   images always share the same parameters, since scoring them under different rendering styles
   would let the VLM key off a rendering difference instead of the actual perturbation
3. run the perturbation-detection VLM prompt against them (run_benchmark.py's
   run_perturbation_detection)
4. score the responses (evaluate_models.py's macro_metrics) and report
   same-view macro_f1 back to Optuna as the value to maximize

This is a first draft, not a tuned production loop: every trial costs len(entries) x 2 Blender
renders plus len(entries) VLM calls, and both samples and resolution directly trade off against
that per-trial render cost (samples is capped at 32 -- the value every other render in this repo
already uses by default -- so the study can only trade quality *down* for speed, never render
slower than the existing pipeline). Start --n-entries and --n-trials small (a handful of
entries, a couple dozen trials) until the loop and its cost are trusted, before scaling up.
Camera position is held fixed -- it's a framing knob, not a rendering-style one -- and this
only ever scores the same-view case; cross-view scoring is out of scope here.

run_benchmark.py's own scoring step downsizes both images to 1280px on the long edge by default
(run_benchmark.py's --max-side), which would otherwise flatten "hd"/"full_hd" down to the exact
same 1280x720 image regardless of what was rendered -- verified against this repo's scenes
(native 1920x1080): only "512" would ever end up visibly different. The run_perturbation_detection
call below passes max_side=None specifically to disable that, so the resolution this script picks
is the resolution the VLM is actually scored on -- which is also why "4k" (~3840x2160) isn't in
the search space at all: sent unresized, it's expensive and some VLM backends reject or choke on
an image that large, which surfaced as whole trials failing outright rather than just scoring low.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import optuna

from src.create_perturbation_dataset import BLEND_DIR
from src.evaluate_models import load_predictions, macro_metrics
from src.nfinite.render_single_labeled import render_labeled
from src.nfinite.render_single_perturbed import render_perturbed
from src.run_benchmark import run_perturbation_detection


def _scene_name(entry_name: str) -> str:
    # Entry dirs are named "<scene>_<variant>" by create_perturbation_dataset.py, and scene
    # names themselves can contain underscores -- only the trailing variant index is reliably
    # a plain integer, so split off exactly that.
    scene, _, variant = entry_name.rpartition("_")
    if not scene or not variant.isdigit():
        raise ValueError(f"{entry_name!r} doesn't look like a '<scene>_<variant>' entry directory name")
    return scene


def suggest_render_params(trial: optuna.Trial) -> dict:
    return {
        "material_mode": trial.suggest_categorical("material_mode", ["full_pbr", "flat_color", "clay"]),
        # Tonemapping curve, not a light-transport change -- "Standard" clips highlights hard
        # (e.g. a blown-out window), "Filmic"/"AgX" both compress them into a soft rolloff
        # instead, AgX being Blender's newer, more saturated default since 4.0.
        "view_transform": trial.suggest_categorical("view_transform", ["Standard", "Filmic", "AgX"]),
        # Degrees, not radians -- apply_shadow_softness converts. 0.5 matches the real sun's
        # angular size (sharp, crisp-edged shadows); 20 spreads it into a soft, area-light-like
        # source with wide, feathered penumbras.
        "shadow_softness": trial.suggest_float("shadow_softness", 0.5, 20.0),
        "gi_bounces": trial.suggest_int("gi_bounces", 0, 12),
        "exposure": trial.suggest_float("exposure", -2.0, 2.0),
        # 32 is the ceiling, not a starting point: every other render in this repo already
        # defaults to 32 samples, so this only ever asks "can we get away with less noise
        # tolerance for less render time", never "is more than the existing default better".
        "samples": trial.suggest_int("samples", 4, 32),
        # Off trades detection fidelity for render speed at low sample counts -- worth knowing
        # whether the denoiser's smoothing is helping or hurting a VLM's ability to spot a
        # perturbation, not just whether it looks cleaner to a human.
        "denoising": trial.suggest_categorical("denoising", [True, False]),
        # Genuinely reaches the VLM at this size -- run_perturbation_detection is called below
        # with max_side=None specifically so this isn't silently capped back down to 1280px
        # regardless of what's picked here (see module docstring). "4k" is deliberately excluded:
        # sent unresized (~3840x2160), it's both expensive and prone to being rejected outright
        # by some VLM backends, which showed up as whole trials failing rather than scoring low.
        "resolution": trial.suggest_categorical("resolution", ["512", "hd", "full_hd"]),
    }


def render_trial_entry(source_entry_dir: Path, trial_entry_dir: Path, blend_dir: Path, render_params: dict) -> None:
    blend_path = blend_dir / f"{_scene_name(source_entry_dir.name)}.blend"

    trial_entry_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_entry_dir / "labels.json", trial_entry_dir / "labels.json")
    shutil.copy2(source_entry_dir / "perturbations.json", trial_entry_dir / "perturbations.json")

    with open(source_entry_dir / "perturbations.json") as f:
        perturbations = json.load(f)

    render_labeled(
        blend_path,
        trial_entry_dir / "labeled.png",
        labels=True,
        **render_params,
    )
    render_perturbed(
        blend_path,
        trial_entry_dir / "perturbed.png",
        perturbations=json.dumps(perturbations),
        **render_params,
    )


def make_objective(
    entry_names: list[str],
    source_dataset_dir: Path,
    blend_dir: Path,
    work_dir: Path,
    model_choice: str,
    keep_renders: bool,
):
    def objective(trial: optuna.Trial) -> float:
        render_params = suggest_render_params(trial)

        trial_dir = work_dir / f"trial_{trial.number:04d}"
        dataset_dir = trial_dir / "dataset"
        output_dir = trial_dir / "responses"

        for entry_name in entry_names:
            render_trial_entry(
                source_entry_dir=source_dataset_dir / entry_name,
                trial_entry_dir=dataset_dir / entry_name,
                blend_dir=blend_dir,
                render_params=render_params,
            )

        # max_side=None: run_benchmark.py's default resize caps everything at 1280px on the long
        # edge, which would flatten "hd"/"full_hd" down to the exact same image and make the
        # "resolution" search space above pointless above "512" -- see optimize_rendering.py's
        # module docstring. Disabling it here means the resolution picked above genuinely reaches
        # the VLM; that's the whole point of tuning it at all.
        run_perturbation_detection(model_choice, dataset_dir=dataset_dir, output_dir=output_dir, max_side=None)

        predictions_df = load_predictions(output_dir, dataset_dir)
        if predictions_df.empty:
            # Every VLM call failed or produced nothing scorable for this parameter set --
            # worst possible score rather than crashing the whole study over one bad trial.
            print(f"trial {trial.number}: no scorable predictions, scoring 0.0")
            macro_f1 = 0.0
        else:
            # load_predictions above is called without include_cross_view, so every row here is
            # "same"-view -- macro_metrics always indexes by (model, view), never just model.
            macro_f1 = float(macro_metrics(predictions_df).loc[(model_choice, "same"), "macro_f1"])

        if not keep_renders:
            shutil.rmtree(dataset_dir, ignore_errors=True)

        return macro_f1

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Existing dataset (from create_perturbation_dataset.py) to draw fixed entries/ground truth from",
    )
    parser.add_argument("work_dir", type=Path, help="Scratch directory for per-trial renders and VLM responses")
    parser.add_argument("--blend-dir", type=Path, default=BLEND_DIR, help="default: %(default)s")
    parser.add_argument("--model", default="qwen32_instruct", help="VLM model choice to score with (default: %(default)s)")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--n-entries", type=int, default=20, help="How many dataset entries to fix and re-render per trial"
    )
    parser.add_argument("--seed", type=int, default=0, help="Which dataset entries get fixed for this study")
    parser.add_argument(
        "--keep-renders",
        action="store_true",
        help="Don't delete each trial's labeled.png/perturbed.png after scoring (disk-heavy over many trials)",
    )
    parser.add_argument(
        "--study-storage",
        default=None,
        help="Optuna storage URL (e.g. sqlite:///study.db) to persist/resume the study; default is in-memory only",
    )
    parser.add_argument("--study-name", default="nfinite_render_params")
    args = parser.parse_args()

    entry_dirs = sorted(p for p in args.dataset_dir.iterdir() if p.is_dir())
    if not entry_dirs:
        raise SystemExit(f"No dataset entries found in {args.dataset_dir}")

    entry_names = [p.name for p in random.Random(args.seed).sample(entry_dirs, k=min(args.n_entries, len(entry_dirs)))]
    print(f"Fixed {len(entry_names)} entries for this study: {entry_names}")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.study_storage,
        load_if_exists=args.study_storage is not None,
    )
    objective = make_objective(
        entry_names,
        source_dataset_dir=args.dataset_dir,
        blend_dir=args.blend_dir,
        work_dir=args.work_dir,
        model_choice=args.model,
        keep_renders=args.keep_renders,
    )
    # catch=(Exception,): a render/VLM pipeline this long will occasionally fail a trial for
    # reasons that have nothing to do with the sampled params (a flaky API call, a transient
    # Blender error) -- without this, Optuna's default behavior is to re-raise and stop the
    # *entire* study on the first such failure, discarding every trial done so far unless
    # --study-storage was set. This logs the trial as FAIL and moves on instead.
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    study.trials_dataframe().to_csv(args.work_dir / "trials.csv", index=False)

    print("\nBest trial:")
    print(f"  same-view macro_f1: {study.best_value:.3f}")
    print(f"  params: {json.dumps(study.best_params, indent=2)}")

    with open(args.work_dir / "best_params.json", "w") as f:
        json.dump({"value": study.best_value, "params": study.best_params}, f, indent=2)


if __name__ == "__main__":
    main()
