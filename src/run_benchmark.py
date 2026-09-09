"""Run the perturbation-detection VLM prompt across a dataset built by create_perturbation_dataset.py.

For every entry in <dataset_dir> (a <scene>_<variant>/ directory holding labeled.png,
perturbed.png and labels.json), the two images are compared via
compute_vlm_perturbation_detection(), and the raw structured response is written to
<output_dir>/<model>/<entry>.json.

With --cross-view, each entry's perturbed_view<n>.png (from create_perturbation_dataset.py
--cross-view) is scored the same way against labeled.png too, written to
<output_dir>/<model>/<entry>_view<n>.json.

Both images are downsized to --max-side pixels on the long edge (never upscaled) before being
sent to the VLM; pass a negative value to send them at their rendered resolution instead --
needed by e.g. optimize_rendering.py, where the resolution being tuned would otherwise collapse
back down to the same --max-side cap regardless of what was actually rendered.
"""

import argparse
import json
from pathlib import Path

from src.vlm_score import compute_vlm_perturbation_detection

def run_perturbation_detection(
    model_choice: str,
    dataset_dir: Path,
    output_dir: Path,
    cross_view: bool = False,
    max_side: int | None = 1280,
) -> None:
    model_dir = output_dir / model_choice
    model_dir.mkdir(parents=True, exist_ok=True)

    for entry_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        target_path = entry_dir / "labeled.png"
        labels_path = entry_dir / "labels.json"

        if not (target_path.exists() and labels_path.exists()):
            print(f"skipping {entry_dir.name}: missing labeled.png/labels.json")
            continue
        with open(labels_path) as f:
            legend = json.load(f)

        renders = [(entry_dir.name, entry_dir / "perturbed.png")]
        if cross_view:
            renders += [
                (f"{entry_dir.name}_{view_path.stem.removeprefix('perturbed_')}", view_path)
                for view_path in sorted(entry_dir.glob("perturbed_view*.png"))
            ]

        for response_name, render_path in renders:
            if (model_dir / f"{response_name}.json").exists():
                print("Skipping...")
                continue

            if not render_path.exists():
                print(f"skipping {response_name}: missing {render_path.name}")
                continue

            print(f"Prompting {model_choice} for {response_name} ...")
            try:
                result = compute_vlm_perturbation_detection(
                    target_path=target_path,
                    render_path=render_path,
                    legend=legend,
                    model_choice=model_choice,
                    max_side=max_side,
                )
            except Exception as error:
                print(f"  failed: {error}")
                continue

            with open(model_dir / f"{response_name}.json", "w") as f:
                print("save results")
                json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("dataset_dir", type=Path, help="Dataset entries to evaluate (default: %(default)s)")
    parser.add_argument("output_dir", type=Path, help="Where to write per-entry responses (default: %(default)s)"
    )
    parser.add_argument(
        "--cross-view",
        action="store_true",
        help="Also score each entry's perturbed_view<n>.png against labeled.png",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help=(
            "Downsize images to this many pixels on the long edge before scoring, or pass a "
            "negative number to send them to the VLM unresized (default: %(default)s)"
        ),
    )
    args = parser.parse_args()

    print(args.max_side)

    run_perturbation_detection(
        args.model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        cross_view=args.cross_view,
        max_side=args.max_side if args.max_side > 0 else None,
    )


if __name__ == "__main__":
    main()
