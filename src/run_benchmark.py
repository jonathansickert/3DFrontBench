"""Run the perturbation-detection VLM prompt across a dataset built by create_perturbation_dataset.py.

For every entry in <dataset_dir> (a <scene>_<variant>/ directory holding labeled.png,
perturbed.png and labels.json), the two images are compared via
compute_vlm_perturbation_detection(), and the raw structured response is written to
<output_dir>/<model>/<entry>.json.
"""

import argparse
import json
from pathlib import Path

from src.create_perturbation_dataset import DATASET_DIR
from src.vlm_score import compute_vlm_perturbation_detection

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"


def run_perturbation_detection(model_choice: str, dataset_dir: Path, output_dir: Path) -> None:
    model_dir = output_dir / model_choice
    model_dir.mkdir(parents=True, exist_ok=True)

    for entry_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        target_path = entry_dir / "labeled.png"
        render_path = entry_dir / "perturbed.png"
        labels_path = entry_dir / "labels.json"
        if not (target_path.exists() and render_path.exists() and labels_path.exists()):
            print(f"skipping {entry_dir.name}: missing labeled.png/perturbed.png/labels.json")
            continue

        with open(labels_path) as f:
            legend = json.load(f)

        print(f"Prompting {model_choice} for {entry_dir.name} ...")
        try:
            result = compute_vlm_perturbation_detection(
                target_path=target_path,
                render_path=render_path,
                legend=legend,
                model_choice=model_choice,
            )
        except Exception as error:
            print(f"  failed: {error}")
            continue

        with open(model_dir / f"{entry_dir.name}.json", "w") as f:
            json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Dataset entries to evaluate (default: %(default)s)")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Where to write per-entry responses (default: %(default)s)"
    )
    args = parser.parse_args()

    run_perturbation_detection(args.model, dataset_dir=args.dataset_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
