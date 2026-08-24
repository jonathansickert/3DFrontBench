from pathlib import Path
import json
import argparse

from src.nfinite.render_single_perturbed import render_perturbed


def render_dataset(dataset_dir: Path, output_dir: Path, perturbations_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(perturbations_path) as f:
        perturbations_by_scene = json.load(f)

    for blend_path in dataset_dir.glob("*.blend"):
        print(f"Rendering scene {blend_path.stem} ...")

        out_path = output_dir / f"{blend_path.stem}.png"
        scene_perturbations = perturbations_by_scene.get(blend_path.stem, {})
        render_perturbed(blend_path, out_path, perturbations=json.dumps(scene_perturbations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("perturbations_path", type=Path)

    args = parser.parse_args()

    render_dataset(args.dataset_dir, args.output_dir, perturbations_path=args.perturbations_path)


if __name__ == "__main__":
    main()
