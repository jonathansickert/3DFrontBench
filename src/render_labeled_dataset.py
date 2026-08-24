import json
from pathlib import Path
import argparse

from src.nfinite.render_single_labeled import OBJECTS_JSON, render_labeled


def render_dataset(dataset_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(OBJECTS_JSON) as f:
        objects_by_scene = json.load(f)
    legend = {}

    for blend_path in dataset_dir.glob("*.blend"):
        print(f"Rendering scene {blend_path.stem} ...")

        out_path = output_dir / f"{blend_path.stem}.png"
        render_labeled(blend_path, out_path, labels=True)

        scene = blend_path.stem
        scene_labels = [entry["Object"] for entry in objects_by_scene[scene]]
        legend[scene] = [{label: index} for index, label in enumerate(scene_labels, start=1)]

    with open(output_dir / "labels.json", "w") as f:
        json.dump(legend, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Source scene directory")
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    render_dataset(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
