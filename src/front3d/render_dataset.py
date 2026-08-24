from pathlib import Path
import argparse

from src.front3d.render_single import MATERIAL_MODES, render_single


def render_dataset(dataset_dir: Path, material_mode: str = "full_pbr"):
    for scene_dir in dataset_dir.iterdir():
        if scene_dir.name.startswith("."):
            continue

        print(f"Rendering scene {scene_dir.name} ...")
        render_single(scene_dir=scene_dir, material_mode=material_mode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Source scene directory")
    parser.add_argument(
        "--material-mode",
        choices=MATERIAL_MODES,
        default="full_pbr",
        help="Material/lighting quality mode",
    )
    args = parser.parse_args()

    render_dataset(args.dataset_dir, material_mode=args.material_mode)


if __name__ == "__main__":
    main()
