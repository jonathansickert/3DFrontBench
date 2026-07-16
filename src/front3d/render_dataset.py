from pathlib import Path
import argparse

from src.front3d.render_single import render_single

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, help="Source scene directory")
    args = parser.parse_args()

    for scene_dir in args.dataset_dir.iterdir():
        if scene_dir.name.startswith("."):
            continue 
        
        print(f"Rendering scene {scene_dir.name} ...")
        render_single(scene_dir=scene_dir)

if __name__ == "__main__":
    main()
