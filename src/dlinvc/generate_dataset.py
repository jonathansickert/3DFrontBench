from dlinvc.json_to_3d_front_scene import build_room_scene
from pathlib import Path
import json


def generate_dataset(samples: int = 30):
    samples_found = 0
    for scene_json_path in Path("3D-FRONT/3D-FRONT").glob("*.json"):
        if samples_found == samples:
            break

        with open(scene_json_path, "r") as f:
            scene_json = json.load(f)

        scene_id = scene_json["uid"]

        for room in scene_json["scene"]["room"]:
            room_id = room["instanceid"]

            res = build_room_scene(scene_json, room, bounding_box=False)
            if res is None:
                continue

            scene, size, furniture_list, is_valid = res
            if is_valid and size >= 10:
                scene_bbox, *_ = build_room_scene(scene_json, room, bounding_box=True)

                scene.export(f"dataset/{scene_id}_{room_id}.glb")
                scene_bbox.export(f"dataset/{scene_id}_{room_id}_bbox.glb")

                metadata = {
                    "scene_id": scene_id,
                    "room_id": room_id,
                    "scene_name": f"{scene_id}_{room_id}.glb",
                    "scene_name_bbox": f"{scene_id}_{room_id}_bbox.glb",
                    "furniture": furniture_list,
                }

                with open(f"dataset/{scene_id}_{room_id}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                samples_found += 1
                print(f"Found: {scene_id}_{room_id} with {size} objects")
                if samples_found == samples:
                    return


if __name__ == "__main__":
    generate_dataset()
