"""
Generate a 3D-FRONT dataset by processing scene data and exporting room geometries.

This script processes 3D-FRONT JSON scene files to extract individual rooms and generate
a structured dataset. For each valid room, it:
- Builds a 3D scene representation and corresponding bounding box
- Exports both as GLB (GL Transmission Format) files
- Creates metadata JSON files containing scene information and furniture lists

The script filters rooms based on size (minimum 10 objects) and validity checks to ensure
high-quality dataset samples. Output files are organized in the 'dataset/' directory with
consistent naming conventions: scene_id_room_id.glb, scene_id_room_id_bbox.glb, and
corresponding metadata JSON files.
"""

from dlinvc.util import render_trimesh_scene, get_pyrender_cam, remove_textures
from dlinvc.dataset.json_to_3d_front_scene import build_room_scene
from pathlib import Path
import json
from PIL import Image


def legacy_generate_dataset(samples: int = 30):
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


SELECTED_SCENES = [
    {"scene_id": "09d742d0-9e99-4e31-ac3d-ad1879cf691b", "room_id": "LivingDiningRoom-9326"},
    {"scene_id": "2eeb434c-855e-4298-a8d5-29b1ca838608", "room_id": "MasterBedroom-5966"},
    {"scene_id": "e44c4ecc-0260-469d-a60d-b71cf1d34029", "room_id": "LivingRoom-15009"},
    {"scene_id": "1ad74b11-777a-4610-a249-7c5e044a0f01", "room_id": "LivingDiningRoom-10290"},
    {"scene_id": "1ad74b11-777a-4610-a249-7c5e044a0f01", "room_id": "MasterBedroom-14403"},
    {"scene_id": "6e7a0c41-15f3-4909-a20c-dab460b45ada", "room_id": "LivingDiningRoom-847"},
    {"scene_id": "ab6d1b3b-31ba-4191-bc17-d03e504484a8", "room_id": "LivingDiningRoom-209062"},
    {"scene_id": "473fb8fd-a6d4-4fc1-bd5d-1afd61a9617e", "room_id": "LivingRoom-9915"},
    {"scene_id": "f006ad67-40d2-4068-950c-2ce3805d3932", "room_id": "LivingDiningRoom-38341"},
    {"scene_id": "109a8e42-f7d7-4893-8262-f2030072b760", "room_id": "LivingRoom-38165"},
    {"scene_id": "0f661df2-0f41-47a4-830c-7444f4a33a03", "room_id": "LivingDiningRoom-12554"},
    {"scene_id": "109c84cd-7194-4b1d-ac52-8d1862565c6c", "room_id": "Corridor-20684"},
    {"scene_id": "9f283b51-0327-43d5-9ec0-44d867530f4e", "room_id": "Library-35224"},
    {"scene_id": "a17c6eb5-a044-400a-b0b6-3e757cccbb15", "room_id": "LivingDiningRoom-15943"},
    {"scene_id": "10e2d93c-2a82-49f2-a95c-7598d24901f2", "room_id": "LivingDiningRoom-8149"},
    {"scene_id": "2e422478-2de0-4d16-b398-657b07c1cac0", "room_id": "LivingDiningRoom-4293"},
    {"scene_id": "aa9d393b-30df-456f-96dc-dad7bb8554c9", "room_id": "LivingRoom-38326"},
    {"scene_id": "996d5631-517a-4906-b232-5e1148c97867", "room_id": "LivingDiningRoom-8344"},
    {"scene_id": "f36889f8-d94b-4ab9-9da1-46e06a27ff90", "room_id": "LivingRoom-12836"},
    {"scene_id": "a0b8b195-676d-47bd-afce-3e135b8fde83", "room_id": "LivingDiningRoom-22853"},
    {"scene_id": "6ae193c4-4fe1-42e8-b2dd-38ded31b7f12", "room_id": "LivingDiningRoom-48379"},
    {"scene_id": "7c2e62e7-ab51-4900-bfdc-1932415fb295", "room_id": "LivingDiningRoom-16509"},
    {"scene_id": "103cce55-24d5-4c71-9856-156962e30511", "room_id": "LivingDiningRoom-89516"},
    {"scene_id": "d650faee-f134-46fd-b8f7-38712998f3b5", "room_id": "LivingRoom-2922"},
    {"scene_id": "9f283b51-0327-43d5-9ec0-44d867530f4e", "room_id": "Hallway-34943"},
    {"scene_id": "1f077e24-2eca-43ca-bc92-1b36eab99467", "room_id": "LivingDiningRoom-180855"},
    {"scene_id": "5e6d0804-54a9-4d4a-b4ed-1b4f22aa1d01", "room_id": "LivingRoom-16933"},
    {"scene_id": "7686a060-ab0d-4014-9e5c-75d75e0752e3", "room_id": "LivingDiningRoom-44815"},
    {"scene_id": "8174e94b-cb97-4d24-bd3a-81a095192bbe", "room_id": "LivingDiningRoom-33640"},
    {"scene_id": "b34c215e-1449-4702-8080-e7173e8e8f67", "room_id": "LivingDiningRoom-14069"},
]


FRONT_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FRONT")
DATA_PATH = Path("/home/jonathansickert/git/DLinVC/dataset")
CAM_PARAMS_PATH = Path("/home/jonathansickert/git/DLinVC/dataset_cam_params")


def generate_dataset():
    for scene in SELECTED_SCENES:
        scene_id = scene["scene_id"]
        room_id = scene["room_id"]

        name = f"{scene_id}_{room_id}"
        scene_target_dir = DATA_PATH / name
        scene_target_dir.mkdir(parents=True, exist_ok=True)

        scene_json_path = FRONT_PATH / f"{scene_id}.json"
        with open(scene_json_path) as f:
            scene_json = json.load(f)

        room_json = None
        for room in scene_json["scene"]["room"]:
            if room["instanceid"] == room_id:
                room_json = room
                break

        assert room_json is not None

        scene_mesh, scaffold_mesh, furniture_list, is_valid = build_room_scene(
            scene_json, room_json, bounding_box=False
        )
        scene_bbox_mesh, _, _, _ = build_room_scene(scene_json, room_json, bounding_box=True)

        assert len(furniture_list) >= 10
        assert is_valid

        metadata = {
            "scene_id": scene_id,
            "room_id": room_id,
            "furniture": furniture_list,
        }

        with open(CAM_PARAMS_PATH / f"{name}_camera.json") as f:
            cam_params = json.load(f)

        with open(scene_target_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(scene_target_dir / "camera.json", "w") as f:
            json.dump(cam_params, f, indent=2)

        scene_mesh.export(scene_target_dir / "scene.glb")
        scaffold_mesh.export(scene_target_dir / "scene_scaffold.glb")
        scene_bbox_mesh.export(scene_target_dir / "scene_bbox.glb")

        remove_textures(scene_mesh)
        remove_textures(scaffold_mesh)
        remove_textures(scene_bbox_mesh)

        cam, c2w, width, height = get_pyrender_cam(cam_params)
        color, depth = render_trimesh_scene(scene_mesh, cam, c2w, width, height)
        color_scaffold, depth_scaffold = render_trimesh_scene(scaffold_mesh, cam, c2w, width, height)
        color_bbox, depth_bbox = render_trimesh_scene(scene_bbox_mesh, cam, c2w, width, height)

        Image.fromarray(color).save(scene_target_dir / "color.png")
        Image.fromarray(depth).save(scene_target_dir / "depth.png")
        Image.fromarray(color_scaffold).save(scene_target_dir / "color_scaffold.png")
        Image.fromarray(depth_scaffold).save(scene_target_dir / "depth_scaffold.png")
        Image.fromarray(color_bbox).save(scene_target_dir / "color_bbox.png")
        Image.fromarray(depth_bbox).save(scene_target_dir / "depth_bbox.png")


if __name__ == "__main__":
    generate_dataset()
