import sys
import json
import bpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.blender.blender_helper import add_lights_for_light_meshes, clear_scene, add_camera, enable_sky_texture

def parse_scene_dir() -> Path:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Usage: blender --background --python export_scaffold_blend.py -- <scene_dir>")
    scene_dir = Path(argv[argv.index("--") + 1])
    if not scene_dir.is_dir():
        raise SystemExit(f"scene_dir does not exist: {scene_dir}")
    return scene_dir


scene_dir = parse_scene_dir()

scaffold_glb = scene_dir / "scene_scaffold.glb"
camera_json = scene_dir / "camera.json"
output_blend = scene_dir / "scene_scaffold.blend"

with open(camera_json) as f:
    cam_dict = json.load(f)

clear_scene()
bpy.ops.import_scene.gltf(filepath=str(scaffold_glb))
enable_sky_texture()
add_lights_for_light_meshes()
add_camera(cam_dict=cam_dict)

bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
print(f"Saved: {output_blend}")


