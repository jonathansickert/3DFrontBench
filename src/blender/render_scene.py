import bpy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.blender.blender_helper import (
    add_lights_for_light_meshes,
    clear_scene,
    add_camera,
    enable_sky_texture
)


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender --background --python render_3d_front_images.py -- <scene_glb> <camera_json> <output_png>"
        )
    args = argv[argv.index("--") + 1 :]
    if len(args) < 3:
        raise SystemExit("Expected three arguments: scene_glb camera_json output_png")
    return args[0], args[1], args[2]


scene_path, camera_path, output_path = _parse_args()

with open(camera_path) as file:
    cam_dict = json.load(file)

clear_scene()
bpy.ops.import_scene.gltf(filepath=scene_path)
add_lights_for_light_meshes()
enable_sky_texture()
add_camera(cam_dict=cam_dict)

# Render Scene
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.use_nodes = False

cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
cycles_prefs.compute_device_type = "CUDA"
cycles_prefs.get_devices()
for device in cycles_prefs.devices:
    device.use = device.type == "CUDA"
bpy.context.scene.cycles.device = "GPU"
bpy.context.scene.cycles.samples = 32
bpy.context.scene.render.filepath = output_path
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
