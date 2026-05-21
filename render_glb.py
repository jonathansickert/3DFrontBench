"""Render a GLB scene from a camera position defined in cam.json.

Usage:
    blender --background --python render_glb.py -- <glb_path> <cam_json_path> <output_path> [samples]

Example:
    blender --background --python render_glb.py -- \
        assets/LivingDiningRoom-16352.glb assets/cam.json output.png 128
"""

import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 3:
        print("Usage: blender --background --python render_glb.py -- <glb> <cam.json> <output.png> [samples]")
        sys.exit(1)

    glb_path = Path(argv[0])
    cam_json_path = Path(argv[1])
    output_path = Path(argv[2])
    samples = int(argv[3]) if len(argv) > 3 else 128

    return glb_path, cam_json_path, output_path, samples


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_glb(glb_path: Path):
    bpy.ops.import_scene.gltf(filepath=str(glb_path))


def setup_camera(cam_data: dict) -> bpy.types.Object:
    width = cam_data["resolution"]["width"]
    height = cam_data["resolution"]["height"]
    focal_length_mm = cam_data["intrinsics"]["focal_length_mm"]
    focal_length_px = cam_data["intrinsics"]["focal_length_px"]
    cx = cam_data["intrinsics"]["cx"]
    cy = cam_data["intrinsics"]["cy"]

    # Derive sensor size from intrinsics (36mm full-frame for 50mm / 1143px)
    sensor_width_mm = focal_length_mm * width / focal_length_px
    sensor_height_mm = focal_length_mm * height / focal_length_px

    cam = bpy.data.cameras.new("Camera")
    cam.lens = focal_length_mm
    cam.sensor_width = sensor_width_mm
    cam.sensor_fit = "HORIZONTAL"

    # Principal point offset — shift is in fraction of sensor size, Y flipped
    shift_x = (cx - width / 2.0) / width
    shift_y = -(cy - height / 2.0) / height
    cam.shift_x = shift_x
    cam.shift_y = shift_y

    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # cam_to_world: camera axes in world space (Blender convention — camera looks along -Z)
    c2w = cam_data["extrinsics"]["cam_to_world"]
    mat = mathutils.Matrix([
        [c2w[0][0], c2w[0][1], c2w[0][2], c2w[0][3]],
        [c2w[1][0], c2w[1][1], c2w[1][2], c2w[1][3]],
        [c2w[2][0], c2w[2][1], c2w[2][2], c2w[2][3]],
        [c2w[3][0], c2w[3][1], c2w[3][2], c2w[3][3]],
    ])
    cam_obj.matrix_world = mat

    return cam_obj


def setup_sky_lighting():
    """Add Blender Sky Texture world lighting.

    Uses MULTIPLE_SCATTERING (physically accurate, equivalent to Nishita in Blender < 5).
    Falls back to HOSEK_WILKIE if unavailable.
    """
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

    ntree = world.node_tree
    nodes = ntree.nodes
    links = ntree.links
    nodes.clear()

    sky = nodes.new("ShaderNodeTexSky")

    # Blender 5+: MULTIPLE_SCATTERING is the physically-based sky (formerly Nishita)
    available_types = {item.identifier for item in sky.bl_rna.properties["sky_type"].enum_items}
    if "NISHITA" in available_types:
        sky.sky_type = "NISHITA"
        sky.dust_density = 1.0
    elif "MULTIPLE_SCATTERING" in available_types:
        sky.sky_type = "MULTIPLE_SCATTERING"
        if hasattr(sky, "aerosol_density"):
            sky.aerosol_density = 1.0
    else:
        sky.sky_type = "HOSEK_WILKIE"

    sky.sun_elevation = math.radians(45.0)
    sky.sun_rotation = math.radians(30.0)
    if hasattr(sky, "altitude"):
        sky.altitude = 0.0
    if hasattr(sky, "air_density"):
        sky.air_density = 1.0
    if hasattr(sky, "ozone_density"):
        sky.ozone_density = 1.0

    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = 1.0

    output = nodes.new("ShaderNodeOutputWorld")

    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def configure_render(output_path: Path, width: int, height: int, samples: int):
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True

    # Use GPU if available, fall back to CPU
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("METAL", "CUDA", "OPENCL"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                if any(d.type == device_type for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    scene.cycles.device = "GPU"
                    print(f"Using GPU ({device_type})")
                    break
            except Exception:
                continue
        else:
            scene.cycles.device = "CPU"
            print("Using CPU")
    except Exception as e:
        print(f"GPU setup failed ({e}), using CPU")

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_path.resolve())


def main():
    glb_path, cam_json_path, output_path, samples = parse_args()

    with open(cam_json_path) as f:
        cam_data = json.load(f)

    width = cam_data["resolution"]["width"]
    height = cam_data["resolution"]["height"]

    print(f"Loading scene: {glb_path}")
    clear_scene()
    load_glb(glb_path)

    print("Setting up camera...")
    setup_camera(cam_data)

    print("Setting up sky lighting (Nishita)...")
    setup_sky_lighting()

    print(f"Configuring render: {width}x{height}, {samples} samples...")
    configure_render(output_path, width, height, samples)

    print(f"Rendering to: {output_path}")
    bpy.ops.render.render(write_still=True)
    print("Done.")


main()
