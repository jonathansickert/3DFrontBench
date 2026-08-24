import colorsys

import bpy
import mathutils

MATERIAL_MODES = ("full_pbr", "texture_flat_lighting", "no_texture_flat_lighting")


def _replace_link(links, to_socket, from_socket):
    for existing in list(to_socket.links):
        links.remove(existing)
    links.new(from_socket, to_socket)


def _flat_color_for_index(index: int, total: int) -> tuple[float, float, float, float]:
    """A flat color distinguishable from its neighbors, spaced evenly around the hue wheel."""
    hue = (index / total) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.9)
    return (r, g, b, 1.0)


def apply_material_mode(mode: str):
    """Rewire every material's shader output to match a rendering-quality mode.

    - "full_pbr": leave materials untouched (Principled BSDF, textures, full Cycles lighting).
    - "texture_flat_lighting": keep each material's texture/base color but emit it directly
      via an Emission shader, bypassing Cycles lighting, shadows, and global illumination.
    - "no_texture_flat_lighting": same as above, but each material emits a flat color of its
      own (evenly spaced around the hue wheel) instead of its texture/base color, so objects
      stay distinguishable from each other without carrying any texture or real-color detail.
    """
    if mode == "full_pbr":
        return
    if mode not in MATERIAL_MODES:
        raise ValueError(f"Unknown material_mode: {mode!r}, expected one of {MATERIAL_MODES}")

    materials = bpy.data.materials

    for index, mat in enumerate(materials):
        if mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if bsdf is None or output is None:
            continue

        emission = nodes.new(type="ShaderNodeEmission")

        if mode == "no_texture_flat_lighting":
            emission.inputs["Color"].default_value = _flat_color_for_index(index, len(materials))
        else:
            base_color_input = bsdf.inputs["Base Color"]
            source_link = next((l for l in base_color_input.links), None)
            if source_link is not None:
                links.new(source_link.from_socket, emission.inputs["Color"])
            else:
                emission.inputs["Color"].default_value = base_color_input.default_value

        _replace_link(links, output.inputs["Surface"], emission.outputs["Emission"])


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def add_lights_for_light_meshes():
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH" or ("light" not in obj.name.lower() and "lamp" not in obj.name.lower()):
            continue

        world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if not world_verts:
            continue
        centroid = sum(world_verts, mathutils.Vector()) / len(world_verts)
        light_data = bpy.data.lights.new(name=f"{obj.name}_point", type="POINT")
        light_data.energy = 300
        light_data.shadow_soft_size = 0.25
        light_data.color = (1.0, 0.95, 0.8)  # warm white
        light_obj = bpy.data.objects.new(name=f"{obj.name}_point", object_data=light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = centroid


def clear_cameras():
    for obj in list(bpy.context.scene.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)


def add_camera(cam_dict: dict):
    cam = bpy.data.cameras.new(name=cam_dict["camera_name"])
    cam.type = cam_dict["camera_type"]
    cam.clip_start = cam_dict["znear"]
    cam.clip_end = cam_dict["zfar"]

    sensor_width = 36.0
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = sensor_width
    cam.lens = cam_dict["fx"] * sensor_width / cam_dict["width"]
    cam.shift_x = (cam_dict["cx"] - cam_dict["width"] / 2.0) / cam_dict["width"]
    cam.shift_y = (cam_dict["cy"] - cam_dict["height"] / 2.0) / cam_dict["width"]
    cam_obj = bpy.data.objects.new(name=cam_dict["camera_name"], object_data=cam)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.matrix_world = mathutils.Matrix(cam_dict["c2w_blender"])

    scene = bpy.context.scene
    scene.render.resolution_x = cam_dict["width"]
    scene.render.resolution_y = cam_dict["height"]
    scene.render.resolution_percentage = 100
    scene.camera = cam_obj


def enable_sky_texture():
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    sky = nodes.new(type="ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = 0.475
    sky.sun_rotation = 0.0
    sky.altitude = 1000
    sky.air_density = 1.0
    sky.dust_density = 0.0
    sky.ozone_density = 1.0

    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.5

    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(sky.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])
