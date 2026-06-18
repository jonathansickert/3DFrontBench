
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


import bpy
import json
import mathutils

def add_lights_for_light_meshes():
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH" or "light" not in obj.name.lower():
            continue
        world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if not world_verts:
            continue
        centroid = sum(world_verts, mathutils.Vector()) / len(world_verts)
        bpy.ops.object.light_add(type="POINT", location=centroid)
        bpy.context.object.data.energy = 500
        bpy.context.object.data.shadow_soft_size = 0.25


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


camera_path = "/home/jonathansickert/git/DLinVC/dataset/0f661df2-0f41-47a4-830c-7444f4a33a03_LivingDiningRoom-12554/camera.json"
scene_path = "/home/jonathansickert/git/DLinVC/outputs/sample_scene.glb"
output_path = "/home/jonathansickert/git/DLinVC/outputs/sample_rendering.png"

with open(camera_path) as file:
    cam_dict = json.load(file)

bpy.ops.import_scene.gltf(filepath=scene_path)


# Add Camera
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

add_lights_for_light_meshes()
enable_sky_texture()


# Render Scene
bpy.context.scene.use_nodes = False
bpy.context.scene.render.filepath = output_path
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
