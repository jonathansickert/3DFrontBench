import bpy
import json
import mathutils

def add_lights_for_light_meshes():
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH" or "light" not in obj.name.lower():
            continue

        # Emission material so the fixture itself glows
        mat = bpy.data.materials.new(name=f"{obj.name}_emission")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 0.95, 0.85, 1.0)
        emission.inputs["Strength"].default_value = 10.0
        output = nodes.new(type="ShaderNodeOutputMaterial")
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)

        # Point light at centroid so it actually illuminates the room
        world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if not world_verts:
            continue
        centroid = sum(world_verts, mathutils.Vector()) / len(world_verts)
        light_data = bpy.data.lights.new(name=f"{obj.name}_point", type="POINT")
        light_data.energy = 500
        light_data.shadow_soft_size = 0.25
        light_obj = bpy.data.objects.new(name=f"{obj.name}_point", object_data=light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = centroid


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


camera_path = "/Users/jonathansickert/git/3DFrontBench/dataset/0f661df2-0f41-47a4-830c-7444f4a33a03_LivingDiningRoom-12554/camera.json"
scene_path = "/Users/jonathansickert/git/3DFrontBench/assets/scene_with_texture.glb"
output_path = "/Users/jonathansickert/git/3DFrontBench/assets/sample_rendering.png"

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
bpy.context.scene.render.engine = "BLENDER_EEVEE"
bpy.context.scene.use_nodes = False
bpy.context.scene.render.filepath = output_path
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
