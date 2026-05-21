import bpy
import json
import sys
from pathlib import Path

argv = sys.argv[sys.argv.index("--") + 1:]
camera_path = argv[0]
scene_a_path = argv[1]
scene_b_path = argv[2]

print(camera_path, scene_a_path, scene_b_path)

with open(camera_path) as file:
    cam_json = json.load(file)

loc = cam_json["loc"]
rot = cam_json["rot"]
focal_length = cam_json["focal_length"]

def enable_sky_texture():
    world = bpy.context.scene.world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    nodes.clear()
    
    sky = nodes.new(type='ShaderNodeTexSky')
    sky.sky_type = 'SINGLE_SCATTERING'
    sky.sun_elevation = 0.475
    sky.sun_rotation = 0.0
    sky.altitude = 1000
    sky.air_density = 0.4
    sky.aerosol_density = 0.0
    sky.ozone_density = 1.0
    
    bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 0.5
    
    output = nodes.new(type='ShaderNodeOutputWorld')
    links.new(sky.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])
    
def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for col in bpy.data.collections:
        bpy.data.collections.remove(col)
        
def prepare_scene():
    enable_sky_texture()
    bpy.ops.object.camera_add(location=loc, rotation=rot)
    cam = bpy.context.object
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = focal_length
    bpy.context.scene.camera = cam

    
def render_scene(filepath):
    bpy.context.scene.render.filepath = str(filepath)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    
    

clear_scene()
prepare_scene()
bpy.ops.import_scene.gltf(filepath=scene_a_path)
render_scene(Path(scene_a_path).with_suffix(".png"))



clear_scene()
prepare_scene()
bpy.ops.import_scene.gltf(filepath=scene_b_path)
render_scene(Path(scene_b_path).with_suffix(".png"))
