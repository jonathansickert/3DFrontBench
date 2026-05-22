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
    sky.sky_type = 'NISHITA'
    sky.sun_elevation = 0.475
    sky.sun_rotation = 0.0
    sky.altitude = 1000
    sky.air_density = 1.0
    sky.dust_density = 0.0
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


def render_color(output_path):
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.filepath = str(output_path)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)


def render_normals(output_path):
    mat = bpy.data.materials.new(name="_NormalViz")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    geo   = nodes.new('ShaderNodeNewGeometry')
    sep   = nodes.new('ShaderNodeSeparateXYZ')
    comb  = nodes.new('ShaderNodeCombineColor')
    emis  = nodes.new('ShaderNodeEmission')
    out   = nodes.new('ShaderNodeOutputMaterial')

    # Remap each normal component [-1,1] → [0,1] with scalar Math nodes
    def remap(in_socket, out_socket):
        add = nodes.new('ShaderNodeMath')
        add.operation = 'ADD'
        add.inputs[1].default_value = 1.0
        div = nodes.new('ShaderNodeMath')
        div.operation = 'DIVIDE'
        div.inputs[1].default_value = 2.0
        links.new(in_socket, add.inputs[0])
        links.new(add.outputs[0], div.inputs[0])
        links.new(div.outputs[0], out_socket)

    links.new(geo.outputs['Normal'], sep.inputs['Vector'])
    remap(sep.outputs['X'], comb.inputs['Red'])
    remap(sep.outputs['Y'], comb.inputs['Green'])
    remap(sep.outputs['Z'], comb.inputs['Blue'])
    links.new(comb.outputs['Color'], emis.inputs['Color'])
    links.new(emis.outputs['Emission'], out.inputs['Surface'])

    # material_override is unreliable in background mode; swap slots directly
    saved = {obj: [s.material for s in obj.material_slots]
             for obj in bpy.context.scene.objects if obj.type == 'MESH'}
    for obj in saved:
        for slot in obj.material_slots:
            slot.material = mat

    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.filepath = str(output_path)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

    for obj, mats in saved.items():
        for slot, orig in zip(obj.material_slots, mats):
            slot.material = orig
    bpy.data.materials.remove(mat)


def render_scene(scene_path):
    scene_path = Path(scene_path).resolve()
    render_color(scene_path.with_suffix(".png"))
    render_normals(scene_path.with_name(scene_path.stem + "_normals.png"))


clear_scene()
prepare_scene()
bpy.ops.import_scene.gltf(filepath=scene_a_path)
render_scene(scene_a_path)


clear_scene()
prepare_scene()
bpy.ops.import_scene.gltf(filepath=scene_b_path)
render_scene(scene_b_path)
