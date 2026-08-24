import bpy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.blender.blender_helper import (
    add_lights_for_light_meshes,
    clear_scene,
    enable_sky_texture,
)


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender --background --python render_scene_top_down_labeled.py -- "
            "<scene_glb> <metadata_json> <output_png> <output_labels_json> [resolution]"
        )
    args = argv[argv.index("--") + 1 :]
    if len(args) < 4:
        raise SystemExit(
            "Expected at least four arguments: scene_glb metadata_json output_png output_labels_json"
        )
    resolution = int(args[4]) if len(args) > 4 else 1536
    return args[0], args[1], args[2], args[3], resolution


# Room-envelope categories to keep, matched by glTF/GLB root-node name prefix.
# CustomizedFeatureWall is included alongside WallOuter because 3D-FRONT uses it
# for outer walls that carry a distinct accent material, not for interior
# partitions -- excluding it would leave a hole in the room boundary.
KEEP_ARCHITECTURE_PREFIXES = ("Floor", "WallOuter", "CustomizedFeatureWall")


def _matches_visible_furniture(obj_name: str, visible_names: set[str]) -> bool:
    # Blender truncates object names to 63 chars, so a furniture root's Blender
    # name may be a prefix of its full metadata name rather than an exact match.
    return any(obj_name == name or name.startswith(obj_name) for name in visible_names)


def _should_keep(obj_name: str, visible_names: set[str]) -> bool:
    if obj_name.startswith(KEEP_ARCHITECTURE_PREFIXES):
        return True
    return _matches_visible_furniture(obj_name, visible_names)


def _ancestor_names(obj):
    names = [obj.name]
    parent = obj.parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.parent
    return names


def _prune_scene(visible_names: set[str]):
    # Check the whole ancestor chain per mesh, not just top-level objects --
    # if the glTF importer ever wraps the scene in an extra root/group object,
    # a top-level-only check would see one unmatched wrapper and delete
    # everything underneath it, including the floor and walls.
    for obj in list(bpy.context.scene.objects):
        if obj.type != "MESH":
            continue
        if not any(_should_keep(name, visible_names) for name in _ancestor_names(obj)):
            bpy.data.objects.remove(obj, do_unlink=True)


def _world_bounds(objects):
    import mathutils

    min_v = mathutils.Vector((float("inf"),) * 3)
    max_v = mathutils.Vector((float("-inf"),) * 3)
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ mathutils.Vector(corner)
            min_v.x, min_v.y, min_v.z = min(min_v.x, world_co.x), min(min_v.y, world_co.y), min(min_v.z, world_co.z)
            max_v.x, max_v.y, max_v.z = max(max_v.x, world_co.x), max(max_v.y, world_co.y), max(max_v.z, world_co.z)
    return min_v, max_v


def _add_overhead_sun():
    # enable_sky_texture()'s sun sits at a low elevation, tuned for the side-on
    # perspective renders where light enters through a window cut into a wall.
    # With the ceiling removed but the outer walls still standing, that low-angle
    # light is blocked by the walls before it reaches most of the floor. A sun
    # lamp pointed straight down travels the same direction the camera looks, so
    # it can't be occluded by the walls for anything the camera can actually see.
    sun_data = bpy.data.lights.new(name="TopDownSun", type="SUN")
    sun_data.energy = 5.0
    sun_data.angle = 0.526

    sun_obj = bpy.data.objects.new(name="TopDownSun", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.0, 0.0, 0.0)


def _add_top_down_camera(min_v, max_v, resolution: int):
    margin = 1.05
    span_x = (max_v.x - min_v.x) * margin
    span_y = (max_v.y - min_v.y) * margin
    ortho_scale = max(span_x, span_y)

    center_x = (min_v.x + max_v.x) / 2.0
    center_y = (min_v.y + max_v.y) / 2.0
    camera_z = max_v.z + 1.0

    cam_data = bpy.data.cameras.new(name="TopDownCamera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = (camera_z - min_v.z) + 1.0

    cam_obj = bpy.data.objects.new(name="TopDownCamera", object_data=cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    # Identity rotation points the camera's local -Z (its view direction) straight
    # down world -Z, since Z is up after the glTF Y-up -> Blender Z-up import.
    cam_obj.location = (center_x, center_y, camera_z)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)

    scene = bpy.context.scene
    scene.camera = cam_obj
    # Square resolution so ortho_scale maps to both image dimensions equally,
    # regardless of the room's own aspect ratio.
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100


def _object_depth(obj) -> int:
    depth = 0
    parent = obj.parent
    while parent is not None:
        depth += 1
        parent = parent.parent
    return depth


def _find_furniture_root(name: str, objects):
    # Multi-part furniture (e.g. cabinets) keeps several leaf meshes with the
    # same ancestor name -- take the shallowest match as the item's root, since
    # that's the transform node the "pos" in metadata.json actually refers to.
    candidates = [obj for obj in objects if obj.name == name or name.startswith(obj.name)]
    if not candidates:
        return None
    return min(candidates, key=_object_depth)


def _project_to_pixels(scene, camera_obj, world_point, resolution: int):
    from bpy_extras.object_utils import world_to_camera_view

    co_norm = world_to_camera_view(scene, camera_obj, world_point)
    pixel_x = co_norm.x * resolution
    pixel_y = (1.0 - co_norm.y) * resolution
    in_view = 0.0 <= co_norm.x <= 1.0 and 0.0 <= co_norm.y <= 1.0 and co_norm.z > 0.0
    return pixel_x, pixel_y, in_view


def _write_labels(furniture: list[dict], resolution: int, output_labels_path: str):
    scene = bpy.context.scene
    camera_obj = scene.camera
    all_objects = list(scene.objects)

    entries = []
    for item in furniture:
        root = _find_furniture_root(item["name"], all_objects)
        if root is None:
            print(f"[top_down] no Blender object matched furniture name {item['name']!r}, skipping label")
            continue
        pixel_x, pixel_y, in_view = _project_to_pixels(scene, camera_obj, root.matrix_world.translation, resolution)
        print(
            f"[top_down] {item['name']!r} -> object {root.name!r} at world "
            f"{tuple(root.matrix_world.translation)} -> pixel ({pixel_x:.1f}, {pixel_y:.1f}) in_view={in_view}"
        )
        entries.append(
            {
                "name": item["name"],
                "label": item.get("label", item["name"]),
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "in_view": in_view,
            }
        )

    with open(output_labels_path, "w") as f:
        json.dump({"resolution": resolution, "objects": entries}, f, indent=2)


scene_path, metadata_path, output_path, output_labels_path, resolution = _parse_args()

with open(metadata_path) as f:
    metadata = json.load(f)
visible_names = set(metadata["visible_furniture"])
visible_furniture = [item for item in metadata["furniture"] if item["name"] in visible_names]

clear_scene()
bpy.ops.import_scene.gltf(filepath=scene_path)

add_lights_for_light_meshes()
enable_sky_texture()

_prune_scene(visible_names)
_add_overhead_sun()

remaining_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not remaining_meshes:
    raise RuntimeError(
        "No mesh objects survived pruning -- the keep-name matching didn't find "
        "the floor/walls/furniture. Nothing to render."
    )

min_v, max_v = _world_bounds(remaining_meshes)
print(f"[top_down] {len(remaining_meshes)} meshes kept: {sorted(o.name for o in remaining_meshes)}")
print(f"[top_down] world bounds min={tuple(min_v)} max={tuple(max_v)}")

_add_top_down_camera(min_v, max_v, resolution=resolution)
# world_to_camera_view() (used by _write_labels below) reads matrix_world directly
# and isn't run through an operator like render is, so it can see a stale
# transform for the camera we just created via direct attribute assignment.
# Force a dependency-graph update so it sees the camera's real pose.
bpy.context.view_layer.update()
print(f"[top_down] camera location={tuple(bpy.context.scene.camera.location)} ortho_scale={bpy.context.scene.camera.data.ortho_scale}")

_write_labels(visible_furniture, resolution, output_labels_path)

# Render
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
