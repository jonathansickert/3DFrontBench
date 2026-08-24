import glob
import json
import math
import os
import shutil
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

# Must match LABELS_MARKER in render_single.py, which pulls the label data
# back out of this script's stdout -- nothing but the rendered PNG is ever
# written to disk.
LABELS_MARKER = "NFINITE_LABELS_JSON:"


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender --background --python render_nfinite.py -- "
            "<input_blend> <output_png> [--labels <objects_json>] [--perturbations <perturbations_json>]"
        )
    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit("Expected at least two arguments: input_blend output_png")

    input_path, output_path = args[0], args[1]
    rest = args[2:]

    objects_json_path = None
    perturbations_json_path = None
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag == "--labels" and i + 1 < len(rest):
            objects_json_path = rest[i + 1]
        elif flag == "--perturbations" and i + 1 < len(rest):
            perturbations_json_path = rest[i + 1]
        else:
            raise SystemExit(f"Unexpected argument: {flag}")
        i += 2

    return input_path, output_path, objects_json_path, perturbations_json_path


def _find_root(scene):
    # Every nfinite scene is built as a single parentless EMPTY named after the
    # scene, with one EMPTY per object category as its direct children and one
    # EMPTY per instance one level below that -- verified across the whole
    # indoor/ dataset before relying on it here.
    roots = [obj for obj in scene.objects if obj.parent is None]
    if len(roots) != 1:
        raise RuntimeError(
            f"Expected exactly one parentless root object, found {len(roots)}: "
            f"{[o.name for o in roots]}"
        )
    return roots[0]


def _find_matches(label: str, categories: list):
    instances = [instance for category in categories for instance in category.children]
    exact_instances = [obj for obj in instances if obj.name == label]
    if exact_instances:
        return exact_instances

    for category in categories:
        if category.name == label:
            return list(category.children)

    return []


def _mesh_descendants(obj):
    meshes = [obj] if obj.type == "MESH" else []
    stack = list(obj.children)
    while stack:
        child = stack.pop()
        if child.type == "MESH":
            meshes.append(child)
        stack.extend(child.children)
    return meshes


def _build_label_entries(root, objects_json_path: str):
    with open(objects_json_path) as f:
        objects_by_scene = json.load(f)
    labels = [entry["Object"] for entry in objects_by_scene[root.name]]
    categories = list(root.children)

    # Each matched Blender object gets its own unique mask_id so its visible
    # pixels can be told apart in the IndexOB pass below, even when several
    # objects share the same digit (e.g. two "MIRROR" instances both labeled 3).
    entries = []
    for index, label in enumerate(labels, start=1):
        for obj in _find_matches(label, categories):
            mask_id = len(entries) + 1
            for mesh_obj in _mesh_descendants(obj):
                mesh_obj.pass_index = mask_id
            entries.append({"index": index, "label": label, "object_name": obj.name, "mask_id": mask_id})
    return entries


_AXES = {"x": Vector((1, 0, 0)), "y": Vector((0, 1, 0)), "z": Vector((0, 0, 1))}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _bounds_min_max(meshes: list) -> tuple:
    min_v = Vector((float("inf"),) * 3)
    max_v = Vector((float("-inf"),) * 3)
    for mesh_obj in meshes:
        for corner in mesh_obj.bound_box:
            world_co = mesh_obj.matrix_world @ Vector(corner)
            min_v.x, min_v.y, min_v.z = min(min_v.x, world_co.x), min(min_v.y, world_co.y), min(min_v.z, world_co.z)
            max_v.x, max_v.y, max_v.z = max(max_v.x, world_co.x), max(max_v.y, world_co.y), max(max_v.z, world_co.z)
    return min_v, max_v


def _bounds_center(meshes: list) -> Vector:
    min_v, max_v = _bounds_min_max(meshes)
    return (min_v + max_v) / 2.0


def _rotate_object(obj, axis: str, degrees: float):
    # Rotate the whole object in place around the axis through its own
    # center of gravity (approximated as its mesh bounding-box center), not
    # around the object's origin -- furniture origins usually sit at floor
    # level, which would make it swing on that point instead of spinning.
    pivot = _bounds_center(_mesh_descendants(obj))
    rotation = Matrix.Rotation(math.radians(degrees), 4, _AXES[axis])
    obj.matrix_world = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot) @ obj.matrix_world


def _scale_object(obj, factor: float):
    # Same reasoning as _rotate_object: scale about the object's own bounding-
    # box center so it grows/shrinks in place instead of drifting away from
    # (or into) the floor when the origin isn't at the geometric center.
    pivot = _bounds_center(_mesh_descendants(obj))
    scale = Matrix.Diagonal((factor, factor, factor)).to_4x4()
    obj.matrix_world = Matrix.Translation(pivot) @ scale @ Matrix.Translation(-pivot) @ obj.matrix_world


def _is_visible(scene, camera, meshes: list) -> bool:
    # A plain camera-frustum check isn't enough: a point can be angularly
    # inside the frustum while still being behind a wall or another object,
    # or the object could be pushed absurdly far away and still project
    # inside [0,1]. So each bounding-box corner is frustum-tested and then
    # ray-cast toward the camera through Blender's BVH (cheap -- no
    # rendering) to confirm nothing solid actually blocks the line of sight.
    from bpy_extras.object_utils import world_to_camera_view

    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_origin = camera.matrix_world.translation
    target_names = {mesh_obj.name for mesh_obj in meshes}

    for mesh_obj in meshes:
        for corner in mesh_obj.bound_box:
            point = mesh_obj.matrix_world @ Vector(corner)

            co_norm = world_to_camera_view(scene, camera, point)
            if not (0.0 <= co_norm.x <= 1.0 and 0.0 <= co_norm.y <= 1.0 and co_norm.z > 0.0):
                continue

            to_camera = cam_origin - point
            distance = to_camera.length
            if distance < 1e-6:
                return True
            direction = to_camera / distance
            # Nudge the ray origin off the surface so it doesn't immediately
            # self-intersect the face it just came from.
            hit, _, _, _, hit_obj, _ = scene.ray_cast(
                depsgraph, point + direction * 1e-3, direction, distance=distance
            )
            if not hit or hit_obj.name in target_names:
                return True
    return False


def _place_object(scene, camera, obj, axis: str, factor: float):
    # Move the object by `factor` times its own extent along that axis, e.g.
    # factor=0.67 on "x" shifts it in +x by 0.67x its own x-extent.
    # Translation doesn't change the extent, so it's safe to recompute bounds
    # fresh for each axis/factor pair even after an earlier move was applied.
    meshes = _mesh_descendants(obj)
    min_v, max_v = _bounds_min_max(meshes)
    extent = (max_v - min_v)[_AXIS_INDEX[axis]]
    displacement = _AXES[axis] * (factor * extent * 0.5)

    original_matrix_world = obj.matrix_world.copy()
    obj.matrix_world = Matrix.Translation(displacement) @ original_matrix_world
    if _is_visible(scene, camera, meshes):
        return

    # That direction pushed the object out of view (out of frame or behind
    # something) -- fall back to the opposite direction along the same axis.
    obj.matrix_world = Matrix.Translation(-displacement) @ original_matrix_world


def _apply_perturbations(scene, root, perturbations_json_path: str):
    with open(perturbations_json_path) as f:
        perturbations_by_scene = json.load(f)
    categories = list(root.children)

    for label, spec in perturbations_by_scene[root.name].items():
        kind = spec[0]
        matches = _find_matches(label, categories)
        if kind == "count":
            for obj in matches:
                for mesh_obj in _mesh_descendants(obj):
                    mesh_obj.hide_render = True
        elif kind == "rotation":
            _, axes, degrees = spec
            axis = axes[0]
            for obj in matches:
                _rotate_object(obj, axis, degrees)
        elif kind == "scale":
            _, factor = spec
            for obj in matches:
                _scale_object(obj, factor)
        elif kind == "placement":
            axis_factor_pairs = spec[1:]
            for obj in matches:
                for i in range(0, len(axis_factor_pairs), 2):
                    axis, factor = axis_factor_pairs[i], axis_factor_pairs[i + 1]
                    _place_object(scene, scene.camera, obj, axis, factor)
        else:
            raise ValueError(f"Unsupported perturbation type {kind!r} for {label!r}")


def _setup_object_index_pass(scene, tmp_dir: str):
    bpy.context.view_layer.use_pass_object_index = True
    scene.render.use_compositing = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    file_output = tree.nodes.new("CompositorNodeOutputFile")

    # Composite mirrors the plain Combined pass straight through so the saved
    # PNG is exactly the beauty render. IndexOB can't be read back from a
    # Viewer node in background mode -- its backing image is only populated
    # when a compositor UI area is redrawing it, which headless rendering
    # never does -- so it's written to a File Output node instead and read
    # back from tmp_dir, which the caller deletes before this process exits.
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    file_output.base_path = tmp_dir
    file_output.format.file_format = "OPEN_EXR"
    file_output.format.color_depth = "32"
    file_output.file_slots.clear()
    file_output.file_slots.new("mask")
    tree.links.new(render_layers.outputs["IndexOB"], file_output.inputs["mask"])


def _read_object_index_map(tmp_dir: str) -> np.ndarray:
    exr_files = glob.glob(os.path.join(tmp_dir, "*.exr"))
    if len(exr_files) != 1:
        raise RuntimeError(f"Expected exactly one object-index EXR in {tmp_dir}, found {exr_files}")

    image = bpy.data.images.load(exr_files[0])
    try:
        width, height = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, image.channels)
        # The object-index value is carried in the R channel; round away the
        # sub-pixel blending Cycles' anti-aliasing introduces at silhouette edges.
        index_map = np.rint(pixels[..., 0]).astype(np.int64)
    finally:
        bpy.data.images.remove(image)

    # Blender's image buffers are bottom-up; flip to match the top-down row
    # order of the PNG this gets overlaid onto.
    return np.flipud(index_map)


def _rle_encode(flat: np.ndarray) -> list:
    change_points = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(flat)]))
    values = flat[starts]
    lengths = ends - starts
    return [[int(v), int(n)] for v, n in zip(values, lengths)]


def _emit_label_data(entries: list, index_map: np.ndarray):
    height, width = index_map.shape
    payload = {
        "resolution_x": width,
        "resolution_y": height,
        "entries": entries,
        "rle": _rle_encode(index_map.reshape(-1)),
    }
    print(LABELS_MARKER + json.dumps(payload))


input_path, output_path, objects_json_path, perturbations_json_path = _parse_args()

bpy.ops.wm.open_mainfile(filepath=input_path)

scene = bpy.context.scene
if scene.camera is None:
    cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]
    if len(cameras) != 1:
        raise RuntimeError(
            f"{input_path} has no active camera and {len(cameras)} camera objects "
            "in the scene; expected exactly one to disambiguate."
        )
    scene.camera = cameras[0]

root = None
if objects_json_path is not None or perturbations_json_path is not None:
    root = _find_root(scene)

if perturbations_json_path is not None:
    _apply_perturbations(scene, root, perturbations_json_path)

label_entries = None
mask_tmp_dir = None
if objects_json_path is not None:
    label_entries = _build_label_entries(root, objects_json_path)
    mask_tmp_dir = tempfile.mkdtemp(prefix="nfinite_mask_")
    _setup_object_index_pass(scene, mask_tmp_dir)

scene.render.filepath = output_path
cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
cycles_prefs.compute_device_type = "CUDA"
cycles_prefs.get_devices()
for device in cycles_prefs.devices:
    device.use = device.type == "CUDA"

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)

if label_entries is not None:
    try:
        index_map = _read_object_index_map(mask_tmp_dir)
    finally:
        shutil.rmtree(mask_tmp_dir, ignore_errors=True)
    _emit_label_data(label_entries, index_map)
