import json
import math
import sys

import bpy
from mathutils import Matrix, Vector


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender --background --python render_nfinite_perturbed.py -- "
            "<input_blend> <output_png> [--perturbations <perturbations_json>]"
        )
    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit("Expected at least two arguments: input_blend output_png")

    input_path, output_path = args[0], args[1]
    rest = args[2:]

    perturbations_json = None
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag == "--perturbations" and i + 1 < len(rest):
            perturbations_json = rest[i + 1]
        else:
            raise SystemExit(f"Unexpected argument: {flag}")
        i += 2

    return input_path, output_path, perturbations_json


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


def _free_space(scene, meshes: list, direction: Vector) -> float:
    # Ray-cast from every mesh's world-space bounding-box corner along
    # `direction` to find how far the object can actually travel before its
    # own geometry would reach a wall or another piece of furniture -- the
    # tightest corner distance found is what genuinely bounds the move.
    # Bounding boxes are convex, so casting from the corners (rather than
    # sampling the faces) already catches the first thing in the way.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_names = {mesh_obj.name for mesh_obj in meshes}

    free_space = math.inf
    for mesh_obj in meshes:
        for corner in mesh_obj.bound_box:
            point = mesh_obj.matrix_world @ Vector(corner)
            hit, location, _, _, hit_obj, _ = scene.ray_cast(
                depsgraph, point + direction * 1e-3, direction
            )
            if hit and hit_obj.name not in target_names:
                free_space = min(free_space, (location - point).length)
    return free_space


def _bounded_displacement(scene, meshes: list, axis: str, desired_distance: float) -> Vector:
    # Cap the requested move at the room that's actually there, minus a
    # small clearance margin, so a perturbation can never clip the object
    # into a wall or another object -- discovered by ray-casting instead of
    # trusting the sampled distance blindly.
    sign = 1.0 if desired_distance >= 0 else -1.0
    direction = _AXES[axis] * sign
    margin = 0.02
    free_space = _free_space(scene, meshes, direction)
    distance = min(abs(desired_distance), max(free_space - margin, 0.0))
    return direction * distance


def _largest_visible_displacement(
    scene, camera, obj, meshes: list, original_matrix_world: Matrix, bound: Vector, iterations: int = 8
) -> Vector:
    # Binary search the fraction between staying put (t=0 -- always visible,
    # since every object entering a perturbation is visible to begin with)
    # and the full free-space-bounded move (t=1) for the largest step that
    # keeps the object visible. This assumes visibility only degrades as the
    # object moves further along `bound`, which holds for the vast majority
    # of placements; it isn't a rigorous guarantee against a wall or another
    # object briefly reappearing in view partway through the move.
    obj.matrix_world = Matrix.Translation(bound) @ original_matrix_world
    if _is_visible(scene, camera, meshes):
        return bound

    lo, hi = 0.0, 1.0
    best = 0.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        obj.matrix_world = Matrix.Translation(bound * mid) @ original_matrix_world
        if _is_visible(scene, camera, meshes):
            best = mid
            lo = mid
        else:
            hi = mid
    return bound * best


def _place_object(scene, camera, obj, axis: str, factor: float):
    # Move the object by up to `factor` times its own extent along that
    # axis, e.g. factor=0.67 on "x" shifts it in +x by up to 0.67x its own
    # x-extent -- but bounded by the free space ray-cast finds in that
    # direction (so the move stays physically plausible) and then trimmed
    # further, if needed, to the largest step that keeps the object visible
    # from the camera. Both the sampled direction and its opposite are tried
    # and whichever ends up with the larger visible displacement wins, so
    # the object always ends the perturbation still visible.
    # Translation doesn't change the extent, so it's safe to recompute
    # bounds fresh for each axis/factor pair even after an earlier move was
    # applied.
    meshes = _mesh_descendants(obj)
    min_v, max_v = _bounds_min_max(meshes)
    extent = (max_v - min_v)[_AXIS_INDEX[axis]]
    desired_distance = factor * extent * 0.5

    original_matrix_world = obj.matrix_world.copy()

    best_displacement = Vector((0.0, 0.0, 0.0))
    for signed_distance in (desired_distance, -desired_distance):
        obj.matrix_world = original_matrix_world
        bound = _bounded_displacement(scene, meshes, axis, signed_distance)
        displacement = _largest_visible_displacement(scene, camera, obj, meshes, original_matrix_world, bound)
        if displacement.length > best_displacement.length:
            best_displacement = displacement

    obj.matrix_world = Matrix.Translation(best_displacement) @ original_matrix_world


def _apply_perturbations(scene, root, perturbations: dict):
    categories = list(root.children)

    for label, spec in perturbations.items():
        kind = spec["perturbation_type"]
        if kind == "none":
            continue

        args = spec["args"] or []
        matches = _find_matches(label, categories)
        if kind == "count":
            for obj in matches:
                for mesh_obj in _mesh_descendants(obj):
                    mesh_obj.hide_render = True
        elif kind == "rotation":
            axes, degrees = args
            axis = axes[0]
            for obj in matches:
                _rotate_object(obj, axis, degrees)
        elif kind == "scale":
            (factor,) = args
            for obj in matches:
                _scale_object(obj, factor)
        elif kind == "placement":
            for obj in matches:
                for i in range(0, len(args), 2):
                    axis, factor = args[i], args[i + 1]
                    _place_object(scene, scene.camera, obj, axis, factor)
        else:
            raise ValueError(f"Unsupported perturbation type {kind!r} for {label!r}")


input_path, output_path, perturbations_json = _parse_args()

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

if perturbations_json is not None:
    root = _find_root(scene)
    perturbations = json.loads(perturbations_json)
    _apply_perturbations(scene, root, perturbations)

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
