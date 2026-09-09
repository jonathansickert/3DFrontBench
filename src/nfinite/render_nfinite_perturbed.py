import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nfinite.configure_rendering import (
    MATERIAL_MODES,
    RENDER_PASSES,
    RESOLUTION_PRESETS,
    VIEW_TRANSFORMS,
    apply_camera_position,
    configure_rendering,
)


def _parse_bool(flag: str, value: str) -> bool:
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"{flag} expects a boolean (true/false), got {value!r}")


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender --background --python render_nfinite_perturbed.py -- "
            "<input_blend> <output_png> [--perturbations <perturbations_json>] "
            "[--material-mode <mode>] [--sun-elevation <degrees>] [--sun-azimuth <degrees>] "
            "[--shadow-softness <degrees>] [--gi-bounces <n>] [--exposure <stops>] "
            "[--view-transform <mode>] [--resolution <preset>] [--samples <n>] "
            "[--denoising <true|false>] [--render-pass <pass>] "
            "[--camera-position <index> --camera-positions-json <path>]"
        )
    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit("Expected at least two arguments: input_blend output_png")

    input_path, output_path = args[0], args[1]
    rest = args[2:]

    perturbations_json = None
    material_mode = "full_pbr"
    sun_elevation = None
    sun_azimuth = None
    shadow_softness = None
    gi_bounces = None
    exposure = None
    view_transform = None
    resolution = None
    samples = 32
    denoising = None
    render_pass = None
    camera_position_index = None
    camera_positions_json_path = None
    i = 0
    while i < len(rest):
        flag = rest[i]
        if flag == "--perturbations" and i + 1 < len(rest):
            perturbations_json = rest[i + 1]
        elif flag == "--camera-position" and i + 1 < len(rest):
            camera_position_index = int(rest[i + 1])
        elif flag == "--camera-positions-json" and i + 1 < len(rest):
            camera_positions_json_path = rest[i + 1]
        elif flag == "--material-mode" and i + 1 < len(rest):
            material_mode = rest[i + 1]
            if material_mode not in MATERIAL_MODES:
                raise SystemExit(
                    f"Unknown material_mode: {material_mode!r}, expected one of {MATERIAL_MODES}"
                )
        elif flag == "--sun-elevation" and i + 1 < len(rest):
            sun_elevation = float(rest[i + 1])
        elif flag == "--sun-azimuth" and i + 1 < len(rest):
            sun_azimuth = float(rest[i + 1])
        elif flag == "--shadow-softness" and i + 1 < len(rest):
            shadow_softness = float(rest[i + 1])
        elif flag == "--gi-bounces" and i + 1 < len(rest):
            gi_bounces = int(rest[i + 1])
        elif flag == "--exposure" and i + 1 < len(rest):
            exposure = float(rest[i + 1])
        elif flag == "--view-transform" and i + 1 < len(rest):
            view_transform = rest[i + 1]
            if view_transform not in VIEW_TRANSFORMS:
                raise SystemExit(
                    f"Unknown view_transform: {view_transform!r}, expected one of {VIEW_TRANSFORMS}"
                )
        elif flag == "--resolution" and i + 1 < len(rest):
            resolution = rest[i + 1]
            if resolution not in RESOLUTION_PRESETS:
                raise SystemExit(
                    f"Unknown resolution preset: {resolution!r}, expected one of {RESOLUTION_PRESETS}"
                )
        elif flag == "--samples" and i + 1 < len(rest):
            samples = int(rest[i + 1])
        elif flag == "--denoising" and i + 1 < len(rest):
            denoising = _parse_bool(flag, rest[i + 1])
        elif flag == "--render-pass" and i + 1 < len(rest):
            render_pass = rest[i + 1]
            if render_pass not in RENDER_PASSES:
                raise SystemExit(f"Unknown render_pass: {render_pass!r}, expected one of {RENDER_PASSES}")
        else:
            raise SystemExit(f"Unexpected argument: {flag}")
        i += 2

    if camera_position_index is not None and camera_positions_json_path is None:
        raise SystemExit("--camera-position requires --camera-positions-json")

    return (
        input_path,
        output_path,
        perturbations_json,
        material_mode,
        sun_elevation,
        sun_azimuth,
        shadow_softness,
        gi_bounces,
        exposure,
        view_transform,
        resolution,
        samples,
        denoising,
        render_pass,
        camera_position_index,
        camera_positions_json_path,
    )


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


def _local_axis(obj, axis: str) -> Vector:
    # objects.json's "x"/"y"/"z" name axes *local to this object* -- e.g. a
    # wall-mounted mirror only allows rotating "y" and sliding "x"/"z"
    # because those are the directions that make sense relative to the wall
    # it's mounted on, not because the scene's global axes happen to run
    # along that wall. Rotating or sliding it along the matching *global*
    # axis instead only coincidentally matches local axes for objects whose
    # own orientation happens to be axis-aligned, and silently produces the
    # wrong motion for anything mounted at an angle. matrix_world's 3x3 part
    # maps the object's own basis vectors into world space, so column
    # _AXIS_INDEX[axis] is exactly that local axis, expressed in world space.
    column = _AXIS_INDEX[axis]
    return obj.matrix_world.to_3x3().col[column].normalized()


def _extent_along(meshes: list, direction: Vector) -> float:
    # Span of the mesh's world-space bounding-box corners projected onto an
    # arbitrary direction. _bounds_min_max's axis-aligned box only gives the
    # right answer for the three global axes -- an object's own local axis
    # (see _local_axis) generally isn't one of them once it's rotated, so
    # the extent along it has to be measured by projection instead.
    direction = direction.normalized()
    projections = [
        (mesh_obj.matrix_world @ Vector(corner)).dot(direction)
        for mesh_obj in meshes
        for corner in mesh_obj.bound_box
    ]
    return max(projections) - min(projections)


def _rotate_object(obj, axis: str, degrees: float):
    # Rotate the whole object in place around the axis through its own
    # center of gravity (approximated as its mesh bounding-box center), not
    # around the object's origin -- furniture origins usually sit at floor
    # level, which would make it swing on that point instead of spinning.
    pivot = _bounds_center(_mesh_descendants(obj))
    rotation = Matrix.Rotation(math.radians(degrees), 4, _local_axis(obj, axis))
    obj.matrix_world = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot) @ obj.matrix_world


def _scale_object(obj, factor: float):
    # Scale about the horizontal (x/y) center but anchor the pivot to the
    # floor contact point (min z), not the bbox center -- pivoting at the
    # center made shrinking objects lift off the floor (and growing ones sink
    # into it), since half of the size change moved the bottom face too.
    min_v, max_v = _bounds_min_max(_mesh_descendants(obj))
    pivot = Vector(((min_v.x + max_v.x) / 2.0, (min_v.y + max_v.y) / 2.0, min_v.z))
    scale = Matrix.Diagonal((factor, factor, factor)).to_4x4()
    obj.matrix_world = Matrix.Translation(pivot) @ scale @ Matrix.Translation(-pivot) @ obj.matrix_world


def _place_object(obj, axis: str, signed_factor: float):
    # Direction is not discovered here: objects.json's "+x"/"-y"/etc. already
    # name the one direction per axis a human verified keeps the object
    # visible for this scene's camera (an unsigned axis means both directions
    # were checked and are fine) -- see sample_placement in
    # create_perturbation_dataset.py. So this just applies the move directly,
    # with no raycasting or other runtime check.
    meshes = _mesh_descendants(obj)
    axis_vector = _local_axis(obj, axis)
    extent = _extent_along(meshes, axis_vector)
    displacement = axis_vector * (signed_factor * extent * 0.5)
    obj.matrix_world = Matrix.Translation(displacement) @ obj.matrix_world


def _apply_perturbations(root, perturbations: dict):
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
            axis, signed_factor = args
            for obj in matches:
                _place_object(obj, axis, signed_factor)
        else:
            raise ValueError(f"Unsupported perturbation type {kind!r} for {label!r}")


(
    input_path,
    output_path,
    perturbations_json,
    material_mode,
    sun_elevation,
    sun_azimuth,
    shadow_softness,
    gi_bounces,
    exposure,
    view_transform,
    resolution,
    samples,
    denoising,
    render_pass,
    camera_position_index,
    camera_positions_json_path,
) = _parse_args()

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
if perturbations_json is not None or camera_position_index is not None:
    root = _find_root(scene)

if camera_position_index is not None:
    apply_camera_position(scene, camera_positions_json_path, root.name, camera_position_index)

if perturbations_json is not None:
    perturbations = json.loads(perturbations_json)
    _apply_perturbations(root, perturbations)

scene.render.filepath = output_path
configure_rendering(
    scene,
    material_mode,
    sun_elevation_degrees=sun_elevation,
    sun_azimuth_degrees=sun_azimuth,
    shadow_softness_degrees=shadow_softness,
    gi_bounces=gi_bounces,
    exposure_stops=exposure,
    view_transform=view_transform,
    resolution=resolution,
    render_pass=render_pass,
    denoising=denoising,
    samples=samples,
)
scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
