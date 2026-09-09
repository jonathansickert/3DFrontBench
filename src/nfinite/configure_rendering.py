import colorsys
import json
import math

import bpy
import numpy as np

MATERIAL_MODES = ("full_pbr", "flat_color", "clay")
_FLAT_ROUGHNESS = 0.5
RESOLUTION_PRESETS = ("512", "hd", "full_hd", "4k")
_RESOLUTION_TARGET_WIDTH = {"512": 512, "hd": 1280, "4k": 3840}
RENDER_PASSES = (
    "beauty",
    "flat_albedo",
    "depth",
    "normal",
    "ao_only",
    "instance_segmentation",
    "wireframe",
    "matcap",
)
VIEW_TRANSFORMS = ("Standard", "Filmic", "AgX")


def _unlink_input(links, input_socket, flat_value=None):
    for existing in list(input_socket.links):
        links.remove(existing)
    if flat_value is not None:
        input_socket.default_value = flat_value


def _flatten_bsdf(links, bsdf, color: tuple, roughness: float):
    _unlink_input(links, bsdf.inputs["Base Color"], flat_value=color)
    if "Roughness" in bsdf.inputs:
        _unlink_input(links, bsdf.inputs["Roughness"], flat_value=roughness)
    if "Metallic" in bsdf.inputs:
        _unlink_input(links, bsdf.inputs["Metallic"], flat_value=0.0)
    normal_input = bsdf.inputs.get("Normal")
    if normal_input is not None:
        _unlink_input(links, normal_input)


def _average_texture_color(image) -> tuple:
    width, height = image.size
    channels = image.channels
    if width == 0 or height == 0 or channels == 0:
        return (0.5, 0.5, 0.5, 1.0)

    buffer = np.empty(width * height * channels, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    mean = buffer.reshape(-1, channels).mean(axis=0)

    if channels >= 3:
        return (float(mean[0]), float(mean[1]), float(mean[2]), 1.0)
    v = float(mean[0])
    return (v, v, v, 1.0)


def _material_flat_color(bsdf) -> tuple:
    """The material's actual color -- its unlinked Base Color value, or the average pixel
    color of whatever image texture feeds it (most materials here are exactly one of these
    two cases; both 1x1 "solid color as texture" tricks and real multi-hundred-pixel photos
    average out correctly). Falls back to the socket's own default_value for anything wired
    through a more complex node chain (e.g. a Mix of vertex color + texture) that isn't worth
    evaluating here."""
    base_color_input = bsdf.inputs["Base Color"]
    if not base_color_input.is_linked:
        return tuple(base_color_input.default_value)

    source = base_color_input.links[0].from_node
    if source.type == "TEX_IMAGE" and source.image is not None:
        return _average_texture_color(source.image)

    return tuple(base_color_input.default_value)


def apply_material_mode(mode: str):
    """Rewire every material's Principled BSDF to match a material-quality mode. Both
    non-default modes keep the BSDF (so real Cycles lighting/shadows/GI still apply) and
    neither bypasses light transport, so both compose with sun_elevation/gi_bounces rather
    than making them no-ops. Same material setup and the same scene-wide roughness in both
    (~0.5, metallic 0, no image textures/normal maps) -- the only difference is color:

    - "full_pbr": the scene as authored -- albedo/roughness/metallic/normal maps, per-material
      surface response, specular highlights that differ between e.g. a ceramic mug and a
      fabric cushion.
    - "clay": every surface gets the *same* grey Principled BSDF. Sofa, wall, floor, chair:
      all identical grey. Objects are distinguishable only by silhouette and by how light
      falls across them.
    - "flat_color": same material setup, but Base Color now varies per object using each
      material's *actual* color (its own unlinked value, or the average color of its own
      texture) -- e.g. chair brown, wall off-white, sofa green. The wood grain is gone; the
      fact that the chair is a different color from the wall is back.
    """
    if mode == "full_pbr":
        return
    if mode not in MATERIAL_MODES:
        raise ValueError(f"Unknown material_mode: {mode!r}, expected one of {MATERIAL_MODES}")

    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if bsdf is None or output is None:
            continue

        color = (0.6, 0.6, 0.6, 1.0) if mode == "clay" else _material_flat_color(bsdf)
        _flatten_bsdf(links, bsdf, color=color, roughness=_FLAT_ROUGHNESS)


def _find_sun_sky_node(scene):
    # Walked structurally from the World's Background node rather than matched by name,
    # since every NFINITE scene wires an (unused) "Sky Texture Night" node alongside the
    # one that actually drives the window daylight -- only the Background's Color link
    # tells them apart reliably.
    world = scene.world
    if world is None or not world.use_nodes:
        raise RuntimeError("Scene has no node-based World to control the sun on")

    background = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if background is None or not background.inputs["Color"].is_linked:
        raise RuntimeError("World has no Background node wired to a sky texture")

    sky = background.inputs["Color"].links[0].from_node
    if sky.type != "TEX_SKY":
        raise RuntimeError(f"Background Color is driven by a {sky.type} node, not a Sky Texture")
    return sky


def apply_camera_position(scene, camera_positions_path: str, scene_name: str, index: int):
    """Move the scene's camera to one of a set of pre-recorded alternative viewpoints,
    keyed by scene name (the root EMPTY's name, matching objects.json) in a JSON file
    shaped like {scene_name: [{"x", "y", "z", "wrot", "xrot", "yrot", "zrot"}, ...]}.
    Each entry is a full pose captured together (location + wxyz rotation quaternion),
    so -- unlike sun_elevation/sun_azimuth -- there's no separate "aim at the room"
    step: the stored quaternion already points the camera at whatever that viewpoint
    was framing.
    """
    with open(camera_positions_path) as f:
        positions_by_scene = json.load(f)

    positions = positions_by_scene.get(scene_name)
    if not positions:
        raise ValueError(f"No camera positions found for scene {scene_name!r} in {camera_positions_path}")
    if not (0 <= index < len(positions)):
        raise ValueError(
            f"camera_position index {index} out of range for {scene_name!r}: "
            f"{len(positions)} alternative(s) available"
        )

    position = positions[index]
    camera = scene.camera
    camera.location = (position["x"], position["y"], position["z"])
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (position["wrot"], position["xrot"], position["yrot"], position["zrot"])


def apply_sun_elevation(scene, degrees: float):
    """Set the elevation (degrees above the horizon) of the sky texture driving the scene's
    dominant daylight source. Low elevations cast long, dramatic shadows through the window;
    high elevations flatten them."""
    sky = _find_sun_sky_node(scene)
    sky.sun_elevation = math.radians(degrees)


def apply_sun_azimuth(scene, degrees: float):
    """Set the compass rotation (degrees) of the sky texture driving the scene's dominant
    daylight source. Unlike sun_elevation (shadow length), this changes shadow *direction* --
    which wall the window light rakes across, and whether a direct sunbeam lands in the shot
    at all. It wraps at 360 degrees, so treat it as periodic rather than a plain bounded
    scalar when sampling it."""
    sky = _find_sun_sky_node(scene)
    sky.sun_rotation = math.radians(degrees)


def apply_shadow_softness(scene, degrees: float):
    """Set the angular size (in degrees) of the sky texture's sun disc -- distinct from
    sun_elevation/sun_azimuth, which move the light without changing how large it appears. A
    small angle (near the real sun's ~0.5 degrees) casts sharp, crisp-edged shadows from a
    near-point source; a large one spreads it into a soft, area-light-like source with wide,
    feathered penumbras."""
    sky = _find_sun_sky_node(scene)
    sky.sun_size = math.radians(degrees)


def apply_gi_bounces(scene, bounces: int):
    """Cap the number of diffuse light bounces Cycles simulates. 0 collapses global
    illumination to direct light only (dark corners, no color bleed); higher values let
    indirect light fill occluded areas more completely."""
    scene.cycles.max_bounces = max(scene.cycles.max_bounces, bounces)
    scene.cycles.diffuse_bounces = bounces


def apply_exposure(scene, stops: float):
    """Set the scene's exposure in stops (EV), applied as a post-render tonemap rather than a
    change to the actual light transport -- unlike sun_elevation/gi_bounces, this doesn't cost
    extra samples to resolve. 0 leaves the scene's authored exposure untouched; positive values
    brighten the whole image (pushing the window toward blowout while lifting interior shadow
    detail), negative values darken it (pulling window highlights back at the cost of a darker
    interior). This is the classic photographic "expose for highlights vs. shadows" trade-off.
    """
    scene.view_settings.exposure = stops


def apply_view_transform(scene, view_transform: str):
    """Set the color-management view transform used to tonemap the render -- another
    post-render display setting like exposure_stops, not a change to the underlying light
    transport. "Standard" is a linear-to-sRGB curve with no highlight rolloff (harsh, easily
    blown-out windows); "Filmic" and "AgX" both compress highlights into a soft rolloff instead
    of clipping them, AgX being Blender's newer, more saturated default look since 4.0."""
    if view_transform not in VIEW_TRANSFORMS:
        raise ValueError(f"Unknown view_transform: {view_transform!r}, expected one of {VIEW_TRANSFORMS}")
    scene.view_settings.view_transform = view_transform


def apply_denoising(scene, enabled: bool):
    """Toggle Cycles' denoiser. On (the scene-authored default), it cleans up sample noise at
    the cost of some fine-detail smoothing; off, low sample counts show visibly grainy/mottled
    surfaces instead of a clean image."""
    scene.cycles.use_denoising = enabled


def apply_resolution(scene, preset: str):
    """Scale the render resolution to a named preset via resolution_percentage, which
    resamples the full frame uniformly -- unlike cropping, this leaves aspect ratio, field of
    view, and framing exactly as authored, so object visibility is unaffected.

    - "512": ~512px on the long edge -- roughly what many VLM preprocessing pipelines
      downsample to internally, so higher native resolution may already be wasted detail.
    - "hd": 1280px on the long edge (720p-equivalent).
    - "full_hd": the scene's native resolution, unchanged.
    - "4k": ~3840px on the long edge (2160p-equivalent) -- supersamples above the scene's
      native resolution. Since the underlying geometry/textures don't gain new detail past
      that point, this mainly buys cleaner anti-aliasing and sharper texture sampling rather
      than genuinely new information, at roughly 4x the render cost of full_hd.
    """
    if preset not in RESOLUTION_PRESETS:
        raise ValueError(f"Unknown resolution preset: {preset!r}, expected one of {RESOLUTION_PRESETS}")

    if preset == "full_hd":
        scene.render.resolution_percentage = 100
        return

    target_width = _RESOLUTION_TARGET_WIDTH[preset]
    scene.render.resolution_percentage = round(target_width / scene.render.resolution_x * 100)


def _override_all_materials(mat):
    # OBJECT-linked (not DATA-linked) so this never mutates a mesh data-block shared by
    # multiple instanced objects -- each object gets its own override independently.
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.material_slots:
            obj.data.materials.append(None)
        for slot in obj.material_slots:
            slot.link = "OBJECT"
            slot.material = mat


def _route_pass_to_composite(pass_name: str, *, remap_signed: bool = False, normalize: bool = False):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    output_socket = render_layers.outputs[pass_name]

    if normalize:
        normalize_node = tree.nodes.new("CompositorNodeNormalize")
        tree.links.new(output_socket, normalize_node.inputs[0])
        output_socket = normalize_node.outputs[0]

    if remap_signed:
        # Normal/vector passes carry components in [-1, 1]; remap to [0, 1] so negative
        # components don't just clip to black in the saved PNG.
        multiply = tree.nodes.new("CompositorNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 1.0
        multiply.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
        tree.links.new(output_socket, multiply.inputs[1])

        add = tree.nodes.new("CompositorNodeMixRGB")
        add.blend_type = "ADD"
        add.inputs[0].default_value = 1.0
        add.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
        tree.links.new(multiply.outputs[0], add.inputs[1])
        output_socket = add.outputs[0]

    tree.links.new(output_socket, composite.inputs["Image"])
    # Data passes aren't meant to look "photographic" -- AgX/Filmic curves would warp the
    # raw depth/normal/AO values into something misleading to visualize or feed to a model.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0


def apply_flat_albedo():
    """Emit each material's texture/base color directly, bypassing all lighting -- the
    lighting-independent counterpart to full_pbr: same colors/textures, zero shading."""
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if bsdf is None or output is None:
            continue

        emission = nodes.new(type="ShaderNodeEmission")
        base_color_input = bsdf.inputs["Base Color"]
        source_link = next((link for link in base_color_input.links), None)
        if source_link is not None:
            links.new(source_link.from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = base_color_input.default_value

        for existing in list(output.inputs["Surface"].links):
            links.remove(existing)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])


def apply_depth():
    """Output the camera-space depth buffer as a normalized grayscale image -- pure
    geometry, no material or light involved at all."""
    bpy.context.view_layer.use_pass_z = True
    _route_pass_to_composite("Depth", normalize=True)


def apply_normal():
    """Output world-space surface normals as an RGB image (XYZ mapped to 0..1) -- pure
    geometry, no material or light involved at all."""
    bpy.context.view_layer.use_pass_normal = True
    _route_pass_to_composite("Normal", remap_signed=True)


def apply_ao_only():
    """Override every object with a shared material that emits the Ambient Occlusion
    *shader node*'s factor as flat grayscale. Deliberately not the render-layer "AO" pass:
    that turned out to be weighted by the world/environment strength (empirically verified
    near-black in enclosed areas away from the window, i.e. sun-elevation-dependent) --
    this node instead samples only nearby geometry, independent of every light including
    the sun."""
    mat = bpy.data.materials.new(name="__ao_only_override")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 1.0
    ao.samples = 16
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(ao.outputs["AO"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    _override_all_materials(mat)
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.exposure = 0.0


def apply_instance_segmentation():
    """Give every mesh object its own flat, unlit color (evenly spaced around the hue
    wheel) -- an identity map for "which object is which", independent of material or
    lighting. Distinct from apply_material_mode('clay'): this preserves per-instance
    identity instead of collapsing everything to one shared gray."""
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    total = max(len(mesh_objects), 1)

    for index, obj in enumerate(mesh_objects):
        hue = (index / total) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)

        mat = bpy.data.materials.new(name=f"__instance_seg_{index}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (r, g, b, 1.0)
        output = nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

        if not obj.material_slots:
            obj.data.materials.append(None)
        for slot in obj.material_slots:
            slot.link = "OBJECT"
            slot.material = mat

    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.exposure = 0.0


def apply_wireframe():
    """Override every object with a single shared material that emits white on mesh edges
    and black elsewhere (via a Wireframe node), independent of material or lighting."""
    mat = bpy.data.materials.new(name="__wireframe_override")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    wireframe = nodes.new("ShaderNodeWireframe")
    wireframe.inputs["Size"].default_value = 0.01
    background = nodes.new("ShaderNodeEmission")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    lines = nodes.new("ShaderNodeEmission")
    lines.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(wireframe.outputs["Fac"], mix.inputs[0])
    links.new(background.outputs["Emission"], mix.inputs[1])
    links.new(lines.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    _override_all_materials(mat)
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.exposure = 0.0


def apply_matcap():
    """Override every object with a single shared material that shades purely by viewing
    angle (a Fresnel term through a color ramp) -- an approximation of classic MatCap
    shading (fixed "studio lighting" baked into a lookup, independent of scene lights)
    built from Cycles nodes rather than a literal matcap image texture."""
    mat = bpy.data.materials.new(name="__matcap_override")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.45

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.08, 1.0)
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = (0.5, 0.55, 0.6, 1.0)
    ramp.color_ramp.elements[2].position = 1.0
    ramp.color_ramp.elements[2].color = (0.95, 0.95, 1.0, 1.0)

    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(fresnel.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    _override_all_materials(mat)
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.exposure = 0.0


def apply_render_pass(mode: str):
    if mode == "beauty":
        return
    if mode not in RENDER_PASSES:
        raise ValueError(f"Unknown render_pass: {mode!r}, expected one of {RENDER_PASSES}")

    {
        "flat_albedo": apply_flat_albedo,
        "depth": apply_depth,
        "normal": apply_normal,
        "ao_only": apply_ao_only,
        "instance_segmentation": apply_instance_segmentation,
        "wireframe": apply_wireframe,
        "matcap": apply_matcap,
    }[mode]()


def configure_cycles(scene, samples: int = 32):
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    cycles_prefs.compute_device_type = "CUDA"
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = device.type == "CUDA"

    scene.cycles.device = "GPU"
    scene.cycles.samples = samples


def configure_rendering(
    scene,
    material_mode: str = "full_pbr",
    sun_elevation_degrees: float | None = None,
    sun_azimuth_degrees: float | None = None,
    shadow_softness_degrees: float | None = None,
    gi_bounces: int | None = None,
    exposure_stops: float | None = None,
    view_transform: str | None = None,
    resolution: str | None = None,
    render_pass: str | None = None,
    denoising: bool | None = None,
    samples: int = 32,
):
    """Apply the rendering-quality axes shared by every nfinite render entrypoint.

    render_pass (when not "beauty"/None) is orthogonal to and overrides material_mode: it
    replaces materials/compositor output outright, so whichever runs last wins. It's applied
    after exposure_stops/view_transform specifically so its own view-transform reset always wins
    for the geometric/data passes (depth/normal/ao_only/instance_segmentation/wireframe/matcap),
    which aren't meant to be tonemapped.
    """
    apply_material_mode(material_mode)
    if sun_elevation_degrees is not None:
        apply_sun_elevation(scene, sun_elevation_degrees)
    if sun_azimuth_degrees is not None:
        apply_sun_azimuth(scene, sun_azimuth_degrees)
    if shadow_softness_degrees is not None:
        apply_shadow_softness(scene, shadow_softness_degrees)
    if gi_bounces is not None:
        apply_gi_bounces(scene, gi_bounces)
    if exposure_stops is not None:
        apply_exposure(scene, exposure_stops)
    if view_transform is not None:
        apply_view_transform(scene, view_transform)
    if resolution is not None:
        apply_resolution(scene, resolution)
    if render_pass is not None:
        apply_render_pass(render_pass)
    if denoising is not None:
        apply_denoising(scene, denoising)
    configure_cycles(scene, samples=samples)
