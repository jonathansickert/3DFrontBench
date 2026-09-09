import subprocess
import os
from pathlib import Path
import argparse

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).parent / "render_nfinite_perturbed.py"
CAMERA_POSITIONS_JSON = Path("/home/jonathansickert/git/3DFrontBench/assets/camera_positions.json")
MATERIAL_MODES = ("full_pbr", "flat_color", "clay")
RESOLUTION_PRESETS = ("512", "hd", "full_hd", "4k")
VIEW_TRANSFORMS = ("Standard", "Filmic", "AgX")
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

assert RENDER_SCRIPT.exists(), RENDER_SCRIPT


def render_perturbed(
    blend_path: str,
    out_path: str,
    perturbations: str,
    material_mode: str = "full_pbr",
    sun_elevation: float | None = None,
    sun_azimuth: float | None = None,
    shadow_softness: float | None = None,
    gi_bounces: int | None = None,
    exposure: float | None = None,
    view_transform: str | None = None,
    resolution: str | None = None,
    samples: int | None = None,
    denoising: bool | None = None,
    render_pass: str | None = None,
    camera_position: int | None = None,
) -> None:
    if material_mode not in MATERIAL_MODES:
        raise ValueError(f"Unknown material_mode: {material_mode!r}, expected one of {MATERIAL_MODES}")
    if resolution is not None and resolution not in RESOLUTION_PRESETS:
        raise ValueError(f"Unknown resolution preset: {resolution!r}, expected one of {RESOLUTION_PRESETS}")
    if view_transform is not None and view_transform not in VIEW_TRANSFORMS:
        raise ValueError(f"Unknown view_transform: {view_transform!r}, expected one of {VIEW_TRANSFORMS}")
    if render_pass is not None and render_pass not in RENDER_PASSES:
        raise ValueError(f"Unknown render_pass: {render_pass!r}, expected one of {RENDER_PASSES}")

    cmd = [
        BLENDER_PATH,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(blend_path),
        str(out_path),
        "--perturbations",
        perturbations,
        "--material-mode",
        material_mode,
    ]

    if sun_elevation is not None:
        cmd += ["--sun-elevation", str(sun_elevation)]
    if sun_azimuth is not None:
        cmd += ["--sun-azimuth", str(sun_azimuth)]
    if shadow_softness is not None:
        cmd += ["--shadow-softness", str(shadow_softness)]
    if gi_bounces is not None:
        cmd += ["--gi-bounces", str(gi_bounces)]
    if exposure is not None:
        cmd += ["--exposure", str(exposure)]
    if view_transform is not None:
        cmd += ["--view-transform", view_transform]
    if resolution is not None:
        cmd += ["--resolution", resolution]
    if samples is not None:
        cmd += ["--samples", str(samples)]
    if denoising is not None:
        cmd += ["--denoising", str(denoising)]
    if render_pass is not None:
        cmd += ["--render-pass", render_pass]
    if camera_position is not None:
        cmd += ["--camera-position", str(camera_position), "--camera-positions-json", str(CAMERA_POSITIONS_JSON)]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Rendering failed: {result.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("blend_path", type=Path, help="Source scene")
    parser.add_argument("out_path", type=Path)
    parser.add_argument(
        "--perturbations",
        type=str,
        required=True,
        help=(
            "Perturbations for this scene as a JSON-formatted string, e.g. "
            '\'{"SOCKET": {"perturbation_type": "rotation", "args": [["x"], 52]}}\''
        ),
    )
    parser.add_argument(
        "--material-mode",
        choices=MATERIAL_MODES,
        default="full_pbr",
        help="Material/lighting quality mode",
    )
    parser.add_argument(
        "--sun-elevation",
        type=float,
        default=None,
        help="Elevation (degrees above horizon) of the window daylight's sky texture",
    )
    parser.add_argument(
        "--sun-azimuth",
        type=float,
        default=None,
        help="Compass rotation (degrees) of the window daylight's sky texture",
    )
    parser.add_argument(
        "--shadow-softness",
        type=float,
        default=None,
        help="Angular size (degrees) of the sky texture's sun disc -- controls shadow softness",
    )
    parser.add_argument(
        "--gi-bounces",
        type=int,
        default=None,
        help="Number of diffuse GI bounces Cycles simulates",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Post-render exposure in stops (EV)",
    )
    parser.add_argument(
        "--view-transform",
        choices=VIEW_TRANSFORMS,
        default=None,
        help="Color-management view transform used to tonemap the render",
    )
    parser.add_argument(
        "--resolution",
        choices=RESOLUTION_PRESETS,
        default=None,
        help="Output resolution preset (aspect ratio/FOV unchanged, only pixel count scales)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Cycles sample count (default: 32, set by render_nfinite_perturbed.py)",
    )
    parser.add_argument(
        "--denoising",
        type=lambda value: value.lower() in ("1", "true", "yes", "on"),
        default=None,
        help="Toggle Cycles' denoiser (default: on, set by render_nfinite_perturbed.py)",
    )
    parser.add_argument(
        "--render-pass",
        choices=RENDER_PASSES,
        default=None,
        help="Lighting-independent render pass/visualization mode",
    )
    parser.add_argument(
        "--camera-position",
        type=int,
        default=None,
        help=(
            "Index into this scene's alternative camera positions in "
            f"{CAMERA_POSITIONS_JSON} (overrides the .blend file's authored camera pose)"
        ),
    )

    args = parser.parse_args()

    render_perturbed(
        args.blend_path,
        args.out_path,
        perturbations=args.perturbations,
        material_mode=args.material_mode,
        sun_elevation=args.sun_elevation,
        sun_azimuth=args.sun_azimuth,
        shadow_softness=args.shadow_softness,
        gi_bounces=args.gi_bounces,
        exposure=args.exposure,
        view_transform=args.view_transform,
        resolution=args.resolution,
        samples=args.samples,
        denoising=args.denoising,
        render_pass=args.render_pass,
        camera_position=args.camera_position,
    )
