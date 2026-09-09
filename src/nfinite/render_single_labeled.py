import json
import subprocess
import os
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).parent / "render_nfinite_labeled.py"
OBJECTS_JSON =  Path("/home/jonathansickert/git/3DFrontBench/assets/objects.json")
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

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# render_nfinite_labeled.py prints label data as a single line prefixed with
# this marker so it can be pulled out of Blender's stdout without ever
# touching disk -- the rendered PNG is the only thing this pipeline persists.
LABELS_MARKER = "NFINITE_LABELS_JSON:"


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _rle_decode(rle: list, length: int) -> np.ndarray:
    flat = np.empty(length, dtype=np.int64)
    pos = 0
    for value, count in rle:
        flat[pos : pos + count] = value
        pos += count
    return flat


def _pole_of_inaccessibility(mask: np.ndarray) -> tuple[int, int]:
    # The point maximally distant from the mask's boundary: guaranteed to
    # land inside the visible silhouette, even for L-shaped or occluded
    # objects where the centroid can fall outside it (or onto another object
    # entirely).
    dist = ndimage.distance_transform_edt(mask)
    max_dist = dist.max()
    candidates = np.argwhere(dist == max_dist)
    if len(candidates) == 1:
        y, x = candidates[0]
        return int(x), int(y)

    labeled, _ = ndimage.label(mask)
    component_sizes = np.bincount(labeled.ravel())
    y, x = max(candidates, key=lambda c: component_sizes[labeled[c[0], c[1]]])
    return int(x), int(y)


def _clamp_into_bounds(box, pad: int, width: int, height: int) -> tuple[float, float]:
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    dx = -x0 if x0 < 0 else min(0.0, width - x1)
    dy = -y0 if y0 < 0 else min(0.0, height - y1)
    return dx, dy


def draw_object_labels(image: Image.Image, index_map: np.ndarray, entries: list[dict]) -> Image.Image:
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    font = _load_font(size=max(20, image.width // 60))
    pad = 5

    for entry in entries:
        mask = index_map == entry["mask_id"]
        if not mask.any():
            continue

        x, y = _pole_of_inaccessibility(mask)
        text = str(entry["index"])

        box = draw.textbbox((x, y), text, font=font, anchor="mm")
        dx, dy = _clamp_into_bounds(box, pad, image.width, image.height)
        x, y = x + dx, y + dy
        box = draw.textbbox((x, y), text, font=font, anchor="mm")

        draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill="black", outline="white")
        draw.text((x, y), text, font=font, fill="white", anchor="mm")

    return labeled


def render_labeled(
    blend_path: str,
    out_path: str,
    labels: bool = False,
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
):
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
    if labels:
        cmd += ["--labels", str(OBJECTS_JSON)]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Rendering failed: {result.stderr}")

    if labels:
        label_line = next(
            line for line in result.stdout.splitlines() if line.startswith(LABELS_MARKER)
        )
        payload = json.loads(label_line[len(LABELS_MARKER) :])
        width, height = payload["resolution_x"], payload["resolution_y"]
        index_map = _rle_decode(payload["rle"], width * height).reshape(height, width)

        image = Image.open(out_path)
        labeled = draw_object_labels(image, index_map, payload["entries"])
        labeled.save(out_path)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("blend_path", type=Path, help="Source scene")
    parser.add_argument("out_path", type=Path)
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Overlay a numbered digit over each object listed in objects.json",
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
        help="Cycles sample count (default: 32, set by render_nfinite_labeled.py)",
    )
    parser.add_argument(
        "--denoising",
        type=lambda value: value.lower() in ("1", "true", "yes", "on"),
        default=None,
        help="Toggle Cycles' denoiser (default: on, set by render_nfinite_labeled.py)",
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

    render_labeled(
        args.blend_path,
        args.out_path,
        labels=args.labels,
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
