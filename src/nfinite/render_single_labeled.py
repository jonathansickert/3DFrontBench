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
OBJECTS_JSON = Path(__file__).parent / "objects.json"

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


def render_labeled(blend_path: str, out_path: str, labels: bool = False):
    cmd = [
        BLENDER_PATH,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(blend_path),
        str(out_path),
    ]

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

    args = parser.parse_args()

    render_labeled(args.blend_path, args.out_path, labels=args.labels)
