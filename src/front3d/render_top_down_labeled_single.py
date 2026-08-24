import json
import subprocess
import os
from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont

BLENDER_PATH = os.getenv("BLENDER_PATH")
RENDER_SCRIPT = Path(__file__).parents[1] / "blender/render_scene_top_down_labeled.py"

assert RENDER_SCRIPT.exists(), RENDER_SCRIPT

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


# Compass directions tried around each marker, closest ring first.
_LABEL_DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _overlap_area(a, b) -> float:
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy


def _place_label(draw, x, y, text, font, marker_radius, placed_boxes):
    base = marker_radius + 10
    best_pos, best_box, best_overlap = None, None, None
    for ring in range(1, 8):
        r = base * ring
        for dx, dy in _LABEL_DIRECTIONS:
            pos = (x + dx * r, y + dy * r)
            box = draw.textbbox(pos, text, font=font, anchor="mm")
            overlap = sum(_overlap_area(box, placed) for placed in placed_boxes)
            if overlap == 0.0:
                return pos, box
            if best_overlap is None or overlap < best_overlap:
                best_pos, best_box, best_overlap = pos, box, overlap
    return best_pos, best_box


def draw_object_labels(image: Image.Image, objects: list[dict]) -> Image.Image:
    labeled = image.convert("RGB").copy()
    draw = ImageDraw.Draw(labeled)
    font = _load_font(size=max(12, image.width // 80))
    marker_radius = max(3, image.width // 300)

    placed_boxes = []
    for i, obj in enumerate(objects, start=1):
        if not obj["in_view"]:
            continue

        x, y = obj["pixel_x"], obj["pixel_y"]
        draw.ellipse(
            (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
            fill="red",
            outline="white",
        )

        text = f"{i}: {obj['label']}"
        text_pos, box = _place_label(draw, x, y, text, font, marker_radius, placed_boxes)
        placed_boxes.append(box)

        # Drawn before the box/text so it's covered inside the label and only
        # visible as a short segment connecting the dot to the label edge.
        draw.line((x, y, text_pos[0], text_pos[1]), fill="yellow", width=1)

        pad = 2
        draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill="black")
        draw.text(text_pos, text, font=font, fill="white", anchor="mm")

    return labeled


def render_top_down_labeled_single(scene_dir: Path):
    glb_path = scene_dir / "scene.glb"
    metadata_path = scene_dir / "metadata.json"
    raw_path = scene_dir / "color_top_down.png"
    labels_path = scene_dir / "color_top_down_labels.json"
    out_path = scene_dir / "color_top_down_labeled.png"

    cmd = [
        BLENDER_PATH,
        "--background",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(glb_path),
        str(metadata_path),
        str(raw_path),
        str(labels_path),
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Rendering failed: {result.stderr}")

    with open(labels_path) as f:
        labels = json.load(f)

    image = Image.open(raw_path)
    labeled = draw_object_labels(image, labels["objects"])
    labeled.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=Path, help="Source scene directory")
    args = parser.parse_args()

    render_top_down_labeled_single(args.scene_dir)
