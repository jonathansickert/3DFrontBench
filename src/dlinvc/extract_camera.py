import bpy
import json
import numpy as np
import os
from pathlib import Path


def extract_intrinsics(cam, render):
    scale = render.resolution_percentage / 100.0
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)

    sensor_fit = cam.data.sensor_fit
    sensor_width_w_mm = cam.data.sensor_width
    sensor_height_h_mm = cam.data.sensor_height
    f_mm = cam.data.lens

    aspect_ratio = width / height
    if sensor_fit == "AUTO":
        use_horizontal = aspect_ratio >= aspect_ratio
    elif sensor_fit == "HORIZONTAL":
        use_horizontal = True
    else:  # sensor_fit == "VERTICAL"
        use_horizontal = False

    if use_horizontal:
        fx = (f_mm / sensor_width_w_mm) * width
        fy = fx
    else:
        fy = (f_mm / sensor_height_h_mm) * height
        fx = fy

    cx = width / 2 + cam.data.shift_x * width
    cy = height / 2 - cam.data.shift_y * height

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "K": K,
    }


def extract_extrinsics(cam):
    c2w_blender = np.array(cam.matrix_world)
    flip_yz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    c2w_opencv = c2w_blender @ flip_yz

    return c2w_blender, c2w_opencv


def main():
    scene = bpy.context.scene
    render = scene.render
    cam = scene.camera

    intrinsics = extract_intrinsics(cam, render)
    c2w_blender, c2w_opencv = extract_extrinsics(cam)

    znear = cam.data.clip_start
    zfar = cam.data.clip_end

    params = {
        "camera_name": cam.name,
        "camera_type": cam.data.type,
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        # intrinsics
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "aspect_ratio": intrinsics["aspect_ratio"],
        "K": intrinsics["K"],
        # extrinsics
        "c2w_blender": c2w_blender.tolist(),
        "c2w_opencv": c2w_opencv.tolist(),
        # clip
        "znear": znear,
        "zfar": zfar,
    }

    props = bpy.context.window_manager.operator_properties_last("IMPORT_SCENE_OT_gltf")
    if props and props.filepath:
        glb_path = props.filepath
    else:
        raise RuntimeError("No .glb file path found.")

    stem = Path(glb_path).stem
    out_path = f"{stem}_camera.json"

    with open(Path("/Users/jonathansickert/git/DLinVC/dataset") / out_path, "w") as file:
        json.dump(params, file, indent=2)


if __name__ == "__main__":
    main()
