"""Convert a single 3D-FRONT room to a .glb mesh.

3D-FRONT scene JSON → trimesh.Scene → .glb

Usage:
    python -m dlinvc.three_d_front_to_glb \
        --scene  ../3D-FRONT/3D-FRONT/<uuid>.json \
        --future ../3D-FRONT/3D-FUTURE/ \
        --output outputs/room.glb \
        [--room-index 0] \
        [--room-type Bedroom]

Scene JSON layout (relevant fields):
  furniture[]: {uid, jid, title, type, valid}   — model catalogue for this scene
  scene.room[]: {id, type, height, mesh, children[]}
    mesh:      {xyz: flat[float], faces: flat[int], normal: flat[float]}
    children[]: {ref→furniture.uid, pos[3], rot[4 xyzw], scale[3]}

3D-FUTURE models live at  <future_dir>/<jid>/raw_model.obj
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tf


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _flat_to_verts(flat: list[float]) -> np.ndarray:
    """Reshape a flat [x0,y0,z0, x1,y1,z1, …] list into (N,3)."""
    return np.array(flat, dtype=np.float64).reshape(-1, 3)


def _flat_to_faces(flat: list[int]) -> np.ndarray:
    """Reshape a flat [i0,j0,k0, …] list into (F,3)."""
    return np.array(flat, dtype=np.int64).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Room structure mesh
# ---------------------------------------------------------------------------

def build_room_mesh(mesh_data: dict) -> trimesh.Trimesh | None:
    """Build a Trimesh from a 3D-FRONT room mesh dict.

    The dict has keys 'xyz', 'faces', and optionally 'normal'.
    'xyz' and 'normal' are flat float arrays; 'faces' is a flat int array.
    """
    xyz = mesh_data.get("xyz")
    faces = mesh_data.get("faces")
    if not xyz or not faces:
        return None

    verts = _flat_to_verts(xyz)
    tris = _flat_to_faces(faces)

    kwargs: dict = dict(vertices=verts, faces=tris, process=False)

    normals_raw = mesh_data.get("normal")
    if normals_raw:
        kwargs["vertex_normals"] = _flat_to_verts(normals_raw)

    return trimesh.Trimesh(**kwargs)


# ---------------------------------------------------------------------------
# Furniture mesh loading
# ---------------------------------------------------------------------------

def load_furniture_mesh(future_dir: Path, jid: str) -> trimesh.Trimesh | None:
    """Load the raw_model.obj for a 3D-FUTURE model.

    Returns a single Trimesh (multiple submeshes are concatenated) or None if
    the file is missing or fails to load.
    """
    obj_path = future_dir / jid / "raw_model.obj"
    if not obj_path.exists():
        print(f"    [skip] model not found: {obj_path}")
        return None

    try:
        loaded = trimesh.load(
            str(obj_path),
            force="mesh",
            process=False,
        )
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            print(f"    [skip] empty mesh: {obj_path}")
            return None
        return loaded
    except Exception as exc:
        print(f"    [warn] failed to load {obj_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def make_transform(pos: list, rot: list, scale: list) -> np.ndarray:
    """Build a 4x4 transform matrix from 3D-FRONT child pose fields.

    pos   — [x, y, z] translation
    rot   — [x, y, z, w] quaternion  (3D-FRONT convention)
    scale — [x, y, z] non-uniform scale
    """
    # Scale matrix
    sx, sy, sz = scale
    S = np.diag([sx, sy, sz, 1.0])

    # Rotation matrix; trimesh/scipy use [w, x, y, z]
    qx, qy, qz, qw = rot
    R = tf.quaternion_matrix([qw, qx, qy, qz])  # 4x4

    # Translation matrix
    T = tf.translation_matrix(pos)  # 4x4

    # T * R * S  (scale first, then rotate, then translate)
    return T @ R @ S


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_room_scene(
    scene_json: dict,
    future_dir: Path,
    room_index: int,
) -> trimesh.Scene:
    """Assemble a trimesh.Scene for one 3D-FRONT room.

    Args:
        scene_json:  Parsed 3D-FRONT JSON.
        future_dir:  Root directory of 3D-FUTURE models.
        room_index:  Which room in scene.room[] to export.

    Returns:
        A trimesh.Scene containing the room structure and all furniture.
    """
    rooms = scene_json["scene"]["room"]
    if room_index >= len(rooms):
        raise IndexError(f"room_index {room_index} out of range (scene has {len(rooms)} rooms)")

    room = rooms[room_index]
    room_type = room.get("type", "unknown")
    room_id = room.get("id", "?")
    print(f"Room [{room_index}]  type={room_type}  id={room_id}")

    # Build furniture uid → definition lookup
    furniture_by_uid: dict[str, dict] = {
        f["uid"]: f for f in scene_json.get("furniture", [])
    }

    out_scene = trimesh.Scene()

    # ---- room structure mesh (floor + walls if encoded) --------------------
    mesh_data = room.get("mesh")
    if mesh_data:
        room_mesh = build_room_mesh(mesh_data)
        if room_mesh is not None:
            out_scene.add_geometry(room_mesh, node_name="room_structure")
            print(f"  room mesh: {len(room_mesh.vertices)} verts, {len(room_mesh.faces)} faces")

    # ---- furniture instances -----------------------------------------------
    children = room.get("children", [])
    print(f"  children: {len(children)}")

    for child in children:
        ref = child.get("ref")
        if ref is None:
            continue

        furniture = furniture_by_uid.get(ref)
        if furniture is None:
            print(f"    [warn] ref '{ref}' not found in furniture catalogue")
            continue

        if not furniture.get("valid", True):
            continue

        jid = furniture.get("jid")
        if not jid:
            continue

        title = furniture.get("title", jid)
        print(f"  + {title}  ({jid})")

        mesh = load_furniture_mesh(future_dir, jid)
        if mesh is None:
            continue

        pos = child.get("pos", [0.0, 0.0, 0.0])
        rot = child.get("rot", [0.0, 0.0, 0.0, 1.0])
        scale = child.get("scale", [1.0, 1.0, 1.0])

        transform = make_transform(pos, rot, scale)
        mesh.apply_transform(transform)

        instance_id = child.get("instanceid", ref)
        node_name = f"{title}__{instance_id}"
        out_scene.add_geometry(mesh, node_name=node_name)

    return out_scene


# ---------------------------------------------------------------------------
# Room discovery helpers
# ---------------------------------------------------------------------------

def find_room_index(scene_json: dict, room_type: str) -> int | None:
    """Return the first room index whose type matches (case-insensitive)."""
    for i, room in enumerate(scene_json["scene"]["room"]):
        if room.get("type", "").lower() == room_type.lower():
            return i
    return None


def list_rooms(scene_json: dict) -> None:
    for i, room in enumerate(scene_json["scene"]["room"]):
        n = len(room.get("children", []))
        print(f"  [{i}]  type={room.get('type','?')!r:<20} id={room.get('id','?')}  children={n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a single 3D-FRONT room to a .glb mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scene", required=True, help="Path to 3D-FRONT scene JSON file")
    parser.add_argument("--future", required=True, help="Root directory of 3D-FUTURE models")
    parser.add_argument("--output", required=True, help="Output .glb path")
    parser.add_argument("--room-index", type=int, default=0,
                        help="Index of the room to export (default: 0)")
    parser.add_argument("--room-type", default=None,
                        help="Export the first room of this type (e.g. Bedroom, LivingRoom). "
                             "Overrides --room-index if found.")
    parser.add_argument("--list-rooms", action="store_true",
                        help="Print all rooms in the scene and exit.")
    args = parser.parse_args()

    scene_json = _load_json(args.scene)

    if args.list_rooms:
        list_rooms(scene_json)
        return

    room_index = args.room_index
    if args.room_type:
        found = find_room_index(scene_json, args.room_type)
        if found is not None:
            room_index = found
            print(f"Using room index {room_index} for type '{args.room_type}'")
        else:
            available = [r.get("type") for r in scene_json["scene"]["room"]]
            print(f"[warn] room type '{args.room_type}' not found. Available: {available}")
            print(f"Falling back to --room-index {room_index}")

    future_dir = Path(args.future)
    scene = build_room_scene(scene_json, future_dir, room_index)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
