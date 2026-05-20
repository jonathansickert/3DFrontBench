"""Convert a single 3D-FRONT room to a .glb mesh.

3D-FRONT scene JSON → trimesh.Scene → .glb

Usage:
    python -m dlinvc.three_d_front_to_glb \
        --scene  ../3D-FRONT/3D-FRONT/<uuid>.json \
        --future ../3D-FRONT/3D-FUTURE/ \
        --output outputs/room.glb \
        [--room-index 0] \
        [--room-type Bedroom] \
        [--list-rooms]

JSON layout (top level):
  material[]: {uid, color[R,G,B,A 0-255], texture url, jid→texture folder}
  mesh[]:     {uid, type, material uid, xyz flat[float], faces flat[int],
               normal flat[float], uv flat[float|null]}
  furniture[]: {uid, jid, title, category?, valid?}
  scene.room[]:
    {type, instanceid, pos, rot, scale,
     children[]:
       furniture children → {instanceid:'furniture/N', ref→furniture.uid,
                              pos, rot[x,y,z,w], scale}
       mesh children      → {instanceid:'mesh/N',      ref→mesh.uid,
                              pos, rot[x,y,z,w], scale}}

Architecture:
  Structural elements (Floor, WallInner, WallOuter, WallTop, WallBottom,
  Ceiling, CustomizedCeiling, Door, Window, Baseboard, Cabinet, …) live
  in the top-level mesh[] and are referenced by a room's children.
  Use the room's mesh-children to get only that room's surfaces.

Coordinate system:
  3D-FRONT is Y-up, same as the GLB standard → no axis swap needed.

Quaternion convention:
  3D-FRONT stores quaternions as [x, y, z, w] (OpenGL / glTF convention).
  [0, 0, 0, 1] = identity.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tf
from trimesh.visual.material import SimpleMaterial


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _flat_to_verts(flat) -> np.ndarray:
    return np.array(flat, dtype=np.float64).reshape(-1, 3)


def _flat_to_faces(flat) -> np.ndarray:
    return np.array(flat, dtype=np.int64).reshape(-1, 3)


def _flat_to_uv(flat) -> np.ndarray | None:
    """Convert flat UV list (may contain None) to (N,2), or None if empty."""
    clean = [float(v) if v is not None else 0.0 for v in flat]
    if not clean:
        return None
    arr = np.array(clean, dtype=np.float64)
    if arr.size % 2 != 0:
        return None
    return arr.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def make_transform(pos: list, rot: list, scale: list) -> np.ndarray:
    """Build a 4×4 matrix from a 3D-FRONT child pose.

    pos   – [x, y, z]
    rot   – [x, y, z, w]  quaternion (OpenGL / glTF convention)
    scale – [x, y, z]
    """
    sx, sy, sz = scale
    S = np.diag([sx, sy, sz, 1.0])

    qx, qy, qz, qw = rot                              # unpack xyzw
    R = tf.quaternion_matrix([qw, qx, qy, qz])        # trimesh/scipy wants wxyz

    T = tf.translation_matrix(pos)

    return T @ R @ S   # scale → rotate → translate


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def _build_material_lookup(data: dict) -> dict[str, dict]:
    return {m["uid"]: m for m in data.get("material", [])}


def _make_simple_material(mat_dict: dict | None) -> SimpleMaterial:
    """Convert a 3D-FRONT material dict to a trimesh SimpleMaterial.

    color is [R, G, B, A] in 0-255 range.
    Falls back to light grey when no material is supplied.
    """
    if mat_dict is None:
        return SimpleMaterial(diffuse=[200, 200, 200, 255])
    color = mat_dict.get("color") or [200, 200, 200, 255]
    if len(color) == 3:
        color = list(color) + [255]
    return SimpleMaterial(diffuse=[int(c) for c in color])


# ---------------------------------------------------------------------------
# Mesh building
# ---------------------------------------------------------------------------

# Surface types to include by default per flag
_FLOOR_TYPES   = {"floor", "invisiblefloor", "virtualfloor"}
_WALL_TYPES    = {"wallinner", "wallouter", "walltop", "wallbottom",
                  "customizedfeaturewall"}
_CEILING_TYPES = {"ceiling", "customizedceiling", "slabtop"}
_DOOR_TYPES    = {"door"}
_WINDOW_TYPES  = {"window"}
# Everything else (Baseboard, Cabinet, Hole, Pocket, …) is included unless
# explicitly filtered out.


def _parse_mesh(mesh_data: dict, mat_lookup: dict) -> trimesh.Trimesh | None:
    xyz = mesh_data.get("xyz")
    faces = mesh_data.get("faces")
    if not xyz or not faces:
        return None

    verts = _flat_to_verts(xyz)
    tris = _flat_to_faces(faces)

    normals_raw = mesh_data.get("normal")
    vertex_normals = _flat_to_verts(normals_raw) if normals_raw else None

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=tris,
        vertex_normals=vertex_normals,
        process=False,
    )

    mat_uid = mesh_data.get("material")
    mat_dict = mat_lookup.get(mat_uid) if mat_uid else None
    tm_mat = _make_simple_material(mat_dict)

    uv_raw = mesh_data.get("uv")
    uv = _flat_to_uv(uv_raw) if uv_raw else None
    if uv is not None and len(uv) == len(verts):
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=tm_mat)
    else:
        mesh.visual.face_colors = tm_mat.diffuse

    return mesh


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_room_scene(
    data: dict,
    future_dir: Path,
    room_index: int,
    include_floors: bool = True,
    include_walls: bool = True,
    include_ceilings: bool = True,
    include_doors: bool = True,
    include_windows: bool = True,
    include_other_meshes: bool = True,
) -> trimesh.Scene:
    """Assemble a trimesh.Scene for one 3D-FRONT room.

    Structural elements are resolved by following the room's mesh-children
    (instanceid 'mesh/…') which reference the top-level mesh[] by uid.
    Furniture is loaded from 3D-FUTURE OBJ files.
    """
    rooms = data["scene"]["room"]
    if room_index >= len(rooms):
        raise IndexError(f"room_index {room_index} out of range ({len(rooms)} rooms)")

    room = rooms[room_index]
    room_type = room.get("type", "unknown")
    room_iid = room.get("instanceid", "?")
    print(f"Room [{room_index}]  type={room_type}  instanceid={room_iid}")

    # Build lookup tables
    mat_lookup = _build_material_lookup(data)
    mesh_by_uid: dict[str, dict] = {m["uid"]: m for m in data.get("mesh", [])}
    furniture_by_uid: dict[str, dict] = {f["uid"]: f for f in data.get("furniture", [])}

    out = trimesh.Scene()
    children = room.get("children", [])

    # ── structural meshes (floors, walls, ceilings, doors, windows, …) ──────
    mesh_children = [c for c in children if "mesh" in c.get("instanceid", "")]
    print(f"  mesh children: {len(mesh_children)}")

    for child in mesh_children:
        ref = child.get("ref", "")
        mesh_data = mesh_by_uid.get(ref)
        if mesh_data is None:
            continue

        mtype = mesh_data.get("type", "").strip()
        mtype_lower = mtype.lower()

        # Apply surface-type filter
        if mtype_lower in _FLOOR_TYPES and not include_floors:
            continue
        if mtype_lower in _WALL_TYPES and not include_walls:
            continue
        if mtype_lower in _CEILING_TYPES and not include_ceilings:
            continue
        if mtype_lower in _DOOR_TYPES and not include_doors:
            continue
        if mtype_lower in _WINDOW_TYPES and not include_windows:
            continue
        is_known = (mtype_lower in _FLOOR_TYPES | _WALL_TYPES |
                    _CEILING_TYPES | _DOOR_TYPES | _WINDOW_TYPES)
        if not is_known and not include_other_meshes:
            continue

        mesh = _parse_mesh(mesh_data, mat_lookup)
        if mesh is None:
            continue

        pos   = child.get("pos",   [0.0, 0.0, 0.0])
        rot   = child.get("rot",   [0.0, 0.0, 0.0, 1.0])  # xyzw; [0,0,0,1]=identity
        scale = child.get("scale", [1.0, 1.0, 1.0])
        mesh.apply_transform(make_transform(pos, rot, scale))

        node_name = f"{mtype}__{child.get('instanceid', ref)}"
        out.add_geometry(mesh, node_name=node_name)

    # ── furniture instances ───────────────────────────────────────────────
    furn_children = [c for c in children if "furniture" in c.get("instanceid", "")]
    print(f"  furniture children: {len(furn_children)}")

    for child in furn_children:
        ref = child.get("ref", "")
        furniture = furniture_by_uid.get(ref)
        if furniture is None:
            print(f"    [warn] ref '{ref}' not in furniture catalogue")
            continue
        if not furniture.get("valid", True):
            continue

        jid = furniture.get("jid", "")
        if not jid:
            continue

        label = furniture.get("category") or furniture.get("title") or jid
        if "/" in label:
            label = label.split("/")[0]

        print(f"  + {label}  ({jid})")

        obj_path = future_dir / jid / "raw_model.obj"
        if not obj_path.exists():
            print(f"    [skip] model not found: {obj_path}")
            continue
        try:
            loaded = trimesh.load(str(obj_path), force="mesh", process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = loaded.dump(concatenate=True)
            if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
                print(f"    [skip] empty mesh: {obj_path}")
                continue
        except Exception as exc:
            print(f"    [warn] failed to load {obj_path}: {exc}")
            continue

        pos   = child.get("pos",   [0.0, 0.0, 0.0])
        rot   = child.get("rot",   [0.0, 0.0, 0.0, 1.0])  # xyzw
        scale = child.get("scale", [1.0, 1.0, 1.0])
        loaded.apply_transform(make_transform(pos, rot, scale))

        node_name = f"{label}__{child.get('instanceid', ref)}"
        out.add_geometry(loaded, node_name=node_name)

    return out


# ---------------------------------------------------------------------------
# Room discovery helpers
# ---------------------------------------------------------------------------

def find_room_index(data: dict, room_type: str) -> int | None:
    for i, room in enumerate(data["scene"]["room"]):
        if room.get("type", "").lower() == room_type.lower():
            return i
    return None


def list_rooms(data: dict) -> None:
    mesh_by_uid = {m["uid"]: m for m in data.get("mesh", [])}

    print(f"{'#':<4}  {'type':<26}  {'instanceid':<30}  mesh   furn")
    for i, room in enumerate(data["scene"]["room"]):
        children = room.get("children", [])
        n_mesh = sum(1 for c in children if "mesh"      in c.get("instanceid", ""))
        n_furn = sum(1 for c in children if "furniture" in c.get("instanceid", ""))

        # Summarise referenced mesh types
        mtypes = {}
        for c in children:
            if "mesh" in c.get("instanceid", ""):
                md = mesh_by_uid.get(c.get("ref", ""))
                if md:
                    t = md.get("type", "?")
                    mtypes[t] = mtypes.get(t, 0) + 1

        print(f"{i:<4}  {room.get('type','?'):<26}  "
              f"{room.get('instanceid','?'):<30}  {n_mesh:<6} {n_furn}")
        if mtypes:
            for t, cnt in sorted(mtypes.items()):
                print(f"       mesh type {t!r} x{cnt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a single 3D-FRONT room to a .glb mesh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scene",  required=True, help="3D-FRONT scene JSON path")
    parser.add_argument("--future", required=True, help="3D-FUTURE model root directory")
    parser.add_argument("--output", default=None,  help="Output .glb path")
    parser.add_argument("--room-index", type=int, default=0, metavar="N",
                        help="Room index to export (default: 0)")
    parser.add_argument("--room-type", default=None,
                        help="Export first room of this type, e.g. Bedroom, LivingRoom")
    parser.add_argument("--list-rooms", action="store_true",
                        help="Print all rooms and exit (no --output needed)")
    # surface toggles
    parser.add_argument("--no-floors",    action="store_true")
    parser.add_argument("--no-walls",     action="store_true")
    parser.add_argument("--no-ceilings",  action="store_true")
    parser.add_argument("--no-doors",     action="store_true")
    parser.add_argument("--no-windows",   action="store_true")
    parser.add_argument("--no-other-meshes", action="store_true",
                        help="Skip Baseboard, Cabinet, Pocket, Hole, … meshes")
    args = parser.parse_args()

    data = _load_json(args.scene)

    if args.list_rooms:
        list_rooms(data)
        return

    if not args.output:
        parser.error("--output is required unless --list-rooms is set")

    room_index = args.room_index
    if args.room_type:
        found = find_room_index(data, args.room_type)
        if found is not None:
            room_index = found
            print(f"Found '{args.room_type}' at index {room_index}")
        else:
            available = [r.get("type") for r in data["scene"]["room"]]
            print(f"[warn] '{args.room_type}' not found. Available: {available}")

    scene = build_room_scene(
        data=data,
        future_dir=Path(args.future),
        room_index=room_index,
        include_floors=not args.no_floors,
        include_walls=not args.no_walls,
        include_ceilings=not args.no_ceilings,
        include_doors=not args.no_doors,
        include_windows=not args.no_windows,
        include_other_meshes=not args.no_other_meshes,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
