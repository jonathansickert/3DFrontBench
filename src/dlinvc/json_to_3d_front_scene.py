import trimesh
import json
from pathlib import Path
from pydantic import BaseModel
import numpy as np
import trimesh.transformations as tf
import argparse

FRONT_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FRONT")
FUTURE_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FUTURE")
TEXTURE_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FRONT-texture")


def make_transform(pos: list, rot: list, scale: list) -> np.ndarray:
    sx, sy, sz = scale
    S = np.diag([sx, sy, sz, 1.0])

    qx, qy, qz, qw = rot
    R = tf.quaternion_matrix([qw, qx, qy, qz])

    T = tf.translation_matrix(pos)

    return T @ R @ S


class FurnitureMesh(BaseModel):
    uid: str
    jid: str
    label: str
    pos: list[float]
    rot: list[float]
    scale: list[float]

    def to_mesh(self, bounding_box: bool = True) -> trimesh.Trimesh:
        mesh = trimesh.load(FUTURE_PATH / self.jid / "raw_model.obj")
        mesh.apply_transform(make_transform(pos=self.pos, rot=self.rot, scale=self.scale))

        if bounding_box:
            mesh = mesh.bounding_primitive
        
        return mesh

    def get_name(self) -> str:
        return f"{self.label}_{self.jid}"


class LayoutMesh(BaseModel):
    uid: str
    type: str
    xyz: list[float]
    normal: list[float]
    faces: list[float]
    uv: list[float]

    material_uid: str
    material_jid: str
    color: list[float]
    use_color: bool
    normal_uv_transform: list[float] | None = None
    seam_width: float | None = None
    uv_transform: list[float] | None = None

    def to_mesh(self) -> trimesh.Trimesh:
        verts = np.array(self.xyz).reshape(-1, 3)
        faces = np.array(self.faces).reshape(-1, 3)
        mesh = trimesh.Trimesh(verts, faces)
        mesh.fix_normals()

        # material_path = Path(TEXTURE_PATH) / self.material_jid
        # if (not self.use_color) and material_path.exists():
        #     uv = np.array(self.uv, dtype=np.float32).reshape(-1, 2)
        #     if self.uv_transform is not None:
        #         m = np.array(self.uv_transform).reshape(3, 3)
        #         uv_h = np.c_[uv, np.ones(len(uv))]
        #         uv = (uv_h @ m.T)[:, :2]

        #     mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.open(material_path / "texture.png"))

        # else:
        color = np.array(self.color)
        mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))

        return mesh

    def get_name(self) -> str:
        return f"{self.type}_{self.uid}"


def load_furniture_objects_for_room(scene_json: dict, room: dict) -> list[FurnitureMesh]:
    furniture_by_uid = {f["uid"]: f for f in scene_json["furniture"]}
    furniture_children = [c for c in room["children"] if "furniture" in c["instanceid"]]

    furnitures = []
    for obj in furniture_children:
        furniture = furniture_by_uid.get(obj["ref"])

        if furniture is None:
            continue

        if not furniture.get("valid", True):
            continue

        jid = furniture["jid"]
        future_path = FUTURE_PATH / jid
        if not future_path.exists():
            continue

        label = furniture.get("title") or furniture.get("category")
        if label is None:
            print(obj)
            print("no label")

        f = FurnitureMesh(
            uid=furniture["uid"], jid=jid, label=label, pos=obj["pos"], rot=obj["rot"], scale=obj["scale"]
        )

        furnitures.append(f)

    return furnitures


def load_mesh_objects_for_room(scene_json: dict, room: dict) -> list[LayoutMesh]:
    mesh_by_uid = {m["uid"]: m for m in scene_json["mesh"]}
    material_by_uid = {m["uid"]: m for m in scene_json["material"]}

    mesh_children = [c for c in room["children"] if "mesh" in c["instanceid"]]

    meshes = []
    for obj in mesh_children:
        mesh = mesh_by_uid.get(obj["ref"])

        if mesh is None:
            continue

        if not mesh.get("valid", True):
            continue

        material = material_by_uid[mesh["material"]]

        m = LayoutMesh(
            uid=mesh["uid"],
            type=mesh["type"],
            xyz=mesh["xyz"],
            normal=mesh["normal"],
            faces=mesh["faces"],
            uv=mesh["uv"],
            material_uid=material["uid"],
            material_jid=material["jid"],
            color=material["color"],
            seam_width=material.get("seamWidth"),
            use_color=material.get("useColor", True),
            normal_uv_transform=material.get("normalUVTransform"),
            uv_transform=material.get("UVTransform"),
        )

        meshes.append(m)

    return meshes


def build_room_scene(scene_json: dict, room: dict, bounding_box: bool) -> trimesh.Scene:
    meshes = load_mesh_objects_for_room(scene_json, room)
    furnitures = load_furniture_objects_for_room(scene_json, room)

    if len(furnitures) == 0:
        print("Room does not have any furniture objects")
        return None

    scene = trimesh.Scene()
    for mesh in meshes:
        m = mesh.to_mesh()
        node_name = mesh.get_name()
        scene.add_geometry(m, node_name=node_name)

    for mesh in furnitures:
        m = mesh.to_mesh(bounding_box=bounding_box)
        node_name = mesh.get_name()
        scene.add_geometry(m, node_name=node_name)

    return scene

def print_room_ids(scene_json: dict):
    rooms = scene_json["scene"]["room"]
    for i, room in enumerate(rooms):
        print(f"{i}: {room['instanceid']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_rooms", action="store_true")
    parser.add_argument("scene_json", type=str)
    parser.add_argument("room_idx", nargs="?", type=int, default=None)
    args = parser.parse_args()

    with open(args.scene_json) as f:
        scene_json = json.load(f)

    if args.print_rooms:
        print_room_ids(scene_json=scene_json)
        
        if args.room_idx is None:
            exit()

    if args.room_idx is None:
        parser.error("Room index required. Use --print_rooms to get the available rooms.")

    room = scene_json["scene"]["room"][args.room_idx]

    scene_normal =  build_room_scene(scene_json=scene_json, room=room, bounding_box=False)
    if scene_normal is not None:
        scene_normal.export(f"./dataset/{room['instanceid']}.glb")

    scene_bbox = build_room_scene(scene_json=scene_json, room=room, bounding_box=False)
    if scene_bbox is not None:
        scene_bbox.export(f"./dataset/{room['instanceid']}_bbox.glb")



    
