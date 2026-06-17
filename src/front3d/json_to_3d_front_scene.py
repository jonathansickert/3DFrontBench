"""
Convert 3D-FRONT JSON scene data into 3D mesh representations.

This module processes 3D-FRONT dataset JSON files to extract and convert room scenes into
3D geometry. It provides:

- Data models: `FurnitureMesh` and `LayoutMesh` for representing furniture objects and
  structural layout elements respectively
- Scene loading: Functions to extract furniture and mesh objects from a room specification
- Scene building: Assembles complete room scenes combining layout and furniture into
  trimesh.Scene objects for further processing or export
- CLI interface: Script mode allows exporting individual rooms to GLB files

The module handles coordinate transformations, material properties, and geometry assembly
while validating object availability in the 3D-FUTURE furniture library.
"""

import trimesh
from pathlib import Path
from pydantic import BaseModel
import numpy as np
from src.util import make_transform
import uuid
import re
import random
from PIL import Image

FRONT_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FRONT")
FUTURE_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FUTURE")
TEXTURE_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/3D-FRONT-texture")
CCTEXTURES_PATH = Path("/home/jonathansickert/git/DLinVC/3D-FRONT/cctextures")




def sample_random_material():
    probably_useful_texture = ["paving stones", "tiles", "wood", "fabric", "bricks", "metal", "wood floor",
                               "ground", "rock", "concrete", "leather", "planks", "rocks", "gravel",
                               "asphalt", "painted metal", "painted plaster", "marble", "carpet",
                               "plastic", "roofing tiles", "bark", "metal plates", "wood siding",
                               "terrazzo", "plaster", "paint", "corrugated steel", "painted wood",
                               "lava cardboard", "clay", "diamond plate", "ice", "moss", "pipe", "candy",
                               "chipboard", "rope", "sponge", "tactile paving", "paper", "cork",
                               "wood chips"]

    prefixes = tuple(t.replace(" ", "") for t in probably_useful_texture)
    candidates = [
        d for d in CCTEXTURES_PATH.iterdir()
        if d.name.lower().startswith(prefixes)
    ]
    random.shuffle(candidates)

    for texture_dir in candidates:
        color_path = texture_dir / f"{texture_dir.name}_2K-JPG_Color.jpg"
        metallic_path = texture_dir / f"{texture_dir.name}_2K-JPG_Metalness.jpg"
        roughness_path = texture_dir / f"{texture_dir.name}_2K-JPG_Roughness.jpg"
        normal_path = texture_dir / f"{texture_dir.name}_2K-JPG_NormalGL.jpg"

        if not all(p.exists() for p in (color_path, metallic_path, roughness_path, normal_path)):
            continue

        return {
            "color": Image.open(color_path),
            "metallic": Image.open(metallic_path),
            "roughness": Image.open(roughness_path),
            "normal": Image.open(normal_path),
        }

    raise RuntimeError(f"No complete CCTextures material found under {CCTEXTURES_PATH}")



class FurnitureMesh(BaseModel):
    uid: str
    jid: str
    pos: list[float]
    rot: list[float]
    scale: list[float]
    label: str | None

    def to_mesh(self, bounding_box: bool = False) -> trimesh.Trimesh:
        mesh = trimesh.load(FUTURE_PATH / self.jid / "raw_model.obj")

        if bounding_box:
            mesh = mesh.bounding_primitive

        return mesh

    def get_name(self) -> str:
        key = f"{self.uid}_{self.jid}_{self.pos}_{self.rot}_{self.scale}"
        furniture_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, key)
        clean_label = re.sub(r"[^A-Za-z_]", "_", self.label)
        clean_label = re.sub(r"_+", "_", clean_label).strip("_")

        return f"{clean_label}_{furniture_uuid}"

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

    def to_mesh(self, texture: dict) -> trimesh.Trimesh:
        verts = np.array(self.xyz).reshape(-1, 3)
        normals = np.array(self.normal).reshape(-1, 3)
        faces = np.array(self.faces, dtype=np.int32).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

        uv = np.array(self.uv).reshape(-1, 2)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=texture["color"],
            metallicRoughnessTexture=texture["roughness"],
            doubleSided=True,
        )

        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uv, material=material,
        )
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


def build_room_scene(
    scene_json: dict, room: dict, bounding_box: bool
) -> tuple[trimesh.Scene, trimesh.Scene, list[dict], bool]:
    meshes = load_mesh_objects_for_room(scene_json, room)
    furnitures = load_furniture_objects_for_room(scene_json, room)

    if len(furnitures) == 0:
        print("Room does not have any furniture objects")
        return None, None, [], False

    texture_by_type = {}
    for mesh in meshes:
        if mesh.type not in texture_by_type:
            texture_by_type[mesh.type] = sample_random_material()

    scene = trimesh.Scene()
    for mesh in meshes:
        m = mesh.to_mesh(texture_by_type[mesh.type])
        node_name = mesh.get_name()
        scene.add_geometry(m, node_name=node_name)

    scaffold = scene.copy()

    furniture_list = []

    is_valid = True

    for mesh in furnitures:
        m = mesh.to_mesh(bounding_box=bounding_box)
        node_name = mesh.get_name()
        transform = make_transform(pos=mesh.pos, rot=mesh.rot, scale=mesh.scale)
        scene.add_geometry(m, node_name=node_name, transform=transform)

        if mesh.label is None:
            is_valid = False

        furniture_list.append(
            {
                "name": mesh.get_name(),
                "uid": mesh.uid,
                "jid": mesh.jid,
                "label": mesh.label,
                "pos": mesh.pos,
                "rot": mesh.rot,
                "scale": mesh.scale,
            }
        )

    return scene, scaffold, furniture_list, is_valid
