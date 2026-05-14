import os
from enum import Enum

from google import genai
from PIL import Image
from pydantic import BaseModel, Field


class SemanticLabel(Enum):
    FURNITURE = "furniture"
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"
    DECOR = "decor"
    OTHER = "other"


class ObjectBoundingBox(BaseModel):
    id: str = Field("Unique identifier for the object")
    label: SemanticLabel = Field("Semantic label of the object")
    extents: list[float] = Field("Bounding box lenghts along axes (x, y, z)")
    center: list[float] = Field(
        "Center coordinates of the bounding box in world space (x, y, z) in right-handed Y-up coordinate system"
    )
    orientation: list[float] = Field(
        "Orientation of the bounding box as a quaternion (x, y, z, w) in right-handed Y-up coordinate system"
    )


class VLMSceneScore(BaseModel):
    object_count: int
    object_placement: int
    object_scale: int
    object_orientation: int
    reasoning: str


class CoarseScene(BaseModel):
    objects: list[ObjectBoundingBox]


class CoarseGeometryAgent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No API key found.")

        self.client = genai.Client(api_key=api_key)

    def generate_geometry(self, image_path: str, prompt: str) -> CoarseScene:
        image = Image.open(image_path)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_schema": CoarseScene,
            },
        )

        coarse_scene = response.parsed
        return coarse_scene


class CoarseGeometryInspectorAgent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No API key found.")

        self.client = genai.Client(api_key=api_key)

    def generate_score(
        self, target_image_path: str, rendering_image_path: str, prompt: str, scene: CoarseScene = None
    ) -> VLMSceneScore:
        target_image = Image.open(target_image_path)
        rendering_image = Image.open(rendering_image_path)

        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[prompt, target_image, rendering_image],
            config={"response_mime_type": "application/json", "response_schema": VLMSceneScore},
        )

        vlm_score = response.parsed
        return vlm_score


if __name__ == "__main__":
    agent = CoarseGeometryAgent()

    with open("prompts/scene_generator_prompt.txt", "r") as f:
        prompt = f.read()
    scene = agent.generate_geometry("assets/living_room_scene.png", prompt)

    with open("outputs/coarse_scene.json", "w") as f:
        f.write(scene.model_dump_json(indent=2))
