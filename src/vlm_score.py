import os

from google import genai
from PIL import Image
from pydantic import BaseModel


class VLMSceneScore(BaseModel):
    object_count: int
    object_placement: int
    object_scale: int
    object_orientation: int
    reasoning: str


class VLMScoreAgent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("No API key found.")

        self.client = genai.Client(api_key=api_key)

    def generate_score(
        self,
        target_image: Image.Image,
        target_image_depth: Image.Image,
        rendering_image: Image.Image,
        rendering_image_depth: Image.Image,
        prompt: str,
    ) -> VLMSceneScore:
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[
                prompt,
                target_image,
                target_image_depth,
                rendering_image,
                rendering_image_depth,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": VLMSceneScore,
            },
        )

        vlm_score = response.parsed
        return vlm_score


def compute_vlm_score(path1: str, path2: str) -> dict[str, float]:
    vlm_agent = VLMSceneScore()

    img = Image.open(path1)
    img = Image.open(path2)
