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
        rendering_image: Image.Image,
        prompt: str,
    ) -> VLMSceneScore:
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[
                prompt,
                target_image,
                rendering_image,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": VLMSceneScore,
            },
        )

        vlm_score = response.parsed
        return vlm_score



def compute_vlm_score(target_path: str, render_path: str, runs: int = 1) -> list[dict[str, float]]:
    vlm_agent = VLMScoreAgent()
    with open("/Users/jonathansickert/git/3DFrontBench/prompts/vlm_score_prompt.txt") as f:
        prompt = f.read()

    target_img = Image.open(target_path)
    render_img = Image.open(render_path)

    results = []
    for _ in range(runs):
        result = vlm_agent.generate_score(
            target_image=target_img,
            rendering_image=render_img,
            prompt=prompt,
        )

        results.append(
            result.model_dump_json()
        )

    return results
