import base64
import io
import os

import openai
from PIL import Image
from pydantic import BaseModel
import time


class VLMSceneScore(BaseModel):
    object_count: int
    object_placement: int
    object_scale: int
    object_orientation: int
    reasoning: str

class VLMPlacementScore(BaseModel):
    object_placement: int
    reasoning: str
    missing_info: str

class VLMScaleScore(BaseModel):
    object_scale: int
    reasoning: str
    missing_info: str

class VLMOrientationScore(BaseModel):
    object_orientation: int
    reasoning: str
    missing_info: str

class VLMCountScore(BaseModel):
    object_count: int
    reasoning: str
    missing_info: str

def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VLMScoreAgent:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        base_url = base_url or os.getenv("GEMINI_BASE_URL")

        if not api_key:
            raise ValueError("No API key found.")
        
        if not base_url:
            raise ValueError("No base URL key found.")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate_score(
        self,
        target_image: Image.Image,
        rendering_image: Image.Image,
        prompt: str,
        response_format: BaseModel,
        max_retries: int = 3,
    ) -> VLMSceneScore:
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.parse(
                    model="google/gemini-3.1-flash-lite",
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _image_to_data_url(target_image)},
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": _image_to_data_url(rendering_image)},
                                },
                            ],
                        },
                    ],
                    response_format=response_format,
                    max_tokens=1024,

                )
                return completion.choices[0].message.parsed
            except (openai.APIConnectionError, openai.InternalServerError):
                if attempt == max_retries - 1:
                    raise
                print("Error in VLM score, waiting 5 second before retrying...")
                time.sleep(5)



def compute_vlm_score(target_path: str, render_path: str, score_type: str) -> dict[str, float]:
    vlm_agent = VLMScoreAgent()

    if score_type == "count":
        with open("/Users/jonathansickert/git/3DFrontBench/prompts/vlm_count_score_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMCountScore
    elif score_type == "placement":
        with open("/Users/jonathansickert/git/3DFrontBench/prompts/vlm_placement_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMPlacementScore
    elif score_type == "orientation":
        with open("/Users/jonathansickert/git/3DFrontBench/prompts/vlm_orientation_score_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMOrientationScore
    elif score_type == "scale":
        with open("/Users/jonathansickert/git/3DFrontBench/prompts/vlm_scale_score_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMScaleScore


    target_img = Image.open(target_path)
    render_img = Image.open(render_path)

    result = vlm_agent.generate_score(
        target_image=target_img,
        rendering_image=render_img,
        response_format=response_format,
        prompt=prompt,
    )

    return result.model_dump()
