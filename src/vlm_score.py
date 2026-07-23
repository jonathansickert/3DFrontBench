import base64
import io
import os

import openai
from PIL import Image
from pydantic import BaseModel
import time
from src.util import resize_image

class VLMPlacementScore(BaseModel):
    score: int
    reasoning: str
    missing_info: str

class VLMScaleScore(BaseModel):
    score: int
    reasoning: str
    missing_info: str

class VLMRotationScore(BaseModel):
    score: int
    reasoning: str
    missing_info: str

class VLMCountScore(BaseModel):
    score: int
    reasoning: str
    missing_info: str

class ObjectScore(BaseModel):
    label: str
    score: int
    note: str

class VLMPlacementScorePerObject(BaseModel):
    object_scores: list[ObjectScore]
    score: int
    reasoning: str
    missing_info: str

class VLMScaleScorePerObject(BaseModel):
    object_scores: list[ObjectScore]
    score: int
    reasoning: str
    missing_info: str

class VLMRotationScorePerObject(BaseModel):
    object_scores: list[ObjectScore]
    score: int
    reasoning: str
    missing_info: str

class VLMCountScorePerObject(BaseModel):
    object_scores: list[ObjectScore]
    score: int
    reasoning: str
    missing_info: str

def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VLMScoreAgent:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model_choice: str = "gemini"):
        if model_choice == "gemini":
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            base_url = base_url or os.getenv("GEMINI_BASE_URL")
            model = os.getenv("GEMINI_MODEL")
        if model_choice == "qwen":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = os.getenv("OPENROUTER_QWEN")
        if model_choice == "glm":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = os.getenv("OPENROUTER_GLM")

        self.model = model

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
        max_tokens: int = 1024,
    ) -> BaseModel:
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.parse(
                    model=self.model,
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
                    max_tokens=max_tokens,

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
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_count_score_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMCountScore
    elif score_type == "placement":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_placement_score_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMPlacementScore
    elif score_type == "rotation":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_rotation_score_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMRotationScore
    elif score_type == "scale":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_scale_score_prompt.txt") as f:
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


def compute_vlm_score_per_object(target_path: str, render_path: str, score_type: str, model_choice: str, resize_to_hd: bool = True) -> dict[str, float]:
    vlm_agent = VLMScoreAgent(model_choice=model_choice)

    if score_type == "count":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_count_score_per_object_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMCountScorePerObject
    elif score_type == "placement":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_placement_score_per_object_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMPlacementScorePerObject
    elif score_type == "rotation":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_rotation_score_per_object_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMRotationScorePerObject
    elif score_type == "scale":
        with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_scale_score_per_object_prompt.txt") as f:
            prompt = f.read()
        response_format = VLMScaleScorePerObject

    target_img = Image.open(target_path)
    render_img = Image.open(render_path)

    if resize_to_hd:
        target_img = resize_image(target_img)
        render_img = resize_image(render_img)

    result = vlm_agent.generate_score(
        target_image=target_img,
        rendering_image=render_img,
        response_format=response_format,
        prompt=prompt,
        max_tokens=4096,
    )

    return result.model_dump()
