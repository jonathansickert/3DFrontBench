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


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VLMScoreAgent:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key found.")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def generate_score(
        self,
        target_image: Image.Image,
        rendering_image: Image.Image,
        prompt: str,
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
                    response_format=VLMSceneScore,
                    max_tokens=1024,

                )
                return completion.choices[0].message.parsed
            except (openai.APIConnectionError, openai.InternalServerError):
                if attempt == max_retries - 1:
                    raise
                print("Error in VLM score, waiting 5 second before retrying...")
                time.sleep(5)



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
