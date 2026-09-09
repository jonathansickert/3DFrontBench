import base64
import io
import os
from typing import Literal

import openai
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
import time
from src.util import resize_image


class ObjectPerturbationPrediction(BaseModel):
    object_number: int
    object_label: str
    evidence: str
    perturbation_type: Literal["removed", "scale", "rotation", "translation", "none"]
    confidence: float = Field(ge=0.0, le=1.0)

class VLMPerturbationDetectionResult(BaseModel):
    predictions: list[ObjectPerturbationPrediction]


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class VLMScoreAgent:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model_choice: str = "gemini"):
        if model_choice == "qwen8_instruct":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = os.getenv("OPENROUTER_QWEN8_INSTRUCT")
        elif model_choice == "qwen8_thinking":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = os.getenv("OPENROUTER_QWEN8_THINKING")
        elif model_choice == "gemma4":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            base_url = base_url or os.getenv("GOOGLE_BASE_URL")
            model = "gemma-4-31b-it"
        elif model_choice == "gemini31":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            base_url = base_url or os.getenv("GOOGLE_BASE_URL")
            model = "gemini-3.1-flash-lite"
        elif model_choice == "gemini35":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            base_url = base_url or os.getenv("GOOGLE_BASE_URL")
            model = "gemini-3.5-flash-lite"
        elif model_choice == "qwen32_instruct":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = os.getenv("OPENROUTER_QWEN32")
        elif model_choice == "qwen235_instruct":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = "qwen/qwen3-vl-235b-a22b-instruct"
        elif model_choice == "opus":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = os.getenv("OPENROUTER_OPUS")
        elif model_choice == "gpt51":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
            model = "openai/gpt-5.1"
        else:
            raise ValueError("Unknown Model: ", model_choice)

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
        response_format: type[BaseModel],
        max_retries: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> BaseModel:
        messages = [
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
        ]

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                message = completion.choices[0].message
                if message.refusal:
                    raise ValueError(f"model refused to answer: {message.refusal}")
                if message.parsed is None:
                    raise ValueError("model response did not match the expected schema")
                return message.parsed
            except (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError) as error:
                if attempt == max_retries - 1:
                    raise
                print(f"Error in VLM score ({error}), waiting 5 seconds before retrying...")
                time.sleep(5)
            except (
                openai.LengthFinishReasonError,
                openai.ContentFilterFinishReasonError,
                ValidationError,
                ValueError,
            ) as error:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"VLM did not return a valid {response_format.__name__} after {max_retries} attempts"
                    ) from error

                print(f"VLM returned a malformed response ({error}), retrying...")
                if isinstance(error, openai.LengthFinishReasonError):
                    # the response was cut off before it could complete the JSON object
                    max_tokens *= 2
                else:
                    # at temperature=0 the model would otherwise repeat the same malformed
                    # response every retry, so nudge it off that deterministic path
                    temperature = max(temperature, 0.2)
                time.sleep(5)


def compute_vlm_perturbation_detection(
    target_path: str,
    render_path: str,
    legend: list[dict[str, int]],
    model_choice: str,
    max_side: int | None = 1280,
) -> dict:
    vlm_agent = VLMScoreAgent(model_choice=model_choice)

    with open("/home/jonathansickert/git/3DFrontBench/prompts/vlm_perturbation_detection_prompt.txt") as f:
        prompt_template = f.read()

    legend_lines = "\n".join(f"{digit}: {label}" for entry in legend for label, digit in entry.items())
    prompt = prompt_template.format(legend=legend_lines)

    target_img = Image.open(target_path)
    render_img = Image.open(render_path)
    # None skips this entirely -- e.g. optimize_rendering.py needs the resolution it rendered at
    # to actually reach the VLM, since resize_image only ever downsizes and this default would
    # otherwise collapse anything above 1280px back down before it's ever scored.
    if max_side is not None:
        target_img = resize_image(target_img, max_side=max_side)
        render_img = resize_image(render_img, max_side=max_side)

    result = vlm_agent.generate_score(
        target_image=target_img,
        rendering_image=render_img,
        response_format=VLMPerturbationDetectionResult,
        prompt=prompt,
        max_tokens=4096,
    )

    return result.model_dump()
