import os
from huggingface_hub import HfApi


if __name__ == "__main__":
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_large_folder(
        folder_path="./dataset",
        repo_id="JonathanSickert/3DFront-eval",
        repo_type="dataset",
    )
