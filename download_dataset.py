import os
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download(
        repo_id="JonathanSickert/3DFront-eval",
        repo_type="dataset",
        local_dir="./dataset",
        token=os.getenv("HF_TOKEN"),
    )
