import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

BASE_DIR = Path("./3D-FRONT")

if __name__ == "__main__":
    # download main 3D front json files
    hf_hub_download(
        repo_id="JIAQI-CHEN/3D-Front",
        repo_type="dataset",
        filename="3D-FRONT.zip",
        local_dir=BASE_DIR,
    )

    # download 3D front textures
    hf_hub_download(
        repo_id="JIAQI-CHEN/3D-Front",
        repo_type="dataset",
        filename="3D-FRONT-texture.zip",
        local_dir=BASE_DIR,
    )

    # download 3D front furniture
    for part in ["part1", "part2", "part3", "part4"]:
        hf_hub_download(
            repo_id="JIAQI-CHEN/3D-Front",
            repo_type="dataset",
            filename=f"3D-FUTURE-model-{part}.zip",
            local_dir=BASE_DIR,
        )

    # download 3D front cctextures
    hf_hub_download(
        repo_id="JIAQI-CHEN/3D-Front",
        repo_type="dataset",
        filename="cctextures.zip",
        local_dir=BASE_DIR,
    )

    # unzip
    for zip_name in ["3D-FRONT.zip", "3D-FRONT-texture.zip", "cctextures.zip"]:
        with zipfile.ZipFile(BASE_DIR / zip_name) as zf:
            zf.extractall(BASE_DIR)

    for part in ["part1", "part2", "part3", "part4"]:
        with zipfile.ZipFile(BASE_DIR / f"3D-FUTURE-model-{part}.zip") as zf:
            zf.extractall(BASE_DIR)

    # merge part dirs into a single 3D-FUTURE directory
    future_dir = BASE_DIR / "3D-FUTURE"
    future_dir.mkdir(exist_ok=True)
    for part in ["part1", "part2", "part3", "part4"]:
        part_dir = BASE_DIR / f"3D-FUTURE-model-{part}"
        for item in part_dir.iterdir():
            shutil.move(str(item), future_dir)

    # cleanup zip files, part dirs, and __MACOSX artifacts
    for zip_path in BASE_DIR.glob("*.zip"):
        zip_path.unlink()
    for part in ["part1", "part2", "part3", "part4"]:
        shutil.rmtree(BASE_DIR / f"3D-FUTURE-model-{part}", ignore_errors=True)
    shutil.rmtree(BASE_DIR / "__MACOSX", ignore_errors=True)
