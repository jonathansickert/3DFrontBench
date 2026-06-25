import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def to_tensor(img: np.ndarray):
    t = torch.tensor(img).float() / 255.0
    t = t * 2 - 1
    t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t.to(device)


def compute_lpips(img1: np.ndarray, img2: np.ndarray) -> float:
    model = lpips.LPIPS()
    model.eval()
    model = model.to(device)

    t1, t2 = to_tensor(img1), to_tensor(img2)
    with torch.no_grad():
        dist = model(t1, t2)

    return float(dist.item())


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    return float(np.mean(np.abs(a - b)))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if np.array_equal(img1, img2):
        return float("inf")
    return float(psnr(img1, img2, data_range=255))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(ssim(img1, img2, data_range=255, channel_axis=2))


def compute_all_metrics(path1: str, path2: str) -> dict[str, float]:
    img1, img2 = load_image(path1), load_image(path2)

    results = {
        "ssim": compute_ssim(img1, img2),
        "mae": compute_mae(img1, img2),
        "psnr": compute_psnr(img1, img2),
        "lpips": compute_lpips(img1, img2),
    }

    return results
