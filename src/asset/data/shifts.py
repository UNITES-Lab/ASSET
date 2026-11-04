from __future__ import annotations

import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from skimage.transform import iradon, radon

from asset.data import relocate, write_task

FAMILIES = {
    "ct_gaussian": ("ct", "gaussian", [60, 120, 180, 240]),
    "ct_poisson": ("ct", "poisson", [0.5, 0.25, 0.1, 0.05]),
    "mri_rician": ("mri", "rician", [0.05, 0.1, 0.15, 0.2, 0.25]),
    "mri_kspace": ("mri", "kspace", [2, 4, 6, 8]),
    "vae": ("mixed", "vae_noise", [1.0, 2.0, 3.0, 4.0, 5.0]),
}
KSPACE_CENTER = {2: 0.16, 4: 0.08, 6: 0.06, 8: 0.04}
POISSON_I0 = 1e5
POISSON_MAX_ATTENUATION = 10.0
POISSON_MAX_EDGE = 512
VAE = "stabilityai/sd-vae-ft-mse"
SEED = 2026


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") ^ SEED) % (2**32)


def to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def on_luminance(image: np.ndarray, transform) -> np.ndarray:
    if not (image.ndim == 3 and image.shape[2] == 3):
        return transform(image)
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    ratio = transform(gray) / (gray + 1e-8)
    return np.clip(image * ratio[..., None], 0.0, 1.0)


def match_shape(array: np.ndarray, shape: tuple) -> np.ndarray:
    result = np.zeros(shape, dtype=array.dtype)
    sy, sx = array.shape[:2]
    ty, tx = shape[:2]
    src_y, src_x = max(0, (sy - ty) // 2), max(0, (sx - tx) // 2)
    dst_y, dst_x = max(0, (ty - sy) // 2), max(0, (tx - sx) // 2)
    h, w = min(sy - src_y, ty - dst_y), min(sx - src_x, tx - dst_x)
    result[dst_y : dst_y + h, dst_x : dst_x + w] = array[src_y : src_y + h, src_x : src_x + w]
    return result


def gaussian(image: np.ndarray, std: float) -> np.ndarray:
    noisy = image.astype(np.float64) + np.random.normal(0, std, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rician(image: np.ndarray, sigma: float) -> np.ndarray:
    def transform(gray):
        n1 = np.random.normal(0, sigma, size=gray.shape)
        n2 = np.random.normal(0, sigma, size=gray.shape)
        noisy = np.sqrt((gray + n1) ** 2 + n2**2)
        peak = noisy.max()
        return noisy / peak if peak > 0 else noisy

    return to_uint8(on_luminance(to_float(image), transform))


def kspace(image: np.ndarray, acceleration: int) -> np.ndarray:
    def transform(gray):
        rows = gray.shape[0]
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        center = int(np.round(KSPACE_CENTER[acceleration] * rows))
        start = rows // 2 - center // 2
        mask = np.zeros(rows, dtype=bool)
        mask[start : start + center] = True
        remaining = np.where(~mask)[0]
        count = max(1, int(np.round(len(remaining) / acceleration)))
        mask[np.random.choice(remaining, size=count, replace=False)] = True
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum * mask[:, np.newaxis].astype(np.float64))))
        low, high = recon.min(), recon.max()
        return (recon - low) / (high - low) if high - low > 0 else np.zeros_like(recon)

    return to_uint8(on_luminance(to_float(image), transform))


def poisson(image: np.ndarray, dose: float) -> np.ndarray:
    def transform(gray):
        theta = np.linspace(0.0, 180.0, max(gray.shape), endpoint=False)
        sinogram = radon(gray, theta=theta, circle=False)
        peak = sinogram.max()
        if peak <= 0:
            return gray
        counts = POISSON_I0 * dose
        transmitted = np.clip(counts * np.exp(-(sinogram / peak * POISSON_MAX_ATTENUATION)), 0, 1e10)
        noisy = np.random.poisson(lam=transmitted).astype(np.float64)
        noisy_sinogram = -np.log(np.maximum(noisy, 1) / counts) / POISSON_MAX_ATTENUATION * peak
        recon = iradon(noisy_sinogram, theta=theta, circle=False)
        return np.clip(match_shape(recon, gray.shape), 0.0, 1.0)

    return to_uint8(on_luminance(to_float(image), transform))


def shrink(image: Image.Image, max_edge: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_edge:
        return image
    scale = max_edge / float(max(width, height))
    return image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.BILINEAR)


def degrade(job: tuple) -> None:
    family, level, source, output, seed = job
    np.random.seed(seed)
    image = Image.open(source)
    if family == "poisson":
        degraded = Image.fromarray(poisson(np.array(shrink(image, POISSON_MAX_EDGE)), level))
        if degraded.size != image.size:
            degraded = degraded.resize(image.size, Image.BILINEAR)
    elif family == "gaussian":
        degraded = Image.fromarray(gaussian(np.array(image), level))
    elif family == "rician":
        degraded = Image.fromarray(rician(np.array(image), level))
    else:
        degraded = Image.fromarray(kspace(np.array(image), level))
    degraded.save(output)


def vae_degrade(vae, job: tuple, device: str) -> None:
    _, sigma, source, output, seed = job
    image = Image.open(source)
    gray = np.array(image.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.stack([gray, gray, gray], axis=0) * 2.0 - 1.0).unsqueeze(0).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.mean
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
        recon = vae.decode(latent + sigma * noise).sample
    array = np.clip((recon.squeeze(0).cpu().numpy().mean(axis=0) + 1.0) / 2.0, 0, 1)
    degraded = Image.fromarray((array * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    (degraded.convert("RGB") if image.mode == "RGB" else degraded).save(output)


def select(rows: list[dict], total: int, modality: str) -> list[dict]:
    if modality in {"ct", "mri"}:
        return [row for row in rows if row["modality"] == modality][:total]
    counts = Counter(row["modality"] for row in rows)
    if counts["ct"] >= total // 2 and counts["mri"] >= total // 2:
        ct = [row for row in rows if row["modality"] == "ct"][: total // 2]
        mri = [row for row in rows if row["modality"] == "mri"][: total - len(ct)]
        return ct + mri
    return rows[:total]


def unique_images(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        seen.setdefault(row["image_path"], row)
    return list(seen.values())


def build_tasks(dataset: str, data_dir: Path, splits: dict[str, list[dict]], workers: int, device: str) -> None:
    jobs = []
    for name, (modality, family, levels) in FAMILIES.items():
        task = f"{dataset}_{name}"
        task_dir = data_dir.resolve() / "tasks" / task
        reference = select(splits["reference_pool"], 32, modality)
        support = select(splits["support_pool"], 32, modality)
        evaluation = [row for row in splits["eval"] if modality == "mixed" or row["modality"] == modality]
        conditions = [f"{family}_{level}" for level in levels]
        anchor = conditions[2]
        for condition, level in zip(conditions, levels):
            (task_dir / "images" / condition).mkdir(parents=True, exist_ok=True)
            rows = evaluation + support if condition == anchor else evaluation
            for row in unique_images(rows):
                output = relocate(row, task_dir, condition)["image_path"]
                jobs.append((family, level, row["image_path"], output, stable_seed(task, condition, row["image_id"])))
        write_task(
            task_dir,
            modality,
            family,
            "medical",
            [relocate(row, task_dir, anchor) for row in support],
            reference,
            {
                "clean": evaluation,
                **{condition: [relocate(row, task_dir, condition) for row in evaluation] for condition in conditions},
            },
        )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(degrade, [job for job in jobs if job[0] != "vae_noise"], chunksize=4))
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float32).to(device).eval()
    for job in jobs:
        if job[0] == "vae_noise":
            vae_degrade(vae, job, device)
