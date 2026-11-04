from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from functools import partial
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def resolution_resample(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = image.size
    scale = float(rng.uniform(0.72, 0.92))
    size = (max(32, int(round(w * scale))), max(32, int(round(h * scale))))
    return image.resize(size, Image.BICUBIC).resize((w, h), Image.BICUBIC)


def low_pass_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.45, 1.25))))


def gamma_contrast(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image) / 255.0
    gamma = float(rng.uniform(0.78, 1.28))
    contrast = float(rng.uniform(0.86, 1.18))
    array = (np.power(np.clip(array, 0.0, 1.0), gamma) - 0.5) * contrast + 0.5
    return to_image(array * 255.0)


def gaussian_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image)
    sigma = float(rng.uniform(4.0, 15.0))
    return to_image(array + rng.normal(0.0, sigma, size=array.shape).astype(np.float32))


def jpeg_compression(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(rng.integers(35, 76)))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def window_level_shift(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image)
    center = 127.5 + float(rng.uniform(-28.0, 28.0))
    width = 255.0 * float(rng.uniform(0.78, 1.22))
    return to_image((array - (center - width / 2.0)) / max(width, 1e-6) * 255.0)


def reconstruction_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.65, 1.55))))


def reconstruction_sharpening(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    radius = float(rng.uniform(0.7, 1.4))
    percent = int(rng.integers(110, 190))
    threshold = int(rng.integers(2, 8))
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def partial_volume(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = image.size
    scale = float(rng.uniform(0.62, 0.84))
    size = (max(32, int(round(w * scale))), max(32, int(round(h * scale))))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 0.65))))
    return blurred.resize(size, Image.BICUBIC).resize((w, h), Image.BICUBIC)


def rician_noise(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image) / 255.0
    sigma = float(rng.uniform(0.018, 0.055))
    real = array + rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    imag = rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    return to_image(np.sqrt(real * real + imag * imag) * 255.0)


def truncate_spectrum(channel: np.ndarray, keep: float) -> np.ndarray:
    h, w = channel.shape
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    keep_h = max(4, int(round(h * keep)))
    keep_w = max(4, int(round(w * keep)))
    mask = np.zeros((h, w), dtype=bool)
    top = (h - keep_h) // 2
    left = (w - keep_w) // 2
    mask[top : top + keep_h, left : left + keep_w] = True
    spectrum = np.where(mask, spectrum, 0.0)
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum))).astype(np.float32)


def gibbs_ringing(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image)
    keep = float(rng.uniform(0.68, 0.86))
    return to_image(np.stack([truncate_spectrum(array[..., c], keep) for c in range(3)], axis=-1))


def motion_ghosting(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image)
    axis = int(rng.integers(0, 2))
    shift = int(rng.integers(max(2, array.shape[axis] // 32), max(3, array.shape[axis] // 9)))
    ghost = np.roll(array, shift, axis=axis)
    if rng.random() < 0.5:
        ghost = np.roll(ghost, shift, axis=axis)
    alpha = float(rng.uniform(0.08, 0.22))
    return to_image((1.0 - alpha) * array + alpha * ghost)


def kspace_spike(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = to_array(image)
    h, w = array.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frequency = float(rng.uniform(5.0, 15.0))
    angle = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    coordinate = (xx * np.cos(angle) + yy * np.sin(angle)) / max(h, w)
    stripe = np.sin(2.0 * np.pi * frequency * coordinate + phase)
    amplitude = float(rng.uniform(5.0, 16.0))
    return to_image(array + amplitude * stripe[..., None])


def anisotropic_resolution(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = image.size
    if rng.random() < 0.5:
        size = (max(32, int(round(w * float(rng.uniform(0.58, 0.82))))), h)
    else:
        size = (w, max(32, int(round(h * float(rng.uniform(0.58, 0.82))))))
    return image.resize(size, Image.BICUBIC).resize((w, h), Image.BICUBIC)


Transform = Callable[[Image.Image, np.random.Generator], Image.Image]

GENERAL: list[tuple[str, Transform]] = [
    ("resolution_resample", resolution_resample),
    ("low_pass_blur", low_pass_blur),
    ("gamma_contrast", gamma_contrast),
    ("gaussian_noise", gaussian_noise),
    ("jpeg_compression", jpeg_compression),
]

CT: list[tuple[str, Transform]] = [
    ("window_level_shift", window_level_shift),
    ("reconstruction_blur", reconstruction_blur),
    ("reconstruction_sharpening", reconstruction_sharpening),
    ("partial_volume", partial_volume),
]

MRI: list[tuple[str, Transform]] = [
    ("rician_noise", rician_noise),
    ("gibbs_ringing", gibbs_ringing),
    ("motion_ghosting", motion_ghosting),
    ("kspace_spike", kspace_spike),
    ("anisotropic_resolution", anisotropic_resolution),
]

EXCLUDED = {"gaussian": {"gaussian_noise"}, "rician": {"rician_noise"}}

CORRUPTIONS: list[tuple[str, str]] = [
    ("shot_noise", "noise"),
    ("gaussian_noise", "noise"),
    ("impulse_noise", "noise"),
    ("speckle_noise", "noise"),
    ("defocus_blur", "blur"),
    ("zoom_blur", "blur"),
    ("snow", "weather"),
    ("brightness", "image_property"),
    ("contrast", "image_property"),
    ("saturate", "image_property"),
    ("elastic_transform", "physical"),
    ("spatter", "physical"),
    ("pixelate", "digital"),
    ("jpeg_compression", "digital"),
]

PROXY_SEVERITY = 3


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("::".join(str(part) for part in parts).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def corrupt(array: np.ndarray, name: str, severity: int, parts: tuple) -> np.ndarray:
    from imagecorruptions import corrupt as apply_corruption

    seed = stable_seed("imagecorruptions", name, severity, *parts)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    corrupted = apply_corruption(array, corruption_name=name, severity=int(severity))
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    return corrupted


def medical_views(
    image: Image.Image, count: int, seed: int, transforms: list[tuple[str, Transform]]
) -> list[Image.Image]:
    if not transforms:
        return [image.copy() for _ in range(count)]
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(count):
        _, transform = transforms[int(rng.integers(0, len(transforms)))]
        views.append(transform(image.copy(), rng))
    return views


def corruption_views(image: Image.Image, count: int, seed: int, target: str, names: list[str]) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    views = []
    for index in range(count):
        name = names[int(rng.integers(0, len(names)))]
        view_seed = stable_seed(seed, target, name, PROXY_SEVERITY, index)
        views.append(Image.fromarray(corrupt(array, name, PROXY_SEVERITY, (seed, view_seed, index))))
    return views


def proxy_bank(task: dict, subset: str = "full") -> Callable[[Image.Image, int, int], list[Image.Image]]:
    if task["proxy_bank"] == "corruption":
        category = dict(CORRUPTIONS)[task["family"]]
        names = [name for name, other in CORRUPTIONS if other != category]
        return partial(corruption_views, target=task["family"], names=names)
    excluded = EXCLUDED.get(task["family"], set())
    general = [item for item in GENERAL if item[0] not in excluded]
    domain = {"ct": CT, "mri": MRI, "mixed": CT + MRI}.get(task["modality"], [])
    domain = [item for item in domain if item[0] not in excluded]
    transforms = {"full": general + domain, "general": general, "modality": domain}[subset]
    return partial(medical_views, transforms=transforms)
