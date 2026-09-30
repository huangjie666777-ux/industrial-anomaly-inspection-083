"""Image decoding, validation and whole-image ResNet preprocessing."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from . import config


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable image."""


@dataclass(frozen=True)
class LoadedImage:
    original: Image.Image  # RGB, original resolution
    tensor: torch.Tensor  # 1x3x224x224, ImageNet normalized
    content_hash: str  # identity of resized pixel content


def _check_dimensions(image: Image.Image) -> None:
    width, height = image.size
    if width < config.MIN_SIDE or height < config.MIN_SIDE:
        raise ImageValidationError(
            f"图像尺寸过小：{width}x{height}，最短边至少 {config.MIN_SIDE} 像素"
        )
    if width > config.MAX_SIDE or height > config.MAX_SIDE:
        raise ImageValidationError(
            f"图像尺寸过大：{width}x{height}，最长边不得超过 {config.MAX_SIDE} 像素"
        )


def load_image(data: bytes, suffix: str = "", content_type: str = "") -> LoadedImage:
    """Decode and validate an upload, returning the RGB image and 224x224 tensor.

    The whole image is resized to 224x224 (no center crop) and normalized with
    ImageNet statistics. Content identity is hashed from resized RGB pixels so
    that re-encodings of the same image content still compare equal.
    """
    suffix = suffix.lower()
    if suffix and suffix not in config.ALLOWED_SUFFIXES:
        raise ImageValidationError("仅支持 PNG 或 JPEG 格式（.png/.jpg/.jpeg）")
    if content_type and content_type not in config.ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(f"不支持的 Content-Type：{content_type}")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("文件无法解码为 PNG/JPEG 图像") from exc
    if image.format not in {"PNG", "JPEG"}:
        raise ImageValidationError(f"仅支持 PNG/JPEG，实际格式为 {image.format}")
    rgb = image.convert("RGB")
    _check_dimensions(rgb)

    resized = rgb.resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    content_hash = hashlib.sha256(np.asarray(resized).tobytes()).hexdigest()

    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return LoadedImage(original=rgb, tensor=tensor, content_hash=content_hash)


def upsample_distance_map(
    distance_map: torch.Tensor, size: tuple[int, int]
) -> np.ndarray:
    """Bilinearly restore a GRID_SIZE x GRID_SIZE distance map to ``size``."""
    matrix = distance_map.view(1, 1, config.GRID_SIZE, config.GRID_SIZE).float()
    restored = F.interpolate(matrix, size=size, mode="bilinear", align_corners=False)
    return restored.squeeze(0).squeeze(0).cpu().numpy()
