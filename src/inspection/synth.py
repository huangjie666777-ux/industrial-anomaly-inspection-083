"""Deterministic synthetic surface images for demos and tests.

These images are NOT training data: they provide clickable, reproducible
examples so a quality engineer can try the workflow without real samples.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw

SIZE = 256


def _base(seed: int, phase: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    texture = (
        12 * np.sin((xx * 2 * math.pi / 42) + phase)
        + 8 * np.cos((yy * 2 * math.pi / 31) - phase * 0.7)
        + 5 * np.sin((xx + yy) * 2 * math.pi / 90 + phase * 1.3)
    )
    noise = rng.normal(0, 4.0, (SIZE, SIZE)).astype(np.float32)
    shading = 8 * (xx / SIZE) - 6 * (yy / SIZE)
    image = 150.0 + texture + noise + shading
    return np.clip(image, 60, 230)


def normal_image(variant: int = 0) -> Image.Image:
    """Normal textured surface; each variant is a distinct image content."""
    seed = 1000 + variant * 37
    phase = 0.35 * variant
    array = _base(seed, phase).astype(np.uint8)
    return Image.fromarray(np.stack([array, array, array], axis=-1), mode="RGB")


def scratch_image(variant: int = 0) -> Image.Image:
    """Normal-like surface with a dark diagonal scratch (true anomaly demo)."""
    image = normal_image(variant).convert("RGB")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(5000 + variant)
    x0, y0 = 40 + rng.integers(0, 40), 40 + rng.integers(0, 60)
    x1, y1 = 210 - rng.integers(0, 40), 210 - rng.integers(0, 50)
    draw.line([(x0, y0), (x1, y1)], fill=(25, 22, 20), width=3)
    draw.line([(x0 + 8, y0 - 10), (x1 - 6, y1 + 8)], fill=(40, 38, 34), width=1)
    return image


def foreign_object_image(variant: int = 0) -> Image.Image:
    """Normal-like surface with a dark foreign particle/blob (true anomaly)."""
    image = normal_image(variant).convert("RGB")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(7000 + variant)
    cx, cy = 80 + rng.integers(0, 100), 80 + rng.integers(0, 90)
    radius = 10 + rng.integers(0, 6)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(35, 28, 20))
    draw.ellipse(
        [cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2],
        fill=(70, 55, 35),
    )
    return image


EXAMPLES = {
    "normal_ref": {
        "maker": normal_image,
        "variants": 5,
        "purpose": "正常产品图，用于上传到【参考正常图】建立记忆库",
    },
    "normal_cal": {
        "maker": lambda v: normal_image(20 + v),
        "variants": 5,
        "purpose": "与参考图内容不同的正常图，用于上传到【独立校准正常图」确定阈值",
    },
    "scratch": {
        "maker": scratch_image,
        "variants": 3,
        "purpose": "含划痕的异常示例，用于【检测新图】验证定位",
    },
    "foreign_object": {
        "maker": foreign_object_image,
        "variants": 3,
        "purpose": "含异物的异常示例，用于【检测新图】验证定位",
    },
}


def example_png(kind: str, variant: int = 0) -> bytes:
    spec = EXAMPLES[kind]
    if not 0 <= variant < spec["variants"]:
        raise KeyError(kind)
    buffer = io.BytesIO()
    spec["maker"](variant).save(buffer, format="PNG")
    return buffer.getvalue()
