"""Runtime configuration for the inspection application."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18-f37072fd.pth"
WEB_DIR = PROJECT_ROOT / "web"
DATA_DIR = Path(os.environ.get("INSPECTION_DATA_DIR", PROJECT_ROOT / "data"))

IMAGE_SIZE = 224
GRID_SIZE = 14  # layer3 spatial resolution; descriptors are GRID_SIZE**2
MAX_MEMORY_ITEMS = 256
PERCENTILE = 95.0
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_SIDE = 32
MAX_SIDE = 8192
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "application/octet-stream",
}

# When the calibration threshold is exactly zero the heatmap needs an explicit,
# calibration-derived upper bound instead of per-image autoscaling.
DISTANCE_FLOOR = 1e-6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
