import io

import torch
from PIL import Image

from inspection import synth
from inspection.preprocessing import load_image, upsample_distance_map


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_whole_image_preprocessing_shape_and_norm():
    loaded = load_image(_png_bytes(synth.normal_image(0)), suffix=".png")
    assert loaded.tensor.shape == (1, 3, 224, 224)
    assert loaded.original.size == (256, 256)
    assert loaded.tensor.min() < 0


def test_distinct_variants_have_distinct_hash():
    a = load_image(_png_bytes(synth.normal_image(0)), suffix=".png")
    c = load_image(_png_bytes(synth.normal_image(1)), suffix=".png")
    assert a.content_hash != c.content_hash
    assert len(a.content_hash) == 64


def test_rejects_non_image():
    try:
        load_image(b"not an image", suffix=".png")
    except Exception:
        return
    raise AssertionError("non-image upload should be rejected")


def test_upsample_restores_size():
    matrix = torch.arange(196, dtype=torch.float32)
    restored = upsample_distance_map(matrix, (100, 130))
    assert restored.shape == (100, 130)
