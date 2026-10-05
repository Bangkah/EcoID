"""Image -> model input tensor (SRS FR-002).

Pipeline: open -> EXIF transpose -> RGB -> resize 224x224 -> /255
          -> ImageNet normalize -> CHW -> add batch dim (float32).

IMPORTANT: Phase 2 training MUST use the same resize + normalization,
otherwise accuracy silently drops (train/serve skew).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ImageSource = Union[str, Path, bytes, Image.Image]


class InvalidImageError(ValueError):
    """Raised when the input cannot be decoded as an image."""


def load_image(source: ImageSource) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()  # force decode now so corrupt files fail here
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise InvalidImageError(f"Cannot decode image: {e}") from e

    img = ImageOps.exif_transpose(img)  # phone photos: honor orientation
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)  # flatten transparency on white
    return img.convert("RGB")


def to_tensor(img: Image.Image) -> np.ndarray:
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC, [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))                    # CHW
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def preprocess(source: ImageSource) -> np.ndarray:
    """Returns float32 array of shape (1, 3, 224, 224)."""
    return to_tensor(load_image(source))
