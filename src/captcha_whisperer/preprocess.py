"""Image preprocessing.

The model expects a 3x60x160 float tensor whose values stay in the original
0..255 range, with OpenCV's channel order (BGR). No mean subtraction and no
scaling are applied.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .constants import INPUT_HEIGHT, INPUT_WIDTH


def _to_bgr_chw(image: Image.Image, height: int, width: int) -> np.ndarray:
    image = image.convert("RGB").resize((width, height), Image.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32)
    bgr = rgb[:, :, ::-1]                      # RGB -> BGR to match the model
    return np.ascontiguousarray(bgr.transpose(2, 0, 1))


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes into a (1, 3, H, W) float32 tensor."""
    image = Image.open(io.BytesIO(data))
    return _to_bgr_chw(image, INPUT_HEIGHT, INPUT_WIDTH)[None]


def preprocess_array(image: np.ndarray) -> np.ndarray:
    """Convert an already-decoded image array into a (1, 3, H, W) tensor.

    Accepts a HxWx3 or HxWx3 BGR/RGB uint8 array; channels are assumed to be
    the last axis. Arrays produced by ``cv2.imread`` (BGR) are used as-is.
    """
    if image.ndim == 2:                        # grayscale
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:                    # drop alpha
        image = image[:, :, :3]
    bgr = np.ascontiguousarray(image.astype(np.float32).transpose(2, 0, 1))
    if bgr.shape[1:] != (INPUT_HEIGHT, INPUT_WIDTH):
        pil = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1].astype(np.uint8)))
        bgr = _to_bgr_chw(pil, INPUT_HEIGHT, INPUT_WIDTH)
    return bgr[None]
