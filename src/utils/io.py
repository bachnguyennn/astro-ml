"""Image I/O and EXIF helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


_DATETIME_ORIGINAL_TAG = 36867


def load_image_rgb(path: str | Path) -> Tuple[np.ndarray, Image.Image]:
    """Load an image and return (uint8 H,W,3 RGB array, PIL Image)."""
    pil = Image.open(path)
    pil.load()
    rgb = np.array(pil.convert("RGB"))
    return rgb, pil


def extract_exif_timestamp(pil_image: Image.Image) -> Optional[datetime]:
    """Return DateTimeOriginal as UTC-aware datetime, or None."""
    try:
        exif = pil_image.getexif()
    except (AttributeError, OSError):
        return None
    if not exif:
        return None
    raw = exif.get(_DATETIME_ORIGINAL_TAG) or exif.get(36868) or exif.get(306)
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", errors="ignore")
    try:
        naive = datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone.utc)
