"""Single-image inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.data.augmentations import build_eval_transforms
from src.models.astrolocnet import AstroLocNet
from src.utils.coordinates import wrap_ra_deg, clamp_dec_deg
from src.utils.io import load_image_rgb


@dataclass
class Prediction:
    ra_deg: float
    dec_deg: float
    rotation_deg: float
    field_width_deg: float


def load_model(checkpoint_path: str | Path, *, device: str = "cpu") -> AstroLocNet:
    """Load a trained AstroLocNet from a checkpoint file."""
    model = AstroLocNet(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    model: AstroLocNet,
    image: str | Path | np.ndarray | Image.Image,
    *,
    image_size: int = 224,
    device: str = "cpu",
) -> Prediction:
    """Run the model on one image and return the parsed prediction."""
    if isinstance(image, (str, Path)):
        rgb, _ = load_image_rgb(image)
        pil = Image.fromarray(rgb)
    elif isinstance(image, np.ndarray):
        pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type {type(image)!r}")

    transform = build_eval_transforms(image_size)
    tensor = transform(pil).unsqueeze(0).to(device)
    out = model(tensor).squeeze(0).cpu().numpy()
    return Prediction(
        ra_deg=float(wrap_ra_deg(out[0])),
        dec_deg=float(clamp_dec_deg(out[1])),
        rotation_deg=float(wrap_ra_deg(out[2])),
        field_width_deg=float(np.exp(out[3])),
    )
