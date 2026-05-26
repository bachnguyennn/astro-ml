"""PyTorch Dataset classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.renderer import render_star_field, sample_random_pointing
from src.utils.io import load_image_rgb


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #


class SyntheticStarFieldDataset(Dataset):
    """Generates star-field images on the fly from the HYG catalog.

    Labels are ``[ra_deg, dec_deg, rotation_deg, log_field_width_deg]``
    as a float32 tensor of shape ``[4]``.

    The dataset is *deterministic per index* when ``seed`` is provided:
    sample ``i`` always renders the same image. This makes the
    validation split reproducible without caching to disk.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        *,
        n_samples: int = 50_000,
        image_size: int = 224,
        field_width_range: Tuple[float, float] = (15.0, 80.0),
        noise_level: float = 0.02,
        sky_gradient: bool = True,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.catalog = catalog
        self.n_samples = int(n_samples)
        self.image_size = image_size
        self.field_width_range = field_width_range
        self.noise_level = noise_level
        self.sky_gradient = sky_gradient
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= idx < self.n_samples:
            raise IndexError(idx)
        # Deterministic per-sample RNG so val set is reproducible.
        seed = (self.seed or 0) * 1_000_003 + idx
        rng = np.random.default_rng(seed)

        ra, dec, rot, fw = sample_random_pointing(rng, self.field_width_range)
        img = render_star_field(
            ra, dec, fw, rot,
            self.catalog,
            image_size=self.image_size,
            noise_level=self.noise_level,
            sky_gradient=self.sky_gradient,
            rng=rng,
        )
        # Hand a uint8 RGB PIL Image to the transform pipeline.
        img_pil = Image.fromarray((img * 255.0).astype(np.uint8))
        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        label = torch.tensor([ra, dec, rot, float(np.log(fw))], dtype=torch.float32)
        return img_tensor, label


# --------------------------------------------------------------------------- #
# Real images (Astrometry.net solved)
# --------------------------------------------------------------------------- #


class AstrometryNetDataset(Dataset):
    """Wraps a folder of solved real images + JSON calibration files.

    Expects pairs ``<id>.{jpg,png}`` + ``<id>.json`` where the JSON has
    keys ``ra``, ``dec``, ``orientation`` (deg) and ``radius`` (deg,
    field radius — we convert to ``field_width = 2 * radius``).
    """

    def __init__(
        self,
        images_dir: str | Path,
        *,
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(images_dir)
        self.image_size = image_size
        self.transform = transform
        self.samples: List[Tuple[Path, Path]] = []
        if self.root.exists():
            for img_path in sorted(self.root.glob("*.[jp][pn]g")):
                cal_path = img_path.with_suffix(".json")
                if cal_path.exists():
                    self.samples.append((img_path, cal_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, cal_path = self.samples[idx]
        rgb, _ = load_image_rgb(img_path)
        cal = json.loads(cal_path.read_text())
        ra = float(cal["ra"])
        dec = float(cal["dec"])
        orientation = float(cal.get("orientation", 0.0))
        radius = float(cal["radius"])  # half-FOV in degrees
        field_width = max(1e-3, 2.0 * radius)

        if self.image_size and (rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size):
            pil = Image.fromarray(rgb).resize((self.image_size, self.image_size), Image.BILINEAR)
        else:
            pil = Image.fromarray(rgb)

        if self.transform is not None:
            img_tensor = self.transform(pil)
        else:
            img_tensor = torch.from_numpy(np.array(pil).transpose(2, 0, 1) / 255.0).float()

        label = torch.tensor(
            [ra % 360.0, np.clip(dec, -90, 90), orientation % 360.0, float(np.log(field_width))],
            dtype=torch.float32,
        )
        return img_tensor, label
