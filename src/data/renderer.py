"""Synthetic star-field renderer.

Pipeline per sample:
    1. Sample (RA, Dec) uniformly on the sphere (NOT uniform in RA/Dec).
    2. Filter the catalog to stars within the field of view.
    3. Gnomonic-project them onto the tangent plane.
    4. Apply field rotation, then rescale to pixel coordinates.
    5. Splat each star as a magnitude-weighted 2D Gaussian.
    6. Add Poisson photon noise + Gaussian readout noise.
    7. Optionally add a smooth sky-background gradient (light pollution).

Returns a float32 ``[H, W, 3]`` image in [0, 1] suitable for direct use
with torchvision transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.coordinates import gnomonic_project


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_random_pointing(
    rng: np.random.Generator,
    field_width_range: Tuple[float, float] = (15.0, 80.0),
) -> Tuple[float, float, float, float]:
    """Uniform-on-sphere pointing + uniform rotation/field-width.

    Uniform-on-sphere means dec is sampled as ``arcsin(U(-1,1))`` rather
    than ``U(-90,90)``, which would over-represent the poles.
    """
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0)))
    ra = rng.uniform(0.0, 360.0)
    rotation = rng.uniform(0.0, 360.0)
    field_width = rng.uniform(*field_width_range)
    return float(ra), float(dec), float(rotation), float(field_width)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


@dataclass
class _RenderConfig:
    image_size: int
    noise_level: float
    sky_gradient: bool
    psf_base_sigma: float        # pixels, brightest star
    psf_min_sigma: float         # pixels, dimmest star
    mag_brightness_floor: float  # mag value mapped to peak brightness
    mag_brightness_ceiling: float  # mag value mapped to ~0 brightness


def _star_intensities(magnitudes: np.ndarray, cfg: _RenderConfig) -> np.ndarray:
    """Map stellar magnitudes to a relative brightness in [0, 1].

    Magnitudes are inverse-log: a 0-mag star is much brighter than a 6-mag
    one. We use Pogson's law (intensity ratio = 100**(-mag/5)) and then
    rescale to a useful display dynamic range.
    """
    rel = 100.0 ** ((cfg.mag_brightness_floor - magnitudes) / 5.0)
    # Clip the long upper tail so a single Sirius doesn't blow the histogram.
    rel = np.clip(rel, 0.0, 5.0)
    return rel


def _star_sigmas(magnitudes: np.ndarray, cfg: _RenderConfig) -> np.ndarray:
    """Bright stars get a wider PSF (bloom)."""
    t = np.clip(
        (cfg.mag_brightness_ceiling - magnitudes)
        / max(1e-6, cfg.mag_brightness_ceiling - cfg.mag_brightness_floor),
        0.0,
        1.0,
    )
    return cfg.psf_min_sigma + t * (cfg.psf_base_sigma - cfg.psf_min_sigma)


def _splat_gaussian(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    intensity: float,
) -> None:
    """Add a 2D Gaussian centered at (cx, cy) into the canvas in place."""
    h, w = canvas.shape
    half = int(np.ceil(3.5 * sigma))
    x0 = max(0, int(np.floor(cx)) - half)
    x1 = min(w, int(np.ceil(cx)) + half + 1)
    y0 = max(0, int(np.floor(cy)) - half)
    y1 = min(h, int(np.ceil(cy)) + half + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    canvas[y0:y1, x0:x1] += intensity * g


def _add_sky_gradient(
    canvas: np.ndarray,
    rng: np.random.Generator,
    strength: float = 0.08,
) -> np.ndarray:
    """Smooth linear gradient simulating light pollution / moonlight."""
    h, w = canvas.shape
    angle = rng.uniform(0.0, 2.0 * np.pi)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.cos(angle) * (xs / w - 0.5) + np.sin(angle) * (ys / h - 0.5)
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    return canvas + strength * field


def render_star_field(
    ra_center: float,
    dec_center: float,
    field_width: float,
    rotation: float,
    catalog: pd.DataFrame,
    *,
    image_size: int = 224,
    noise_level: float = 0.02,
    sky_gradient: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Render one synthetic star-field image.

    Parameters
    ----------
    ra_center, dec_center : float
        Pointing in degrees.
    field_width : float
        Angular width of the *square* image in degrees.
    rotation : float
        Field rotation in degrees (counter-clockwise on the image).
    catalog : DataFrame
        Output of :func:`src.data.catalog.load_hyg_catalog`. Must have
        ``ra_deg``, ``dec_deg``, ``mag`` columns.
    image_size : int
        Output side length in pixels.
    noise_level : float
        Std of Gaussian readout noise (Poisson shot noise added separately).
    sky_gradient : bool
        Add a smooth light-pollution gradient.
    rng : np.random.Generator
        Reproducibility hook.

    Returns
    -------
    np.ndarray of shape ``(image_size, image_size, 3)``, dtype float32, in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = _RenderConfig(
        image_size=image_size,
        noise_level=noise_level,
        sky_gradient=sky_gradient,
        psf_base_sigma=2.2,
        psf_min_sigma=0.7,
        mag_brightness_floor=-1.5,
        mag_brightness_ceiling=8.0,
    )

    # 1. Coarse pre-filter: keep stars within a generous box around the pointing.
    #    Gnomonic projection is exact for the subsequent step; this is just
    #    to make the projection itself cheap.
    half = field_width * 0.75
    dec_min = max(-90.0, dec_center - half)
    dec_max = min(90.0, dec_center + half)
    sel = (catalog["dec_deg"] >= dec_min) & (catalog["dec_deg"] <= dec_max)

    # RA wrap-around: only filter when the box doesn't straddle 0/360.
    cos_dec = max(0.1, np.cos(np.deg2rad(dec_center)))
    ra_pad = half / cos_dec
    if dec_max < 89.0 and dec_min > -89.0 and ra_pad < 170.0:
        ra_lo = (ra_center - ra_pad) % 360.0
        ra_hi = (ra_center + ra_pad) % 360.0
        if ra_lo < ra_hi:
            sel &= (catalog["ra_deg"] >= ra_lo) & (catalog["ra_deg"] <= ra_hi)
        else:
            sel &= (catalog["ra_deg"] >= ra_lo) | (catalog["ra_deg"] <= ra_hi)
    visible = catalog[sel]

    # 2. Project to tangent plane.
    x_deg, y_deg = gnomonic_project(
        visible["ra_deg"].to_numpy(),
        visible["dec_deg"].to_numpy(),
        ra_center, dec_center,
    )
    keep = np.isfinite(x_deg) & np.isfinite(y_deg)
    x_deg, y_deg = x_deg[keep], y_deg[keep]
    mags = visible["mag"].to_numpy()[keep]

    # 3. Field rotation + scale to pixel space.
    theta = np.deg2rad(rotation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot = cos_t * x_deg - sin_t * y_deg
    y_rot = sin_t * x_deg + cos_t * y_deg

    pixels_per_deg = image_size / field_width
    cx = image_size / 2.0 + x_rot * pixels_per_deg
    cy = image_size / 2.0 - y_rot * pixels_per_deg  # flip y for image axes

    in_frame = (cx >= -8) & (cx < image_size + 8) & (cy >= -8) & (cy < image_size + 8)
    cx, cy, mags = cx[in_frame], cy[in_frame], mags[in_frame]

    # 4. Splat.
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    intensities = _star_intensities(mags, cfg)
    sigmas = _star_sigmas(mags, cfg)
    for x_i, y_i, sig, inten in zip(cx, cy, sigmas, intensities):
        _splat_gaussian(canvas, float(x_i), float(y_i), float(sig), float(inten))

    # 5. Sky background + noise.
    if sky_gradient:
        canvas = _add_sky_gradient(canvas, rng, strength=0.08)
    # Poisson shot noise (scaled so dim stars still get some grain).
    photon_scale = 80.0
    canvas = rng.poisson(np.maximum(canvas, 0.0) * photon_scale).astype(np.float32) / photon_scale
    # Gaussian readout noise.
    canvas += rng.normal(0.0, noise_level, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas, 0.0, None)

    # 6. Stretch to [0, 1] and convert to RGB. We tint slightly blue to
    #    look like a real night sky on screen.
    canvas = canvas / max(canvas.max(), 1e-6)
    rgb = np.stack([canvas * 0.85, canvas * 0.9, canvas * 1.0], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
