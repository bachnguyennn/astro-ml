"""Inference-time plots used in notebooks 08 and 09."""

from __future__ import annotations

import io
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image

from src.utils.coordinates import gnomonic_project


# --------------------------------------------------------------------------- #
# Constellation overlay
# --------------------------------------------------------------------------- #


def plot_constellation_overlay(
    image_rgb: np.ndarray,
    *,
    ra_center: float,
    dec_center: float,
    field_width_deg: float,
    rotation_deg: float = 0.0,
    catalog=None,
    mag_limit: float = 6.0,
    title: Optional[str] = None,
) -> Figure:
    """Overlay catalog stars projected onto the image.

    If ``catalog`` is None, only the field-center marker is drawn.
    """
    h, w = image_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0b0d17")
    ax.set_facecolor("#0b0d17")
    ax.imshow(image_rgb)
    ax.set_xticks([]); ax.set_yticks([])

    if catalog is not None:
        # Filter to bright stars, project, overlay.
        bright = catalog[catalog["mag"] <= mag_limit]
        x_deg, y_deg = gnomonic_project(
            bright["ra_deg"].to_numpy(),
            bright["dec_deg"].to_numpy(),
            ra_center, dec_center,
        )
        theta = np.deg2rad(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = cos_t * x_deg - sin_t * y_deg
        y_rot = sin_t * x_deg + cos_t * y_deg
        pix_per_deg = w / field_width_deg
        cx = w / 2.0 + x_rot * pix_per_deg
        cy = h / 2.0 - y_rot * pix_per_deg
        in_frame = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h) & np.isfinite(cx) & np.isfinite(cy)
        cx, cy = cx[in_frame], cy[in_frame]
        ax.scatter(cx, cy, s=18, facecolors="none", edgecolors="#7c5cff", linewidths=1.0, alpha=0.85)

    # Center crosshair.
    cx, cy = w / 2.0, h / 2.0
    ax.plot([cx - 15, cx + 15], [cy, cy], color="#ffd166", linewidth=1.2)
    ax.plot([cx, cx], [cy - 15, cy + 15], color="#ffd166", linewidth=1.2)

    if title:
        ax.text(8, 22, title, color="#e8e8ff", fontsize=11,
                bbox=dict(facecolor="#161a2e", edgecolor="none", pad=4))
    fig.tight_layout(pad=0)
    return fig


# --------------------------------------------------------------------------- #
# Optional world map (uses matplotlib, no Plotly dep here)
# --------------------------------------------------------------------------- #


def plot_world_map(
    lat: float, lon: float, *,
    lat_uncertainty_deg: float = 10.0,
    lon_uncertainty_deg: float = 10.0,
    title: Optional[str] = None,
) -> Figure:
    """Simple cartesian world map sketch with a marker + uncertainty halo.

    Notebook 09 uses this for the end-to-end demo. For richer maps,
    swap in plotly.go.Scattergeo.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0b0d17")
    ax.set_facecolor("#0b0d17")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.grid(True, color="#2a2f4f", alpha=0.4)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(colors="#9aa0b4")
    for spine in ax.spines.values():
        spine.set_color("#2a2f4f")

    # Halo
    n = 120
    a = np.linspace(0, 2 * np.pi, n)
    ax.fill(lon + lon_uncertainty_deg * np.cos(a),
            lat + lat_uncertainty_deg * np.sin(a),
            color="#7c5cff", alpha=0.22)
    ax.plot(lon, lat, marker="*", markersize=18,
            markerfacecolor="#ffd166", markeredgecolor="white")
    ax.set_xlabel("Longitude (°)", color="#9aa0b4")
    ax.set_ylabel("Latitude (°)", color="#9aa0b4")
    if title:
        ax.set_title(title, color="#e8e8ff")
    fig.tight_layout()
    return fig
