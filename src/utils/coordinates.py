"""Spherical-coordinate utilities.

Gnomonic (tangent-plane) projection is the right model for small-FOV
astrophotography because it preserves great circles as straight lines
through the tangent point. We use it for both rendering (sky -> pixel)
and inverse projection (pixel -> sky) in the classical solver.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Gnomonic projection
# --------------------------------------------------------------------------- #


def gnomonic_project(
    ra: np.ndarray,
    dec: np.ndarray,
    ra0: float,
    dec0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (ra, dec) onto the tangent plane centered at (ra0, dec0).

    All inputs in degrees; outputs in degrees on the tangent plane.
    Points on the back hemisphere (cos_c <= 0) are returned as NaN so
    the caller can filter them out.

    Reference: Calabretta & Greisen 2002, "Representations of celestial
    coordinates in FITS", section 5.1.3.
    """
    ra_r, dec_r, ra0_r, dec0_r = (np.deg2rad(np.asarray(a)) for a in (ra, dec, ra0, dec0))

    cos_c = (
        np.sin(dec0_r) * np.sin(dec_r)
        + np.cos(dec0_r) * np.cos(dec_r) * np.cos(ra_r - ra0_r)
    )

    # Avoid div-by-zero; mask back hemisphere.
    safe = cos_c > 1e-6
    x = np.where(
        safe,
        np.cos(dec_r) * np.sin(ra_r - ra0_r) / np.where(safe, cos_c, 1.0),
        np.nan,
    )
    y = np.where(
        safe,
        (
            np.cos(dec0_r) * np.sin(dec_r)
            - np.sin(dec0_r) * np.cos(dec_r) * np.cos(ra_r - ra0_r)
        )
        / np.where(safe, cos_c, 1.0),
        np.nan,
    )
    return np.rad2deg(x), np.rad2deg(y)


def gnomonic_unproject(
    x: np.ndarray,
    y: np.ndarray,
    ra0: float,
    dec0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse: (x, y) on tangent plane (deg) -> (ra, dec) on sphere (deg)."""
    x_r, y_r, ra0_r, dec0_r = (np.deg2rad(np.asarray(a)) for a in (x, y, ra0, dec0))
    rho = np.hypot(x_r, y_r)
    # When rho == 0 we're at the tangent point itself.
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    safe = rho > 1e-12
    dec_r = np.where(
        safe,
        np.arcsin(
            cos_c * np.sin(dec0_r)
            + (np.where(safe, y_r, 0.0) * sin_c * np.cos(dec0_r))
            / np.where(safe, rho, 1.0)
        ),
        dec0_r,
    )
    ra_r = ra0_r + np.arctan2(
        x_r * sin_c,
        rho * np.cos(dec0_r) * cos_c - y_r * np.sin(dec0_r) * sin_c,
    )
    return np.rad2deg(ra_r) % 360.0, np.rad2deg(dec_r)


# --------------------------------------------------------------------------- #
# Spherical metric helpers
# --------------------------------------------------------------------------- #


def angular_separation_deg(
    ra1: np.ndarray | float,
    dec1: np.ndarray | float,
    ra2: np.ndarray | float,
    dec2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle separation in degrees. Numerically stable form."""
    ra1, dec1, ra2, dec2 = (np.deg2rad(np.asarray(a)) for a in (ra1, dec1, ra2, dec2))
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    # Haversine
    a = np.sin(d_dec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(d_ra / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(a)))


def wrap_ra_deg(ra: np.ndarray | float) -> np.ndarray | float:
    """Wrap right ascension into [0, 360)."""
    return np.mod(ra, 360.0)


def clamp_dec_deg(dec: np.ndarray | float) -> np.ndarray | float:
    """Clamp declination to [-90, 90]."""
    return np.clip(dec, -90.0, 90.0)
