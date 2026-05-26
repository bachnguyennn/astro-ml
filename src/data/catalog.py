"""HYG catalog loader.

The HYG v3 CSV ships with one row per star. We only need a handful of
columns for rendering and we filter by magnitude to drop stars that
would be invisible to a consumer camera.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Columns we care about. HYG v3 actually uses these names; if upstream
# renames anything we fall back to numeric column inference.
REQUIRED_COLS = ("ra", "dec", "mag")


def load_hyg_catalog(
    path: str | Path,
    *,
    mag_limit: Optional[float] = 8.0,
    drop_sun: bool = True,
) -> pd.DataFrame:
    """Load the HYG CSV and return a tidy ``DataFrame``.

    The HYG file stores RA in hours [0, 24) — we convert to degrees
    [0, 360) so the rest of the pipeline can stay in degrees.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"HYG catalog not found at {p}. See data/catalogs/README.md "
            "for the download command."
        )

    df = pd.read_csv(p, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        raise ValueError(
            f"HYG CSV at {p} is missing required columns: {missing}. "
            f"Got columns: {list(df.columns)[:20]}..."
        )

    out = pd.DataFrame({
        "ra_deg": df[cols["ra"]].astype(float) * 15.0,   # hours -> degrees
        "dec_deg": df[cols["dec"]].astype(float),
        "mag": df[cols["mag"]].astype(float),
    })
    # Optional color index for future use.
    if "ci" in cols:
        out["ci"] = pd.to_numeric(df[cols["ci"]], errors="coerce")

    # Row 0 of HYG v3 is the Sun (id=0). Drop it for night-sky rendering.
    if drop_sun:
        sun_mask = (out["ra_deg"] == 0) & (out["dec_deg"] == 0) & (out["mag"] < -20)
        out = out[~sun_mask].reset_index(drop=True)

    out = filter_catalog(out, mag_limit=mag_limit)
    return out


def filter_catalog(
    catalog: pd.DataFrame,
    *,
    mag_limit: Optional[float] = 8.0,
) -> pd.DataFrame:
    """Drop stars dimmer than ``mag_limit`` and any rows with NaN mag/ra/dec."""
    df = catalog.dropna(subset=["ra_deg", "dec_deg", "mag"])
    if mag_limit is not None:
        df = df[df["mag"] <= mag_limit]
    df = df[(df["dec_deg"] >= -90) & (df["dec_deg"] <= 90)]
    df["ra_deg"] = np.mod(df["ra_deg"], 360.0)
    return df.reset_index(drop=True)
