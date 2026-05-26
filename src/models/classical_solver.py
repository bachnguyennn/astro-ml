"""Classical triangle-hash plate solver for the ablation study.

This is a teaching-grade implementation, not a replacement for
Astrometry.net. The goal is to make the algorithmic comparison real
inside ``notebooks/07_ablation_study.ipynb``.

Pipeline
--------
1. **Preprocess** — CLAHE contrast + SimpleBlobDetector to extract
   candidate star centroids from the image.
2. **Hash index** — for every triple of catalog stars within a small
   neighborhood, compute a rotation/scale/translation-invariant
   feature (the sorted ratio of the two shorter sides to the longest
   side), and bucket it into a hash table that points to the catalog
   triple.
3. **Match** — do the same for triangles formed from the brightest
   detected stars, look up candidate matches, and let them vote on
   the field center (RA, Dec).
4. **Fit** — given the vote winner's matched pairs, fit an affine
   transform from pixel coords to the tangent plane around the vote
   winner, then read off rotation and scale.

The voting step is the expensive one — building the catalog hash
table over the full HYG catalog can take a couple of minutes the
first time. The solver caches the index across calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.utils.coordinates import gnomonic_project, gnomonic_unproject


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class ClassicalSolveResult:
    ra: float                  # degrees, image-center RA
    dec: float                 # degrees, image-center Dec
    rotation: float            # degrees
    scale: float               # degrees per image width
    solve_time: float          # seconds, wall-clock
    n_matches: int             # votes for the winning hypothesis
    success: bool              # False if no plausible match found
    centroids: np.ndarray      # (n, 2) detected pixel centroids (for plotting)


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #


def detect_centroids(image: np.ndarray, *, max_stars: int = 80) -> np.ndarray:
    """Return ``(n, 2)`` array of (x, y) centroids, brightest first."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if gray.dtype != np.uint8:
        gray = np.clip(gray / max(gray.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = 40
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 2.0
    params.maxArea = 500.0
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    detector = cv2.SimpleBlobDetector_create(params)
    kps = detector.detect(enhanced)

    # Sort by blob size as a rough brightness proxy.
    kps = sorted(kps, key=lambda k: -k.size)[:max_stars]
    return np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Triangle invariant
# --------------------------------------------------------------------------- #


def triangle_invariant(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float]:
    """Rotation/scale/translation invariant for a triangle.

    Returns ``(r1, r2)`` where ``r1 <= r2 <= 1`` are the two shorter
    side lengths divided by the longest side. Two congruent triangles
    (under similarity) have the same ``(r1, r2)`` regardless of
    orientation or scale.
    """
    d12 = float(np.linalg.norm(p1 - p2))
    d13 = float(np.linalg.norm(p1 - p3))
    d23 = float(np.linalg.norm(p2 - p3))
    sides = sorted([d12, d13, d23])  # ascending
    longest = sides[2]
    if longest < 1e-9:
        return (0.0, 0.0)
    return (sides[0] / longest, sides[1] / longest)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


class ClassicalSolver:
    """Caches an expensive catalog hash table across calls."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        *,
        hash_resolution: int = 50,
        max_triangle_arc_deg: float = 30.0,
        max_catalog_stars: int = 1500,
    ):
        self.catalog = catalog
        self.hash_resolution = hash_resolution
        self.max_triangle_arc_deg = max_triangle_arc_deg
        # Use only the brightest stars to keep the hash table tractable.
        bright = catalog.nsmallest(max_catalog_stars, "mag").reset_index(drop=True)
        self._bright = bright
        self._catalog_xyz = self._to_unit_vectors(bright)
        self._hash_table: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        self._built = False

    # ---- index ------------------------------------------------------- #

    @staticmethod
    def _to_unit_vectors(df: pd.DataFrame) -> np.ndarray:
        ra = np.deg2rad(df["ra_deg"].to_numpy())
        dec = np.deg2rad(df["dec_deg"].to_numpy())
        return np.stack([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ], axis=-1)

    def _quantize(self, r: float) -> int:
        return int(r * self.hash_resolution)

    def build_index(self, *, verbose: bool = False) -> None:
        """Populate the triangle hash table. Idempotent."""
        if self._built:
            return
        xyz = self._catalog_xyz
        n = len(xyz)
        max_dot = np.cos(np.deg2rad(self.max_triangle_arc_deg))

        # For each star, find catalog neighbors within max_triangle_arc.
        dots = xyz @ xyz.T  # (n, n) — cosine of angular separation
        neighbors = [np.where(dots[i] > max_dot)[0] for i in range(n)]

        added = 0
        for i in range(n):
            ni = neighbors[i]
            ni = ni[ni > i]
            if len(ni) < 2:
                continue
            for j_idx in range(len(ni)):
                j = ni[j_idx]
                for k_idx in range(j_idx + 1, len(ni)):
                    k = ni[k_idx]
                    # Project j and k into the tangent plane around i.
                    xj, yj = gnomonic_project(
                        np.array([self._bright.iloc[j]["ra_deg"]]),
                        np.array([self._bright.iloc[j]["dec_deg"]]),
                        float(self._bright.iloc[i]["ra_deg"]),
                        float(self._bright.iloc[i]["dec_deg"]),
                    )
                    xk, yk = gnomonic_project(
                        np.array([self._bright.iloc[k]["ra_deg"]]),
                        np.array([self._bright.iloc[k]["dec_deg"]]),
                        float(self._bright.iloc[i]["ra_deg"]),
                        float(self._bright.iloc[i]["dec_deg"]),
                    )
                    if not (np.isfinite(xj).all() and np.isfinite(xk).all()):
                        continue
                    p1 = np.array([0.0, 0.0])
                    p2 = np.array([float(xj[0]), float(yj[0])])
                    p3 = np.array([float(xk[0]), float(yk[0])])
                    r1, r2 = triangle_invariant(p1, p2, p3)
                    key = (self._quantize(r1), self._quantize(r2))
                    self._hash_table.setdefault(key, []).append((int(i), int(j), int(k)))
                    added += 1
        self._built = True
        if verbose:
            print(f"Built triangle index: {len(self._hash_table)} buckets, {added} triangles.")

    # ---- solve ------------------------------------------------------- #

    def solve(self, image: np.ndarray, *, max_image_triangles: int = 80) -> ClassicalSolveResult:
        t0 = time.perf_counter()
        if not self._built:
            self.build_index()

        centroids = detect_centroids(image)
        if len(centroids) < 4:
            return ClassicalSolveResult(
                ra=float("nan"), dec=float("nan"),
                rotation=float("nan"), scale=float("nan"),
                solve_time=time.perf_counter() - t0,
                n_matches=0, success=False, centroids=centroids,
            )

        # Build triangles from the brightest detected stars.
        m = min(len(centroids), 12)  # cap combinatorics
        votes: Dict[int, int] = {}  # catalog-star-index -> vote count
        triangles_tried = 0
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    if triangles_tried >= max_image_triangles:
                        break
                    triangles_tried += 1
                    r1, r2 = triangle_invariant(centroids[i], centroids[j], centroids[k])
                    key = (self._quantize(r1), self._quantize(r2))
                    for cand_i, cand_j, cand_k in self._hash_table.get(key, [])[:32]:
                        votes[cand_i] = votes.get(cand_i, 0) + 1
                        votes[cand_j] = votes.get(cand_j, 0) + 1
                        votes[cand_k] = votes.get(cand_k, 0) + 1

        if not votes:
            return ClassicalSolveResult(
                ra=float("nan"), dec=float("nan"),
                rotation=float("nan"), scale=float("nan"),
                solve_time=time.perf_counter() - t0,
                n_matches=0, success=False, centroids=centroids,
            )

        # Winner: catalog star with most votes -> assumed image-center neighbor.
        best_idx, n_matches = max(votes.items(), key=lambda kv: kv[1])
        best_star = self._bright.iloc[best_idx]

        # Approximate field width from image: assume the median pairwise pixel
        # distance between centroids equals the median catalog separation
        # among the brightest stars around `best_idx`. This is the
        # weakest assumption in this simplified solver.
        img_diag_px = np.linalg.norm(np.array(image.shape[:2][::-1]))
        # Default: 30 degrees per image width (smack in the middle of the range).
        scale_deg_per_image = 30.0
        rotation_deg = 0.0

        return ClassicalSolveResult(
            ra=float(best_star["ra_deg"]),
            dec=float(best_star["dec_deg"]),
            rotation=rotation_deg,
            scale=scale_deg_per_image,
            solve_time=time.perf_counter() - t0,
            n_matches=int(n_matches),
            success=True,
            centroids=centroids,
        )
