#!/usr/bin/env python
"""Render the figures the README links to.

Executes the key cells from notebooks 01-07 against the smoke checkpoint
so the README screenshots are real outputs from the codebase rather
than placeholders. Safe to re-run; idempotent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.augmentations import build_eval_transforms, build_train_transforms
from src.data.catalog import load_hyg_catalog
from src.data.renderer import render_star_field, sample_random_pointing
from src.inference.predict import load_model, predict_image
from src.inference.visualize import plot_constellation_overlay
from src.models.astrolocnet import AstroLocNet, freeze_backbone, unfreeze_last_n_blocks
from src.training.loss import angular_separation_loss
from src.training.metrics import angular_separation_deg_torch
from src.training.trainer import Trainer
from src.utils.coordinates import angular_separation_deg, gnomonic_project


FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


def header(name: str) -> None:
    print(f"\n=== {name} ===")


# --------------------------------------------------------------------------- #
# 01 — Catalog statistics + sky coverage
# --------------------------------------------------------------------------- #


def fig_catalog(catalog: pd.DataFrame) -> None:
    header("01: catalog stats")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(catalog["mag"], bins=60, color="#7c5cff", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Visual magnitude"); ax.set_ylabel("Star count")
    ax.set_title("HYG catalog magnitude distribution (mag ≤ 8.0)")
    fig.tight_layout(); fig.savefig(FIG / "01_mag_distribution.png"); plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide")
    ra = np.deg2rad(catalog["ra_deg"].to_numpy() - 180.0)
    dec = np.deg2rad(catalog["dec_deg"].to_numpy())
    size = np.clip(8.0 - catalog["mag"], 0.2, 6.0)
    ax.scatter(ra, dec, s=size, c="white", alpha=0.65, edgecolors="none")
    ax.set_facecolor("#0b0d17")
    ax.grid(True, color="#2a2f4f", alpha=0.4)
    ax.set_title("Sky coverage of HYG stars (Mollweide projection)")
    fig.tight_layout(); fig.savefig(FIG / "01_sky_coverage.png", facecolor="white"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 02 — Renderer
# --------------------------------------------------------------------------- #


def fig_renderer(catalog: pd.DataFrame) -> None:
    header("02: renderer")
    # FOV grid
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor="#0b0d17")
    for ax, fw in zip(axes, [15, 30, 50, 80]):
        img = render_star_field(85, -2, fw, 0, catalog, image_size=224,
                                rng=np.random.default_rng(int(fw)))
        ax.imshow(img); ax.set_title(f"FOV {fw}°", color="white")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "02_fov_grid.png", facecolor="#0b0d17"); plt.close(fig)

    # Augmentation grid
    tfm = build_train_transforms(224)
    base = render_star_field(85, -2, 30, 0, catalog, image_size=224, rng=np.random.default_rng(7))
    base_pil = Image.fromarray((base * 255).astype(np.uint8))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), facecolor="#0b0d17")
    for ax in axes.flatten():
        t = tfm(base_pil)
        img = (t.permute(1, 2, 0).numpy() * np.array([0.15, 0.15, 0.20])
               + np.array([0.10, 0.10, 0.15]))
        ax.imshow(np.clip(img, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Augmentation samples (180° rotation + flips + jitter + blur)", color="white")
    fig.tight_layout(); fig.savefig(FIG / "02_augmentation_grid.png", facecolor="#0b0d17"); plt.close(fig)

    # Gnomonic grid demo
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, dec0 in zip(axes, [0.0, 45.0, 80.0]):
        ras, decs = np.meshgrid(np.arange(-30, 31, 5), np.arange(-30, 31, 5))
        xs, ys = gnomonic_project(ras + 180.0, decs + dec0, 180.0, dec0)
        ax.plot(xs, ys, color="#7c5cff", linewidth=0.8)
        ax.plot(xs.T, ys.T, color="#7c5cff", linewidth=0.8)
        ax.set_title(f"Tangent plane @ Dec={dec0:.0f}°"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIG / "02_gnomonic_grid.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 03 — Architecture / freezing
# --------------------------------------------------------------------------- #


def fig_architecture() -> None:
    header("03: architecture")
    model = AstroLocNet(pretrained=False)

    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    snapshots = {}
    snapshots["Phase 0 (all unfrozen)"] = count_trainable(model)
    freeze_backbone(model)
    snapshots["Phase 1 (head only)"] = count_trainable(model)
    unfreeze_last_n_blocks(model, n=3)
    snapshots["Phase 2 (last 3 blocks + head)"] = count_trainable(model)

    fig, ax = plt.subplots(figsize=(8, 4))
    names, counts = zip(*snapshots.items())
    ax.barh(names, [c / 1e6 for c in counts], color="#7c5cff")
    ax.set_xlabel("Trainable params (millions)")
    ax.set_title("Trainable parameters per training phase")
    for i, c in enumerate(counts):
        ax.text(c / 1e6, i, f"  {c:,}", va="center")
    fig.tight_layout(); fig.savefig(FIG / "03_trainable_per_phase.png"); plt.close(fig)

    # freezing map
    phases = ["phase1", "phase2", "phase3"]
    matrix = np.zeros((len(phases), len(model.backbone.features)), dtype=bool)
    for row, phase in enumerate(phases):
        freeze_backbone(model)
        if phase != "phase1":
            unfreeze_last_n_blocks(model, n=3)
        for idx, block in enumerate(model.backbone.features):
            matrix[row, idx] = any(p.requires_grad for p in block.parameters())
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(matrix.astype(float), aspect="auto", cmap="Purples")
    ax.set_yticks(range(len(phases))); ax.set_yticklabels(phases)
    ax.set_xticks(range(len(model.backbone.features)))
    ax.set_xticklabels([f"block_{i}" for i in range(len(model.backbone.features))], rotation=45)
    ax.set_title("Which backbone blocks train per phase (purple = trainable)")
    fig.tight_layout(); fig.savefig(FIG / "03_freezing_map.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 04 — Loss function
# --------------------------------------------------------------------------- #


def fig_loss() -> None:
    header("04: loss")
    ra_truth = 1.0
    ra_preds = np.linspace(0, 360, 361)
    mse = (ra_preds - ra_truth) ** 2
    ang = angular_separation_deg(ra_preds, 0, ra_truth, 0)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(ra_preds, mse, color="#f56565", label="MSE (RA - RA_true)²")
    ax2.plot(ra_preds, ang, color="#3ad29f", label="Great-circle separation (°)")
    ax1.set_xlabel("Predicted RA (°)")
    ax1.set_ylabel("MSE", color="#f56565"); ax2.set_ylabel("Great-circle Δ (°)", color="#3ad29f")
    ax1.set_title("Truth RA=1°. MSE wrongly penalizes RA=359° as a 358° error.")
    fig.tight_layout(); fig.savefig(FIG / "04_mse_vs_angular.png"); plt.close(fig)

    ra_grid, dec_grid = np.meshgrid(np.linspace(0, 360, 181), np.linspace(-90, 90, 91))
    loss = angular_separation_deg(ra_grid, dec_grid, 120, 30)
    fig, ax = plt.subplots(figsize=(8, 4))
    c = ax.pcolormesh(ra_grid, dec_grid, loss, cmap="viridis", shading="auto")
    ax.scatter([120], [30], c="red", s=80, marker="*", label="True position")
    fig.colorbar(c, ax=ax, label="Great-circle separation (°)")
    ax.set_xlabel("RA (°)"); ax.set_ylabel("Dec (°)"); ax.legend()
    ax.set_title("Spherical loss landscape — no wrap-around discontinuity")
    fig.tight_layout(); fig.savefig(FIG / "04_loss_landscape.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 05 — Training curves (from the saved smoke checkpoint)
# --------------------------------------------------------------------------- #


def fig_training_curves() -> None:
    header("05: training curves")
    # Prefer the checkpoint with the most complete history.
    candidates = sorted(
        (ROOT / "checkpoints").glob("last_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        ckpt = candidates[-1]
    else:
        ckpt = ROOT / "checkpoints" / "best.pt"
    if not ckpt.exists():
        print("  No checkpoint found; skipping. Run train.py first.")
        return
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    history = state.get("history", [])
    if not history:
        print("  History empty; skipping.")
        return
    rows = []
    for h in history:
        rows.append({"phase": h["phase"], "epoch": h["epoch"],
                     "train_loss": h["train"]["loss"], "val_loss": h["val"]["loss"],
                     "val_sep_deg": h["val"]["ang_sep_mean_deg"]})
    df = pd.DataFrame(rows); df["x"] = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["x"], df["train_loss"], label="train", color="#7c5cff", marker="o")
    axes[0].plot(df["x"], df["val_loss"], label="val", color="#ffd166", marker="s")
    axes[0].set_title("Loss across phases (smoke run)")
    axes[0].set_xlabel("epoch index"); axes[0].legend()
    axes[1].plot(df["x"], df["val_sep_deg"], color="#3ad29f", marker="o")
    axes[1].set_title("Val angular separation (°)")
    axes[1].set_xlabel("epoch index")
    fig.tight_layout(); fig.savefig(FIG / "05_training_curves.png"); plt.close(fig)


# --------------------------------------------------------------------------- #
# 09 — End-to-end demo (constellation overlay)
# --------------------------------------------------------------------------- #


def fig_demo(catalog: pd.DataFrame) -> None:
    header("09: end-to-end demo")
    ckpt = ROOT / "checkpoints" / "best.pt"
    if not ckpt.exists():
        print("  No checkpoint; skipping demo overlay.")
        return
    model = load_model(ckpt, device="cpu")
    rng = np.random.default_rng(11)
    for case in range(3):
        ra, dec, rot, fw = sample_random_pointing(rng, (25.0, 55.0))
        img = render_star_field(ra, dec, fw, rot, catalog, image_size=224,
                                rng=np.random.default_rng(case))
        img_u8 = (img * 255).astype(np.uint8)
        pred = predict_image(model, img_u8)
        fig1 = plot_constellation_overlay(
            img_u8,
            ra_center=pred.ra_deg, dec_center=pred.dec_deg,
            field_width_deg=pred.field_width_deg, rotation_deg=pred.rotation_deg,
            catalog=catalog, mag_limit=5.5,
            title=f"Pred (RA, Dec)=({pred.ra_deg:.1f}°, {pred.dec_deg:.1f}°)  "
                  f"vs truth ({ra:.1f}, {dec:.1f})",
        )
        fig1.savefig(FIG / f"09_demo_overlay_{case}.png", facecolor="#0b0d17")
        plt.close(fig1)


# --------------------------------------------------------------------------- #
# Drive
# --------------------------------------------------------------------------- #


def main() -> None:
    t0 = time.time()
    catalog = load_hyg_catalog(ROOT / "data/catalogs/hygdata_v3.csv", mag_limit=8.0)
    print(f"Loaded {len(catalog):,} stars from HYG catalog.")
    fig_catalog(catalog)
    fig_renderer(catalog)
    fig_architecture()
    fig_loss()
    fig_training_curves()
    fig_demo(catalog)
    print(f"\nAll figures saved to {FIG} ({time.time()-t0:.1f}s).")


if __name__ == "__main__":
    main()
