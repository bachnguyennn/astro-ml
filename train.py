#!/usr/bin/env python
"""CLI training entrypoint.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --smoke
    python train.py --config configs/default.yaml --skip-phase3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Local imports
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.augmentations import build_eval_transforms, build_train_transforms
from src.data.catalog import load_hyg_catalog
from src.data.dataset import AstrometryNetDataset, SyntheticStarFieldDataset
from src.models.astrolocnet import AstroLocNet
from src.training.trainer import PhaseConfig, Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AstroLocNet.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny dataset + 1 epoch per phase; CPU-friendly sanity test.")
    p.add_argument("--skip-phase3", action="store_true",
                   help="Skip the real-image fine-tuning phase.")
    p.add_argument("--device", default=None, help="Override config device (cpu/cuda/mps/auto).")
    p.add_argument("--train-samples", type=int, default=None,
                   help="Override data.train_samples (e.g. 5000 for fast Colab runs).")
    p.add_argument("--val-samples", type=int, default=None,
                   help="Override data.val_samples.")
    p.add_argument("--epochs-phase1", type=int, default=None)
    p.add_argument("--epochs-phase2", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override data.num_workers (Colab T4 = 2 vCPUs; try 2).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainer_config(cfg: dict, *, smoke: bool, device_override: str | None) -> TrainerConfig:
    train_cfg = cfg["train"]
    phase_cfg = lambda key: PhaseConfig(**train_cfg[key])
    tc = TrainerConfig(
        batch_size=train_cfg["batch_size"],
        amp=train_cfg.get("amp", True),
        device=device_override or train_cfg.get("device", "auto"),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
        rotation_weight=cfg["loss"]["rotation_weight"],
        scale_weight=cfg["loss"]["scale_weight"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        log_dir=cfg["output"]["log_dir"],
        best_name=cfg["output"]["best_name"],
        phase1=phase_cfg("phase1"),
        phase2=phase_cfg("phase2"),
        phase3=phase_cfg("phase3"),
    )
    if smoke:
        tc.phase1.epochs = 1
        tc.phase2.epochs = 1
        tc.phase3.epochs = 1
        tc.batch_size = 16
    return tc


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg.get("seed", 42))

    trainer_cfg = build_trainer_config(cfg, smoke=args.smoke, device_override=args.device)
    # CLI overrides (handy on Colab where you want to shrink for speed).
    if args.epochs_phase1 is not None: trainer_cfg.phase1.epochs = args.epochs_phase1
    if args.epochs_phase2 is not None: trainer_cfg.phase2.epochs = args.epochs_phase2

    # Data
    catalog = load_hyg_catalog(
        cfg["data"]["catalog_path"],
        mag_limit=cfg["data"]["catalog_mag_limit"],
    )

    train_n = args.train_samples or (cfg["data"]["train_samples"] if not args.smoke else 256)
    val_n = args.val_samples or (cfg["data"]["val_samples"] if not args.smoke else 64)

    train_tfm = build_train_transforms(cfg["data"]["image_size"])
    eval_tfm = build_eval_transforms(cfg["data"]["image_size"])

    train_ds = SyntheticStarFieldDataset(
        catalog, n_samples=train_n,
        image_size=cfg["data"]["image_size"],
        field_width_range=tuple(cfg["data"]["field_width_range"]),
        noise_level=cfg["data"]["noise_level"],
        sky_gradient=cfg["data"]["sky_gradient"],
        transform=train_tfm,
        seed=cfg.get("seed", 42),
    )
    val_ds = SyntheticStarFieldDataset(
        catalog, n_samples=val_n,
        image_size=cfg["data"]["image_size"],
        field_width_range=tuple(cfg["data"]["field_width_range"]),
        noise_level=cfg["data"]["noise_level"],
        sky_gradient=cfg["data"]["sky_gradient"],
        transform=eval_tfm,
        seed=10_000 + cfg.get("seed", 42),
    )
    n_workers = args.num_workers if args.num_workers is not None else (
        0 if args.smoke else cfg["data"]["num_workers"]
    )
    persistent = n_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=trainer_cfg.batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=trainer_cfg.batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )

    # Model
    model = AstroLocNet(
        pretrained=True,
        dropout_p_1=cfg["model"]["dropout_p_1"],
        dropout_p_2=cfg["model"]["dropout_p_2"],
        hidden_dim=cfg["model"]["hidden_dim"],
    )

    def progress_cb(evt: dict) -> None:
        print(json.dumps(evt, default=str))

    trainer = Trainer(model, trainer_cfg, progress_cb=progress_cb)

    print(f"[train.py] device={trainer.device} amp={trainer.use_amp} "
          f"train_n={train_n} val_n={val_n}")
    print(f"[train.py] === Phase 1 (head only, backbone frozen) ===")
    trainer.run_phase1(train_loader, val_loader)
    print(f"[train.py] === Phase 2 (unfreeze last "
          f"{trainer_cfg.phase2.unfreeze_last_n_blocks} blocks) ===")
    trainer.run_phase2(train_loader, val_loader)

    if not args.skip_phase3 and cfg.get("real_data", {}).get("enabled", False):
        print(f"[train.py] === Phase 3 (real-image fine-tune) ===")
        real_tfm = build_eval_transforms(cfg["data"]["image_size"])
        real_ds = AstrometryNetDataset(
            cfg["real_data"]["images_dir"],
            image_size=cfg["data"]["image_size"],
            transform=real_tfm,
        )
        if len(real_ds) == 0:
            print("[train.py] No real images found; skipping phase 3.")
        else:
            real_loader = DataLoader(real_ds, batch_size=trainer_cfg.batch_size, shuffle=True)
            trainer.run_phase3(real_loader, val_loader)
    else:
        print("[train.py] Phase 3 skipped (--skip-phase3 or real_data.enabled=false).")

    print(f"[train.py] Best val angular separation: {trainer.best_metric:.4f}°")
    print(f"[train.py] Best checkpoint: {trainer.checkpoint_dir / trainer_cfg.best_name}")


if __name__ == "__main__":
    main()
