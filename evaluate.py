#!/usr/bin/env python
"""CLI evaluation: run a trained checkpoint on a test set and report metrics.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml \
                       --test-dir data/test_images --report-dir reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.augmentations import build_eval_transforms
from src.data.catalog import load_hyg_catalog
from src.data.dataset import AstrometryNetDataset, SyntheticStarFieldDataset
from src.models.astrolocnet import AstroLocNet
from src.training.metrics import MetricsAccumulator
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AstroLocNet.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--test-dir", type=Path, default=None,
                   help="Directory of real test images + .json calibrations. "
                        "If omitted, uses a synthetic validation set.")
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--device", default="auto")
    p.add_argument("--samples", type=int, default=512,
                   help="Synthetic-eval sample count (ignored if --test-dir).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )

    model = AstroLocNet(pretrained=False)
    Trainer.load_state(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    eval_tfm = build_eval_transforms(cfg["data"]["image_size"])
    if args.test_dir:
        ds = AstrometryNetDataset(args.test_dir, image_size=cfg["data"]["image_size"], transform=eval_tfm)
        if len(ds) == 0:
            print(f"[evaluate.py] No images found in {args.test_dir}")
            sys.exit(1)
        source = f"real:{args.test_dir}"
    else:
        catalog = load_hyg_catalog(cfg["data"]["catalog_path"], mag_limit=cfg["data"]["catalog_mag_limit"])
        ds = SyntheticStarFieldDataset(
            catalog, n_samples=args.samples,
            image_size=cfg["data"]["image_size"],
            field_width_range=tuple(cfg["data"]["field_width_range"]),
            noise_level=cfg["data"]["noise_level"],
            sky_gradient=cfg["data"]["sky_gradient"],
            transform=eval_tfm,
            seed=99999,
        )
        source = f"synthetic:{args.samples}"

    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
    acc = MetricsAccumulator()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device); labels = labels.to(device)
            preds = model(images)
            # Compute per-sample loss for record-keeping.
            acc.update(preds, labels, 0.0)
    metrics = acc.compute()
    metrics["source"] = source
    metrics["n_samples"] = len(ds)

    print(json.dumps(metrics, indent=2))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.report_dir / f"eval_{source.replace('/', '_').replace(':', '_')}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate.py] Wrote {out_path}")


if __name__ == "__main__":
    main()
