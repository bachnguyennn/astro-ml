"""Training and evaluation metrics.

Operates on the **decoded** prediction tensor (4-column degree values)
so the numbers stay comparable to the legacy raw-output model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from src.models.astrolocnet import AstroLocNet


def _maybe_decode(pred: torch.Tensor) -> torch.Tensor:
    """Decode 7-D sin/cos outputs to 4-D degree outputs (no-op if already 4-D)."""
    if pred.shape[-1] == AstroLocNet.OUTPUT_DIM:
        return AstroLocNet.decode_predictions(pred)
    return pred


def angular_separation_deg_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample great-circle separation in degrees. Returns ``[B]`` tensor."""
    pred = _maybe_decode(pred)
    ra_p = torch.deg2rad(pred[:, 0])
    dec_p = torch.deg2rad(pred[:, 1])
    ra_t = torch.deg2rad(target[:, 0])
    dec_t = torch.deg2rad(target[:, 1])
    d_ra = ra_p - ra_t
    d_dec = dec_p - dec_t
    a = torch.sin(d_dec * 0.5) ** 2 + torch.cos(dec_t) * torch.cos(dec_p) * torch.sin(d_ra * 0.5) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    return torch.rad2deg(2.0 * torch.asin(torch.sqrt(a)))


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_dec = _maybe_decode(pred)
    sep = angular_separation_deg_torch(pred_dec, target)

    # Rotation MAE with wrap-around.
    rot_diff = (pred_dec[:, 2] - target[:, 2]) % 360.0
    rot_diff = torch.minimum(rot_diff, 360.0 - rot_diff)

    scale_mae = torch.abs(torch.exp(pred_dec[:, 3]) - torch.exp(target[:, 3]))

    return {
        "ang_sep_mean_deg": float(sep.mean().item()),
        "ang_sep_median_deg": float(sep.median().item()),
        "pct_within_5_deg": float((sep <= 5.0).float().mean().item() * 100.0),
        "pct_within_1_deg": float((sep <= 1.0).float().mean().item() * 100.0),
        "rotation_mae_deg": float(rot_diff.mean().item()),
        "scale_mae_deg": float(scale_mae.mean().item()),
    }


@dataclass
class MetricsAccumulator:
    """Accumulates per-batch tensors and reduces at .compute() time."""
    preds: List[torch.Tensor] = field(default_factory=list)
    targets: List[torch.Tensor] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float) -> None:
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
        self.losses.append(float(loss))

    def compute(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        all_pred = torch.cat(self.preds, dim=0)
        all_target = torch.cat(self.targets, dim=0)
        metrics = compute_metrics(all_pred, all_target)
        metrics["loss"] = float(np.mean(self.losses))
        return metrics

    def reset(self) -> None:
        self.preds.clear()
        self.targets.clear()
        self.losses.clear()
