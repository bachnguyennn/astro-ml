"""Angular-separation loss for spherical regression.

MSE on raw RA/Dec is wrong because:

1. **RA wraps at 0°/360°.** MSE thinks RA=1° and RA=359° are 358° apart
   when they're really 2° apart on the sphere.
2. **The metric is spherical, not Euclidean.** Two points near the pole
   with very different RA values can still be very close in
   great-circle distance.

This loss uses the haversine form of the great-circle distance — the
numerically stable version. Rotation uses a circular ``1 - cos`` loss,
and log-scale uses plain MSE (it's already in a flat metric space).
"""

from __future__ import annotations

import torch
from torch import nn


def angular_separation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    rotation_weight: float = 0.1,
    scale_weight: float = 0.1,
) -> torch.Tensor:
    """Great-circle loss on (RA, Dec) + circular rotation + MSE on log-scale.

    Both ``pred`` and ``target`` are ``[B, 4]`` tensors with columns
    ``[ra_deg, dec_deg, rotation_deg, log_field_width]``.
    """
    ra_pred = torch.deg2rad(pred[:, 0])
    dec_pred = torch.deg2rad(pred[:, 1])
    ra_true = torch.deg2rad(target[:, 0])
    dec_true = torch.deg2rad(target[:, 1])

    # Haversine — numerically stable near antipode and at small separations.
    d_ra = ra_pred - ra_true
    d_dec = dec_pred - dec_true
    a = torch.sin(d_dec * 0.5) ** 2 + torch.cos(dec_true) * torch.cos(dec_pred) * torch.sin(d_ra * 0.5) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    angular_sep_rad = 2.0 * torch.asin(torch.sqrt(a))

    # Circular rotation loss in [0, 2]; 0 when aligned.
    rot_diff = torch.deg2rad(pred[:, 2] - target[:, 2])
    rot_loss = 1.0 - torch.cos(rot_diff)

    # log_scale is metric, plain squared error.
    scale_loss = (pred[:, 3] - target[:, 3]) ** 2

    return torch.mean(angular_sep_rad) + rotation_weight * torch.mean(rot_loss) + scale_weight * torch.mean(scale_loss)


class AstroLocLoss(nn.Module):
    """nn.Module wrapper for use in Trainer/optim chains."""

    def __init__(self, rotation_weight: float = 0.1, scale_weight: float = 0.1):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.scale_weight = scale_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return angular_separation_loss(
            pred, target,
            rotation_weight=self.rotation_weight,
            scale_weight=self.scale_weight,
        )
