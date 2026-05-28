"""Loss function for the 7-output (sin/cos) AstroLocNet head.

The model emits 7 raw values: ``[sin_ra, cos_ra, sin_dec, cos_dec,
sin_rot, cos_rot, log_scale]``. The label tensor is the legacy 4-D
``[ra_deg, dec_deg, rot_deg, log_scale]`` so we don't have to change
the dataset.

We compute loss in two parts:

1. **Direct sin/cos MSE** on the 6 trig outputs vs the encoded
   targets. This is the dominant gradient signal and is smooth
   everywhere — no wrap discontinuity.
2. **Optional angular consistency term** — the great-circle distance
   between *decoded* predictions and ground-truth (RA, Dec). Tiny
   weight; just nudges the geometry while the trig MSE handles most
   of the learning. Helps Phase 2 converge faster.

Plus plain MSE on ``log_scale``.

This is the right setup for spherical regression:
- No discontinuity at 0°/360°.
- Network outputs don't need to lie on the unit circle — ``atan2``
  recovers the angle from any nonzero ``(sin, cos)`` pair.
- Loss landscape is smooth, so gradient descent actually descends.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.astrolocnet import AstroLocNet


def angular_separation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    rotation_weight: float = 0.1,
    scale_weight: float = 0.1,
    angular_consistency_weight: float = 0.2,
) -> torch.Tensor:
    """Sin/cos MSE + scale MSE + geometric great-circle nudge.

    Parameters
    ----------
    pred : Tensor [B, 7]
        Raw model outputs (sin_ra, cos_ra, sin_dec, cos_dec,
        sin_rot, cos_rot, log_scale).
    target : Tensor [B, 4]
        Legacy label format: (ra_deg, dec_deg, rot_deg, log_scale).
    rotation_weight, scale_weight : float
        Multipliers on the rotation and scale contributions.
    angular_consistency_weight : float
        Small great-circle term on decoded (RA, Dec) — keeps the
        trig outputs geometrically meaningful.
    """
    target_sincos = AstroLocNet.encode_labels(target)  # [B, 7]

    # 1) Direct MSE on each trig pair and on log_scale.
    sq = (pred - target_sincos) ** 2
    ra_mse = sq[:, 0:2].mean()
    dec_mse = sq[:, 2:4].mean()
    rot_mse = sq[:, 4:6].mean()
    scale_mse = sq[:, 6].mean()

    # 2) Angular consistency on decoded (RA, Dec) — haversine, smooth.
    decoded = AstroLocNet.decode_predictions(pred)
    ra_p, dec_p = torch.deg2rad(decoded[:, 0]), torch.deg2rad(decoded[:, 1])
    ra_t, dec_t = torch.deg2rad(target[:, 0]), torch.deg2rad(target[:, 1])
    d_ra = ra_p - ra_t
    d_dec = dec_p - dec_t
    a = torch.sin(d_dec * 0.5) ** 2 + torch.cos(dec_t) * torch.cos(dec_p) * torch.sin(d_ra * 0.5) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    angular_sep_rad = 2.0 * torch.asin(torch.sqrt(a))

    return (
        ra_mse
        + dec_mse
        + rotation_weight * rot_mse
        + scale_weight * scale_mse
        + angular_consistency_weight * angular_sep_rad.mean()
    )


class AstroLocLoss(nn.Module):
    """nn.Module wrapper for use in the Trainer."""

    def __init__(
        self,
        rotation_weight: float = 0.1,
        scale_weight: float = 0.1,
        angular_consistency_weight: float = 0.2,
    ):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.scale_weight = scale_weight
        self.angular_consistency_weight = angular_consistency_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return angular_separation_loss(
            pred, target,
            rotation_weight=self.rotation_weight,
            scale_weight=self.scale_weight,
            angular_consistency_weight=self.angular_consistency_weight,
        )
