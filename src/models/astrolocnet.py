"""AstroLocNet: EfficientNet-B0 backbone + 7-output regression head.

Parameterization choice
-----------------------
Earlier versions predicted raw ``[ra_deg, dec_deg, rotation_deg, log_scale]``
and relied on a haversine loss to handle the RA / rotation wrap-around at
0°/360°. Empirically this trains very slowly: the network has no incentive
to keep predictions in the valid ranges ([0, 360), [-90, +90]), so the
optimizer wanders in unbounded space and converges to barely-better-than-
random.

The fix is the standard one for spherical / pose regression — predict
``(sin, cos)`` pairs for every wrapping angle and reconstruct the angle
via ``atan2``. The loss surface becomes smooth everywhere and the
network never has to learn a discontinuous mapping.

Output layout
~~~~~~~~~~~~~
::

    [0] sin(RA)        in [-1, 1]
    [1] cos(RA)
    [2] sin(Dec)
    [3] cos(Dec)
    [4] sin(rotation)
    [5] cos(rotation)
    [6] log(field_width_deg)

Use :func:`AstroLocNet.decode_predictions` to turn raw outputs into
degree-valued ``[ra, dec, rotation, log_scale]`` tensors compatible
with the legacy 4-column label format.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class AstroLocNet(nn.Module):
    """EfficientNet-B0 backbone + 7-output sin/cos regression head."""

    OUTPUT_DIM = 7  # sin_ra, cos_ra, sin_dec, cos_dec, sin_rot, cos_rot, log_scale

    def __init__(
        self,
        *,
        pretrained: bool = True,
        dropout_p_1: float = 0.3,
        dropout_p_2: float = 0.2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p_1),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p_2),
            nn.Linear(hidden_dim, self.OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw 7-D outputs (NOT decoded to angles)."""
        return self.backbone(x)

    # ------------------------------------------------------------------ #
    # Encode / decode between (RA, Dec, rot, log_scale) and the 7-D head.
    # ------------------------------------------------------------------ #

    @staticmethod
    def encode_labels(labels_deg: torch.Tensor) -> torch.Tensor:
        """``[B, 4]`` (ra, dec, rot, log_scale) → ``[B, 7]`` sin/cos targets."""
        ra = torch.deg2rad(labels_deg[:, 0])
        dec = torch.deg2rad(labels_deg[:, 1])
        rot = torch.deg2rad(labels_deg[:, 2])
        log_scale = labels_deg[:, 3]
        return torch.stack([
            torch.sin(ra), torch.cos(ra),
            torch.sin(dec), torch.cos(dec),
            torch.sin(rot), torch.cos(rot),
            log_scale,
        ], dim=-1)

    @staticmethod
    def decode_predictions(preds: torch.Tensor) -> torch.Tensor:
        """``[B, 7]`` raw outputs → ``[B, 4]`` (ra, dec, rot, log_scale) in degrees.

        Uses ``atan2(sin, cos)`` so the result is always in a valid range
        regardless of the raw output magnitudes (the network does not
        need to learn to produce unit-norm pairs).
        """
        ra = torch.rad2deg(torch.atan2(preds[:, 0], preds[:, 1])) % 360.0
        dec = torch.rad2deg(torch.atan2(preds[:, 2], preds[:, 3]))
        dec = torch.clamp(dec, -90.0, 90.0)
        rot = torch.rad2deg(torch.atan2(preds[:, 4], preds[:, 5])) % 360.0
        log_scale = preds[:, 6]
        return torch.stack([ra, dec, rot, log_scale], dim=-1)

    # ------------------------------------------------------------------ #
    # Parameter group helpers — used by the trainer to set differential LRs.
    # ------------------------------------------------------------------ #

    @property
    def head_parameters(self) -> List[nn.Parameter]:
        return list(self.backbone.classifier.parameters())

    @property
    def backbone_feature_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.backbone.named_parameters() if not n.startswith("classifier")]

    def block_parameters(self, block_idx: int) -> List[nn.Parameter]:
        """Return parameters of a specific EfficientNet feature block (0..8)."""
        block = self.backbone.features[block_idx]
        return list(block.parameters())


# --------------------------------------------------------------------------- #
# Freezing helpers
# --------------------------------------------------------------------------- #


def freeze_backbone(model: AstroLocNet) -> int:
    """Freeze every parameter except the classifier head. Returns count frozen."""
    frozen = 0
    for name, p in model.backbone.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def unfreeze_last_n_blocks(model: AstroLocNet, n: int = 3) -> int:
    """Unfreeze the final ``n`` feature blocks. Returns count unfrozen.

    EfficientNet-B0 has 9 feature blocks (indices 0..8). ``n=3`` unfreezes
    blocks 6, 7, 8 — the high-level stages most useful to adapt to the
    synthetic star-field domain.
    """
    total_blocks = len(model.backbone.features)
    start = max(0, total_blocks - n)
    unfrozen = 0
    for idx in range(start, total_blocks):
        for p in model.backbone.features[idx].parameters():
            p.requires_grad = True
            unfrozen += p.numel()
    return unfrozen
