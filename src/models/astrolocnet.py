"""AstroLocNet: EfficientNet-B0 backbone + regression head."""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class AstroLocNet(nn.Module):
    """Predicts ``[ra_deg, dec_deg, rotation_deg, log_field_width_deg]``.

    Note on the output head: we intentionally regress raw (RA, Dec)
    rather than (sin, cos) pairs. The wrap-around problem is handled by
    the angular-separation loss, not by the parameterization. See
    ``notebooks/04_loss_function.ipynb`` for the discussion.
    """

    OUTPUT_DIM = 4  # ra, dec, rotation, log_scale

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
        return self.backbone(x)

    # ------------------------------------------------------------------ #
    # Parameter group helpers — used by the trainer to set differential LRs.
    # ------------------------------------------------------------------ #

    @property
    def head_parameters(self) -> List[nn.Parameter]:
        return list(self.backbone.classifier.parameters())

    @property
    def backbone_feature_parameters(self) -> List[nn.Parameter]:
        # Everything except the classifier head.
        return [p for n, p in self.backbone.named_parameters() if not n.startswith("classifier")]

    def block_parameters(self, block_idx: int) -> List[nn.Parameter]:
        """Return the parameters of a specific EfficientNet feature block.

        EfficientNet-B0's feature stack has 9 entries (indices 0..8).
        Index 0 is the stem; indices 1..7 are the MBConv stages;
        index 8 is the head conv.
        """
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
    blocks 6, 7, 8 — the high-level stages that are most useful to adapt
    to the synthetic star-field domain.
    """
    total_blocks = len(model.backbone.features)
    start = max(0, total_blocks - n)
    unfrozen = 0
    for idx in range(start, total_blocks):
        for p in model.backbone.features[idx].parameters():
            p.requires_grad = True
            unfrozen += p.numel()
    return unfrozen
