"""Three-phase training loop with checkpointing and early stopping.

Phase 1: head only, backbone frozen.
Phase 2: unfreeze last N blocks; differential LRs.
Phase 3: real-image fine-tuning at lower LRs (optional; runs only when
         ``real_data.enabled`` is true).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.models.astrolocnet import AstroLocNet, freeze_backbone, unfreeze_last_n_blocks
from src.training.loss import AstroLocLoss
from src.training.metrics import MetricsAccumulator


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class PhaseConfig:
    epochs: int
    lr_head: float = 1.0e-4
    lr_backbone_early: float = 1.0e-5
    lr_backbone_late: float = 5.0e-5
    unfreeze_last_n_blocks: int = 0


@dataclass
class TrainerConfig:
    batch_size: int = 32
    amp: bool = True
    device: str = "auto"
    early_stopping_patience: int = 5
    rotation_weight: float = 0.1
    scale_weight: float = 0.1
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "checkpoints/runs"
    best_name: str = "best.pt"
    phase1: PhaseConfig = field(default_factory=lambda: PhaseConfig(epochs=5, lr_head=1e-3))
    phase2: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        epochs=15, lr_head=1e-4, lr_backbone_early=1e-5,
        lr_backbone_late=5e-5, unfreeze_last_n_blocks=3,
    ))
    phase3: PhaseConfig = field(default_factory=lambda: PhaseConfig(
        epochs=10, lr_head=5e-5, lr_backbone_early=1e-6,
        lr_backbone_late=1e-5, unfreeze_last_n_blocks=3,
    ))


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class Trainer:
    """Orchestrates the multi-phase training loop.

    Designed to be both CLI-driven (``train.py``) and notebook-driven —
    each phase exposes a ``run_phase_*`` method you can call individually
    in ``notebooks/05_training.ipynb``.
    """

    def __init__(
        self,
        model: AstroLocNet,
        config: TrainerConfig,
        *,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        self.loss_fn = AstroLocLoss(
            rotation_weight=config.rotation_weight,
            scale_weight=config.scale_weight,
        ).to(self.device)

        self.use_amp = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.history: List[Dict[str, Any]] = []
        self.best_metric = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(config.log_dir) / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress_cb = progress_cb or (lambda evt: None)

    # ------------------------------------------------------------------ #
    # Public phases
    # ------------------------------------------------------------------ #

    def run_phase1(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, Any]]:
        freeze_backbone(self.model)
        optimizer = optim.AdamW(self.model.head_parameters, lr=self.config.phase1.lr_head)
        return self._run_phase("phase1", self.config.phase1.epochs, optimizer, train_loader, val_loader)

    def run_phase2(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, Any]]:
        # Re-freeze, then unfreeze the last N blocks.
        freeze_backbone(self.model)
        unfreeze_last_n_blocks(self.model, n=self.config.phase2.unfreeze_last_n_blocks)
        param_groups = self._make_phase2_param_groups(self.config.phase2)
        optimizer = optim.AdamW(param_groups)
        return self._run_phase("phase2", self.config.phase2.epochs, optimizer, train_loader, val_loader)

    def run_phase3(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, Any]]:
        freeze_backbone(self.model)
        unfreeze_last_n_blocks(self.model, n=self.config.phase3.unfreeze_last_n_blocks)
        param_groups = self._make_phase2_param_groups(self.config.phase3)
        optimizer = optim.AdamW(param_groups)
        return self._run_phase("phase3", self.config.phase3.epochs, optimizer, train_loader, val_loader)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _make_phase2_param_groups(self, pc: PhaseConfig) -> List[dict]:
        # Backbone "early" = first (len-N) blocks, "late" = last N blocks.
        n_unfreeze = pc.unfreeze_last_n_blocks
        total_blocks = len(self.model.backbone.features)
        late_start = max(0, total_blocks - n_unfreeze)
        early_params: List[nn.Parameter] = []
        late_params: List[nn.Parameter] = []
        for idx, block in enumerate(self.model.backbone.features):
            for p in block.parameters():
                if not p.requires_grad:
                    continue
                (late_params if idx >= late_start else early_params).append(p)
        groups = [
            {"params": late_params, "lr": pc.lr_backbone_late},
            {"params": self.model.head_parameters, "lr": pc.lr_head},
        ]
        if early_params:
            groups.insert(0, {"params": early_params, "lr": pc.lr_backbone_early})
        return groups

    def _run_phase(
        self,
        name: str,
        epochs: int,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict[str, Any]]:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        no_improve = 0
        phase_history: List[Dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(train_loader, optimizer)
            val_metrics = self._evaluate(val_loader)
            scheduler.step()
            entry = {
                "phase": name, "epoch": epoch,
                "lr": [g["lr"] for g in optimizer.param_groups],
                "train": train_metrics, "val": val_metrics,
            }
            self.history.append(entry)
            phase_history.append(entry)
            self.progress_cb(entry)

            val_metric = val_metrics["ang_sep_mean_deg"]
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.save_checkpoint(self.checkpoint_dir / self.config.best_name)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.config.early_stopping_patience:
                    self.progress_cb({"event": "early_stop", "phase": name, "epoch": epoch})
                    break

        self.save_checkpoint(self.checkpoint_dir / f"last_{name}.pt")
        self._write_history()
        return phase_history

    def _train_one_epoch(self, loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        self.model.train()
        acc = MetricsAccumulator()
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    preds = self.model(images)
                    loss = self.loss_fn(preds, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
            acc.update(preds, labels, loss.item())
        return acc.compute()

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        acc = MetricsAccumulator()
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds = self.model(images)
            loss = self.loss_fn(preds, labels)
            acc.update(preds, labels, loss.item())
        return acc.compute()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
            "config": asdict(self.config),
        }, path)

    @staticmethod
    def load_state(model: AstroLocNet, path: str | Path, *, map_location: str = "cpu") -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["state_dict"])
        return ckpt

    def _write_history(self) -> None:
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(spec)
