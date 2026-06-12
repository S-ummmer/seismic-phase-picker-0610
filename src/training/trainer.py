# src/training/trainer.py
# D:\Github\Mk-project\seismic-phase-picker\src/training\trainer.py
"""PhaseNet 模型训练器。

标准 PyTorch 训练循环：
- CrossEntropyLoss（带类别权重）
- Adam + CosineAnnealingLR
- 每个 epoch 在 dev 集验证
- 保存最佳模型（基于 val_loss）
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PhaseNetTrainer:
    """PhaseNet 模型训练器。

    用法:
        trainer = PhaseNetTrainer(
            model, train_loader, val_loader,
            lr=1e-3, epochs=50, device="cpu",
            save_dir="outputs/checkpoints",
        )
        history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        device: str = "cpu",
        save_dir: str = "outputs/checkpoints",
        log_interval: int = 50,
        patience: int = 10,
    ):
        """
        Parameters
        ----------
        patience : int
            EarlyStopping 耐心值（连续 N 个 epoch val_loss 无改善则停止），
            0 = 禁用。
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # EarlyStopping
        self.patience = patience
        self.patience_counter = 0

        # 损失函数
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 优化器 + 调度器
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs,
        )

        # 记录
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train(self) -> dict:
        """执行完整训练流程。"""
        print(f"\n{'='*60}")
        print(f"PhaseNet Training Started")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches:   {len(self.val_loader)}")
        print(f"Epochs: {self.epochs}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate()

            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            # 打印
            marker = ""
            improved = val_loss < self.best_val_loss

            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt")
                marker = " *"
            else:
                self.patience_counter += 1
                if self.patience > 0:
                    marker = f" (early_stop {self.patience_counter}/{self.patience})"

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} | "
                f"lr={lr:.2e}{marker}"
            )

            # EarlyStopping 检查
            if self.patience > 0 and self.patience_counter >= self.patience:
                print(
                    f"\nEarly stopping triggered: "
                    f"val_loss not improved for {self.patience} epochs."
                )
                break

        # 训练结束
        self._save_checkpoint("last_model.pt")
        print(f"\nTraining complete. Best val_loss={self.best_val_loss:.4f} at epoch {self.best_epoch}")
        return self.history

    # ── 内部 ──────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (waveforms, masks, _meta) in enumerate(self.train_loader):
            waveforms = waveforms.to(self.device)   # (B, 3, N)
            masks = masks.to(self.device)            # (B, N)

            self.optimizer.zero_grad()

            # forward
            logits = self.model(waveforms, logits=True)  # (B, 3, N)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            with torch.no_grad():
                preds = logits.argmax(dim=1)          # (B, N)
                # 只算非零标签位置（噪声占比大，全样本准确率无意义）
                total_correct += (preds == masks).sum().item()
                total_samples += masks.numel()

            if (batch_idx + 1) % self.log_interval == 0:
                print(
                    f"  [E{epoch}] batch {batch_idx+1}/{len(self.train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    @torch.no_grad()
    def _validate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for waveforms, masks, _meta in self.val_loader:
            waveforms = waveforms.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(waveforms, logits=True)
            loss = self.criterion(logits, masks)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == masks).sum().item()
            total_samples += masks.numel()

        avg_loss = total_loss / max(len(self.val_loader), 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    def _save_checkpoint(self, filename: str):
        if self.save_dir is None:
            return
        path = self.save_dir / filename
        try:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": len(self.history["train_loss"]),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            }, path)
        except (PermissionError, RuntimeError) as e:
            print(f"  [Warning] Could not save checkpoint: {e}")
            # 尝试 fallback 到临时目录
            import tempfile
            alt_path = Path(tempfile.gettempdir()) / f"phasenet_{filename}"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": len(self.history["train_loss"]),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            }, alt_path)
            print(f"  [Info] Saved to {alt_path} instead")
