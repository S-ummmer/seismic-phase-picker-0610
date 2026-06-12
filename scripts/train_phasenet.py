# scripts/train_phasenet.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\train_phasenet.py
"""PhaseNet 训练入口。

在 STEAD 数据集上训练三分类模型（噪声/P/S）。

用法:
    # 小样本试跑
    python scripts/train_phasenet.py --epochs 2 --max-train 200 --max-val 50

    # 完整训练
    python scripts/train_phasenet.py --epochs 50 --batch-size 32 --lr 1e-3

    # 从检查点恢复
    python scripts/train_phasenet.py --resume outputs/checkpoints/last_model.pt --epochs 10
"""

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seisbench.models import PhaseNet

from src.training.dataset import SteadDataset
from src.training.trainer import PhaseNetTrainer


def main():
    parser = argparse.ArgumentParser(description="Train PhaseNet on STEAD")

    # 数据
    parser.add_argument("--hdf5", default="data/raw/stead/waveforms.hdf5")
    parser.add_argument("--csv", default="data/raw/stead/metadata.csv")
    parser.add_argument("--window-size", type=int, default=6001,
                        help="PhaseNet 输入长度 (6001 样本 = 60s @100Hz, 震前10s+震后50s)")
    parser.add_argument("--p-offset", type=float, default=10.0,
                        help="窗口起始到 P 到时的目标秒数（默认 10s）")
    parser.add_argument("--s-offset", type=float, default=50.0,
                        help="窗口起始到 S 到时的最大秒数（默认 50s）")

    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10,
                        help="EarlyStopping 耐心值，0=禁用")
    parser.add_argument("--device", default="cpu",
                        help="cpu / cuda / cuda:0")

    # 标签
    parser.add_argument("--p-buffer", type=int, default=20,
                        help="P 波标记半窗口（样本）")
    parser.add_argument("--s-buffer", type=int, default=20,
                        help="S 波标记半窗口（样本）")
    parser.add_argument("--require-both-ps", action="store_true",
                        help="只使用同时有 P+S 标签的 trace")

    # 数据量限制（快速验证用）
    parser.add_argument("--max-train", type=int, default=0,
                        help="训练集最大样本数 (0=全部)")
    parser.add_argument("--max-val", type=int, default=0,
                        help="验证集最大样本数 (0=全部)")

    # 保存与恢复
    parser.add_argument("--save-dir", default="outputs/checkpoints")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存检查点（快速验证用）")
    parser.add_argument("--resume", default="",
                        help="从检查点恢复训练")

    # 日志
    parser.add_argument("--log-interval", type=int, default=50,
                        help="每 N 个 batch 打印一次 loss")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers")

    args = parser.parse_args()

    # ── 设备 ──────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"PhaseNet Training Configuration")
    print(f"{'='*60}")
    print(f"Data:   {args.hdf5}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"P/S buffer: {args.p_buffer}/{args.s_buffer} samples")
    print(f"Require both PS: {args.require_both_ps}")
    print(f"Max train/val: {args.max_train}/{args.max_val}")
    print(f"{'='*60}\n")

    # ── 数据集 ────────────────────────────────────────────
    print("Loading datasets...")

    train_dataset = SteadDataset(
        hdf5_path=args.hdf5,
        csv_path=args.csv,
        split="train",
        window_size=args.window_size,
        target_sr=100.0,
        p_buffer=args.p_buffer,
        s_buffer=args.s_buffer,
        p_offset_sec=args.p_offset,
        s_offset_sec=args.s_offset,
        require_both_ps=args.require_both_ps,
        max_traces=args.max_train,
    )

    val_dataset = SteadDataset(
        hdf5_path=args.hdf5,
        csv_path=args.csv,
        split="dev",
        window_size=args.window_size,
        target_sr=100.0,
        p_buffer=args.p_buffer,
        s_buffer=args.s_buffer,
        p_offset_sec=args.p_offset,
        s_offset_sec=args.s_offset,
        require_both_ps=args.require_both_ps,
        max_traces=args.max_val,
    )

    print(f"Train: {len(train_dataset)} traces")
    print(f"Val:   {len(val_dataset)} traces")

    # ── 类别权重 ──────────────────────────────────────────
    print("\nComputing class weights...")
    class_weights = train_dataset.class_weights()
    print(f"Class weights (Noise/P/S): {class_weights.tolist()}")

    # ── DataLoader ────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
    )

    # ── 模型 ──────────────────────────────────────────────
    print("\nInitializing PhaseNet...")
    model = PhaseNet(phases="NPS")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Resumed at epoch {ckpt.get('epoch', '?')}, "
              f"best_val_loss={ckpt.get('best_val_loss', '?'):.4f}")
    else:
        print("Training from scratch (random init)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # ── 训练 ──────────────────────────────────────────────
    trainer = PhaseNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
        save_dir=None if args.no_save else args.save_dir,
        log_interval=args.log_interval,
        patience=args.patience,
    )

    history = trainer.train()

    # ── 总结 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"Best val_loss:     {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch})")
    print(f"Final train_loss:  {history['train_loss'][-1]:.4f}")
    print(f"Final val_loss:    {history['val_loss'][-1]:.4f}")
    print(f"Final train_acc:   {history['train_acc'][-1]:.4f}")
    print(f"Final val_acc:     {history['val_acc'][-1]:.4f}")
    if not args.no_save:
        print(f"\nCheckpoints saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
