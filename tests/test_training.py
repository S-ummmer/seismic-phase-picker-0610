# tests/test_training.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_training.py
"""训练管线单元测试。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from src.training.label_generator import (
    generate_phase_mask, mask_to_onehot, mask_class_distribution,
)
from src.training.dataset import SteadDataset
from src.training.trainer import PhaseNetTrainer


class TestLabelGenerator:
    """标签生成器单元测试。"""

    def test_basic_mask(self):
        mask = generate_phase_mask(3001, p_sample=500, s_sample=1200)
        assert mask.shape == (3001,)
        assert mask.dtype == np.int64
        # P 窗口
        assert (mask[480:520] == 1).all()
        # S 窗口
        assert (mask[1180:1220] == 2).all()
        # 其余是噪声
        assert mask[0] == 0
        assert mask[-1] == 0

    def test_p_only(self):
        mask = generate_phase_mask(3001, p_sample=500, s_sample=None)
        # buffer=20 → 起始=480, 结束=520, 共 40 个样本 (2*buffer)
        assert (mask == 1).sum() == 40
        assert (mask == 2).sum() == 0

    def test_s_only(self):
        mask = generate_phase_mask(3001, p_sample=None, s_sample=1200)
        assert (mask == 2).sum() == 40
        assert (mask == 1).sum() == 0

    def test_overlap_s_priority(self):
        # P=50, S=55, buffer=20 → 重叠区 S 优先
        mask = generate_phase_mask(200, p_sample=50, s_sample=55, p_buffer=20, s_buffer=20)
        overlap = mask[35:70]
        assert 1 not in overlap  # P 被 S 覆盖

    def test_boundary_clamping(self):
        # P 在边缘，buffer 超出范围
        mask = generate_phase_mask(100, p_sample=5, s_sample=95, p_buffer=20, s_buffer=20)
        assert mask[0] == 1   # p_sample=5, buffer=20 → start=0
        assert mask[-1] == 2  # s_sample=95, buffer=20 → end=100

    def test_mask_to_onehot(self):
        mask = np.array([0, 0, 1, 0, 2, 0], dtype=np.int64)
        onehot = mask_to_onehot(mask, n_classes=3)
        assert onehot.shape == (3, 6)
        assert (onehot[0] == [1, 1, 0, 1, 0, 1]).all()
        assert (onehot[1] == [0, 0, 1, 0, 0, 0]).all()
        assert (onehot[2] == [0, 0, 0, 0, 1, 0]).all()

    def test_class_distribution(self):
        mask = np.array([0, 0, 0, 1, 0, 2], dtype=np.int64)
        dist = mask_class_distribution(mask)
        assert dist["noise"] == 4 / 6
        assert dist["P"] == 1 / 6
        assert dist["S"] == 1 / 6


class TestSteadDataset:
    """STEAD Dataset 集成测试。"""

    def test_dataset_creation(self):
        ds = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=10,
        )
        assert len(ds) == 10

    def test_sample_shape(self):
        ds = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=5,
        )
        wf, mask, meta = ds[0]
        assert isinstance(wf, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert wf.shape == (3, 3001)
        assert mask.shape == (3001,)
        assert wf.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_mask_values(self):
        ds = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=5,
        )
        for i in range(len(ds)):
            _, mask, _ = ds[i]
            unique = set(mask.unique().tolist())
            assert unique.issubset({0, 1, 2}), f"Unexpected mask values: {unique}"

    def test_class_weights(self):
        ds = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=10,
        )
        weights = ds.class_weights()
        assert weights.shape == (3,)
        # 噪声权重最小，S 权重最大（因为 S 最少）
        assert weights[0] < 1.0
        assert weights[2] > 1.0

    def test_require_both_ps(self):
        """require_both_ps=True 时，CSV 中无 P/S 任一标签的 trace 被过滤。"""
        import pandas as pd
        df = pd.read_csv("data/raw/stead/metadata.csv")

        # 对比：不筛选 vs 筛选
        ds_all = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=50,
            require_both_ps=False,
        )
        ds_filtered = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=50,
            require_both_ps=True,
        )
        # 筛选后数量应该 ≤ 不筛选
        assert len(ds_filtered) <= len(ds_all)
        # 筛选后每条在 CSV 中都应有 P 和 S
        for i in range(len(ds_filtered)):
            _, _, meta = ds_filtered[i]
            csv_idx = df.iloc[ds_filtered._indices[i]]
            has_p_csv = pd.notna(csv_idx["trace_P_arrival_sample"])
            has_s_csv = pd.notna(csv_idx["trace_S_arrival_sample"])
            assert has_p_csv and has_s_csv, (
                f"Sample {i}: should have been filtered, but wasn't"
            )

    def test_p_not_missing(self):
        """所有 trace P 标签不能丢（即使 S 超出窗口）。"""
        ds = SteadDataset(
            "data/raw/stead/waveforms.hdf5",
            "data/raw/stead/metadata.csv",
            split="dev", max_traces=50,
        )
        for i in range(len(ds)):
            _, mask, meta = ds[i]
            if meta["has_p"]:
                has_p_in_mask = (mask == 1).sum() > 0
                assert has_p_in_mask, (
                    f"[{i}] {meta['trace_name']}: has_p=True but P not in mask"
                )


class TestTrainerInit:
    """训练器初始化测试。"""

    def test_create(self):
        model = torch.nn.Linear(10, 3)
        train_loader = torch.utils.data.DataLoader(
            [(torch.randn(10), torch.tensor(0))], batch_size=1,
        )
        val_loader = torch.utils.data.DataLoader(
            [(torch.randn(10), torch.tensor(0))], batch_size=1,
        )
        trainer = PhaseNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1, save_dir=None,
        )
        assert trainer.epochs == 1
        assert trainer.best_val_loss == float("inf")
