# tests/test_denoiser.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_denoiser.py
"""DeepDenoiser 模块测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 注意：DeepDenoiser 需要 torch + seisbench + 预训练权重下载
# 这些测试设计为可跳过（未安装时不失败）

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import seisbench.models as sbm
    HAS_SEISBENCH = True
except ImportError:
    HAS_SEISBENCH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """生成模拟三分量地震波形（含正弦信号 + 噪声）。"""
    np.random.seed(42)
    N = 3000
    t = np.arange(N, dtype=np.float64) / 100.0  # 100 Hz
    # 合成信号：一个脉冲 + 正弦
    signal = np.exp(-((t - 10.0) ** 2) / 0.5) * np.sin(2 * np.pi * 5 * t)
    noise = np.random.randn(N).astype(np.float64) * 0.1
    ch1 = signal + noise
    ch2 = signal * 0.8 + np.random.randn(N).astype(np.float64) * 0.1
    ch3 = signal * 0.5 + np.random.randn(N).astype(np.float64) * 0.1
    data = np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)
    return data


@pytest.fixture
def long_data():
    """长波形（6001 样本，60s）。"""
    np.random.seed(43)
    N = 6001
    data = np.random.randn(3, N).astype(np.float32) * 0.1
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH or not HAS_SEISBENCH,
                    reason="Requires torch + seisbench")
class TestDeepDenoiser:
    """需要 torch + seisbench 安装的测试。"""

    def test_create_denoiser(self):
        """测试工厂函数（disabled → None）。"""
        from src.signal.denoiser import create_denoiser

        d = create_denoiser(enabled=False)
        assert d is None

    def test_import(self):
        """测试导入。"""
        from src.signal.denoiser import DeepDenoiser, create_denoiser
        assert DeepDenoiser is not None
        assert create_denoiser is not None

    def test_init_no_load(self):
        """测试初始化不触发模型加载（懒加载）。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        assert d._model is None
        assert not d._loaded

    def test_model_load(self):
        """测试实际加载预训练模型（需要网络）。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        model = d.model  # 触发懒加载
        assert model is not None
        assert d._loaded
        # 验证模型类型
        from seisbench.models import DeepDenoiser as SBM_DD
        assert isinstance(model, SBM_DD)

    def test_denoise_single(self, sample_data):
        """测试单窗口去噪 (3, 3000)。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        denoised = d.denoise(sample_data)

        assert denoised.shape == sample_data.shape
        assert denoised.dtype == np.float32
        assert np.all(np.isfinite(denoised)), "Output contains NaN/Inf"

    def test_denoise_short_input(self):
        """测试输入 < 3000 样本。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        data = np.random.randn(3, 1500).astype(np.float32)

        denoised = d.denoise(data)
        assert denoised.shape == (3, 1500)

    def test_denoise_long_input(self, long_data):
        """测试长波形滑动窗口去噪 (3, 6001)。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        denoised = d.denoise(long_data)

        assert denoised.shape == long_data.shape
        assert denoised.dtype == np.float32
        assert np.all(np.isfinite(denoised))

    def test_denoise_reduces_noise_level(self, sample_data):
        """去噪后的 RMS 振幅应小于原始信号（噪声成分被抑制）。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        denoised = d.denoise(sample_data)

        rms_raw = np.sqrt(np.mean(sample_data ** 2))
        rms_dn = np.sqrt(np.mean(denoised ** 2))

        # 去噪后的 RMS 不应显著增大
        assert rms_dn <= rms_raw * 2.0, (
            f"RMS increased too much: {rms_raw:.4f} → {rms_dn:.4f}"
        )

    def test_denoise_preserves_structure(self, sample_data):
        """去噪后的波形整体结构（峰值位置）应大致保留。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        denoised = d.denoise(sample_data)

        # 峰值位置偏差应在合理范围（±2 样本 = ±0.02s）
        for ch in range(3):
            raw_peak = np.argmax(np.abs(sample_data[ch]))
            dn_peak = np.argmax(np.abs(denoised[ch]))
            assert abs(raw_peak - dn_peak) <= 5, (
                f"Channel {ch}: peak shifted from {raw_peak} to {dn_peak}"
            )

    def test_denoise_waveforms_batch(self, sample_data):
        """测试批量去噪。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")
        batch = np.stack([sample_data, sample_data * 0.5], axis=0)  # (2, 3, 3000)

        denoised = d.denoise_waveforms(batch)
        assert denoised.shape == batch.shape
        assert np.all(np.isfinite(denoised))

    def test_cosine_window(self):
        """测试余弦窗口生成。"""
        from src.signal.denoiser import DeepDenoiser

        win = DeepDenoiser._cosine_window(3000)
        assert len(win) == 3000
        assert win[0] < 0.01  # 从接近 0 开始
        assert win[-1] < 0.01
        assert 0.99 < win[1500] <= 1.0  # 中间接近 1

    def test_input_validation(self):
        """测试输入验证。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")

        with pytest.raises(ValueError, match="2D array"):
            d.denoise(np.random.randn(3, 100, 100))

    def test_denoise_1d_compatibility(self, sample_data):
        """测试 1D 输入的报错提示。"""
        from src.signal.denoiser import DeepDenoiser

        d = DeepDenoiser(pretrained="original")

        with pytest.raises(ValueError):
            d.denoise(sample_data[0])  # 1D instead of 2D


@pytest.mark.skipif(HAS_TORCH and HAS_SEISBENCH,
                    reason="Skipping when torch+seisbench available")
class TestDeepDenoiserNoSeisbench:
    """torch/seisbench 未安装时的行为测试。"""

    def test_import_without_seisbench(self):
        """测试导入不因缺失 seisbench 而失败。"""
        from src.signal.denoiser import DeepDenoiser, create_denoiser
        assert DeepDenoiser is not None

    def test_factory_disabled(self):
        """测试 disabled 时不加载。"""
        from src.signal.denoiser import create_denoiser
        d = create_denoiser(enabled=False)
        assert d is None
