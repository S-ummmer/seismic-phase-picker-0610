# tests/test_reader.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_reader.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.io import Waveform


class TestWaveform:
    """Waveform 数据类单元测试。"""

    def setup_method(self):
        np.random.seed(42)
        self.data = np.random.randn(3, 1000).astype(np.float32)  # 3ch, 10s @ 100Hz
        self.wf = Waveform(
            data=self.data,
            sampling_rate=100.0,
            starttime=1234567890.0,
            channel="ENZ",
            station="ST001",
        )

    def test_basic_properties(self):
        assert self.wf.n_channels == 3
        assert self.wf.n_samples == 1000
        assert abs(self.wf.duration - 10.0) < 1e-6
        end_t = self.wf.time_at_index(999)
        assert abs(end_t - 1234567899.99) < 0.01

    def test_time_at_index(self):
        t0 = self.wf.time_at_index(0)
        assert abs(t0 - 1234567890.0) < 1e-6
        t_end = self.wf.time_at_index(999)
        assert abs(t_end - 1234567899.99) < 0.01

    def test_repr(self):
        s = repr(self.wf)
        assert "ST001" in s
        assert "(3, 1000)" in s
