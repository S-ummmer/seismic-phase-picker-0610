# tests/test_reader.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_reader.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data.reader import Waveform


class TestWaveform:
    """Waveform 数据类单元测试。"""

    def setup_method(self):
        np.random.seed(42)
        self.data = np.random.randn(3, 1000).astype(np.float32)  # 3ch, 10s @ 100Hz
        self.wf = Waveform(
            data=self.data,
            sampling_rate=100.0,
            start_time=1234567890.0,
            channel_names=("Z", "N", "E"),
            station_id="ST001",
        )

    def test_basic_properties(self):
        assert self.wf.n_channels == 3
        assert self.wf.n_samples == 1000
        assert self.wf.duration == pytest.approx(10.0)
        assert self.wf.end_time == pytest.approx(1234567900.0)

    def test_time_at_index(self):
        t0 = self.wf.time_at_index(0)
        assert t0 == pytest.approx(1234567890.0)
        t_end = self.wf.time_at_index(999)
        assert t_end == pytest.approx(1234567899.99)

    def test_index_at_time(self):
        idx = self.wf.index_at_time(1234567895.0)  # 5s in
        assert idx == 500
        idx_clamp = self.wf.index_at_time(1234567890.0 - 100)  # before start
        assert idx_clamp == 0

    def test_time_to_sample(self):
        assert self.wf.time_to_sample(5.0) == 500
        assert self.wf.time_to_sample(0.0) == 0
        assert self.wf.time_to_sample(-1.0) == 0  # clamped

    def test_sample_to_time(self):
        assert self.wf.sample_to_time(500) == pytest.approx(5.0)

    def test_repr(self):
        s = repr(self.wf)
        assert "ST001" in s
        assert "(3, 1000)" in s
