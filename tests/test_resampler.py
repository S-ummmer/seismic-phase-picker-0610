# tests/test_resampler.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_resampler.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.io import Waveform
from src.signal.resampler import Resampler


class TestResampler:
    """重采样器单元测试。"""

    def setup_method(self):
        np.random.seed(42)
        self.wf = Waveform(
            data=np.random.randn(3, 500).astype(np.float32),  # 5s @ 100Hz
            sampling_rate=100.0,
            starttime=0.0,
            station="TEST",
        )

    def test_upsample(self):
        resampler = Resampler(target_sr=200.0)
        result = resampler.resample(self.wf)
        assert result.sampling_rate == 200.0
        assert result.n_samples == 1000

    def test_downsample(self):
        resampler = Resampler(target_sr=50.0)
        result = resampler.resample(self.wf)
        assert result.sampling_rate == 50.0
        assert result.n_samples == 250

    def test_same_rate_noop(self):
        resampler = Resampler(target_sr=100.0)
        result = resampler.resample(self.wf)
        assert result is self.wf  # same object returned

    def test_metadata_preserved(self):
        resampler = Resampler(target_sr=200.0)
        result = resampler.resample(self.wf)
        assert result.starttime == self.wf.starttime
        assert result.station == self.wf.station
