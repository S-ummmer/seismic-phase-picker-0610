# tests/test_preprocessor.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_preprocessor.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data.reader import Waveform
from src.data.preprocessor import Preprocessor


class TestPreprocessor:
    """预处理器单元测试。"""

    def setup_method(self):
        np.random.seed(42)
        self.wf = Waveform(
            data=np.random.randn(3, 1000).astype(np.float32) + 5.0,  # 含偏置
            sampling_rate=100.0,
            start_time=0.0,
        )

    def test_demean(self):
        pp = Preprocessor(demean=True, normalize=False)
        result = pp.process(self.wf)
        for ch in range(result.n_channels):
            assert abs(result.data[ch].mean()) < 1e-5

    def test_normalize(self):
        pp = Preprocessor(demean=False, normalize=True)
        result = pp.process(self.wf)
        for ch in range(result.n_channels):
            assert abs(result.data[ch].std() - 1.0) < 1e-5

    def test_trim_shorter(self):
        pp = Preprocessor(demean=False, normalize=False, trim_length=500)
        result = pp.process(self.wf)
        assert result.n_samples == 500

    def test_pad_longer(self):
        pp = Preprocessor(demean=False, normalize=False, trim_length=1500)
        result = pp.process(self.wf)
        assert result.n_samples == 1500
