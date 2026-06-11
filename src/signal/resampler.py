# src/signal/resampler.py
# D:\Github\Mk-project\seismic-phase-picker\src\signal\resampler.py
"""波形重采样器 — 统一到目标采样率。"""

import numpy as np
from scipy import signal
from ..io import Waveform


class Resampler:
    """波形重采样器。

    将输入波形重采样到目标采样率。使用抗混叠滤波。
    """

    def __init__(self, target_sr: float):
        """
        Parameters
        ----------
        target_sr : float
            目标采样率 (Hz)。
        """
        self.target_sr = target_sr

    def resample(self, waveform: Waveform) -> Waveform:
        """重采样单个 Waveform。

        Parameters
        ----------
        waveform : Waveform
            输入波形。

        Returns
        -------
        Waveform
            重采样后的波形。
        """
        if abs(waveform.sampling_rate - self.target_sr) < 1e-6:
            return waveform

        ratio = self.target_sr / waveform.sampling_rate
        new_length = int(waveform.n_samples * ratio)

        # signal.resample 沿 axis=-1 重采样
        if waveform.data.ndim == 1:
            resampled = signal.resample(waveform.data, new_length)
        else:
            resampled = signal.resample(waveform.data, new_length, axis=1)

        return Waveform(
            data=resampled.astype(np.float32),
            sampling_rate=self.target_sr,
            starttime=waveform.starttime,
            station=waveform.station,
            channel=waveform.channel,
        )
