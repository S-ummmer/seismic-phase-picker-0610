# src/data/resampler.py
# D:\Github\Mk-project\seismic-phase-picker\src\data\resampler.py

import numpy as np
from scipy import signal
from .reader import Waveform


class Resampler:
    """波形重采样器。

    将输入波形重采样到目标采样率。
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
            重采样后的波形 (新的 data array, 其余元信息保持不变)。
        """
        if waveform.sampling_rate == self.target_sr:
            return waveform

        ratio = self.target_sr / waveform.sampling_rate
        new_length = int(waveform.n_samples * ratio)

        resampled_data = signal.resample(
            waveform.data, new_length, axis=1
        )

        return Waveform(
            data=resampled_data.astype(np.float32),
            sampling_rate=self.target_sr,
            start_time=waveform.start_time,
            channel_names=waveform.channel_names,
            station_id=waveform.station_id,
            meta=waveform.meta,
        )
