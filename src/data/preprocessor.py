# src/data/preprocessor.py
# D:\Github\Mk-project\seismic-phase-picker\src\data\preprocessor.py

import numpy as np
from .reader import Waveform


class Preprocessor:
    """波形预处理流水线。

    支持：去均值 / 去趋势 / 归一化 / 带通滤波 / 长度裁剪。
    """

    def __init__(
        self,
        demean: bool = True,
        detrend: bool = False,
        normalize: bool = True,
        bandpass: tuple = None,          # (low, high) Hz
        trim_length: int = None,         # target samples
        sampling_rate: float = 100.0,
    ):
        self.demean = demean
        self.detrend = detrend
        self.normalize = normalize
        self.bandpass = bandpass
        self.trim_length = trim_length
        self.sampling_rate = sampling_rate

    def process(self, waveform: Waveform) -> Waveform:
        """对 Waveform 执行预处理。

        Parameters
        ----------
        waveform : Waveform
            输入波形。

        Returns
        -------
        Waveform
            预处理后的波形。
        """
        data = waveform.data.copy()

        if self.demean:
            data -= data.mean(axis=1, keepdims=True)

        if self.detrend:
            from scipy.signal import detrend
            data = detrend(data, axis=1)

        if self.bandpass is not None:
            from scipy.signal import butter, sosfiltfilt
            low, high = self.bandpass
            nyquist = self.sampling_rate / 2
            sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
            data = sosfiltfilt(sos, data, axis=1)

        if self.normalize:
            std = data.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            data /= std

        if self.trim_length is not None:
            if data.shape[1] > self.trim_length:
                data = data[:, :self.trim_length]
            elif data.shape[1] < self.trim_length:
                pad_width = self.trim_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

        return Waveform(
            data=data.astype(np.float32),
            sampling_rate=waveform.sampling_rate,
            start_time=waveform.start_time,
            channel_names=waveform.channel_names,
            station_id=waveform.station_id,
            meta=waveform.meta,
        )
