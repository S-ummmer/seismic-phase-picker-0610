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
            # 对单通道或多通道都适用
            if data.ndim == 1:
                data = data - data.mean()
            else:
                data = data - data.mean(axis=1, keepdims=True)

        if self.detrend:
            from scipy.signal import detrend as scipy_detrend
            if data.ndim == 1:
                data = scipy_detrend(data)
            else:
                data = np.apply_along_axis(scipy_detrend, 1, data)

        if self.bandpass is not None:
            from scipy.signal import butter, sosfiltfilt
            low, high = self.bandpass
            nyquist = waveform.sampling_rate / 2
            sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
            if data.ndim == 1:
                data = sosfiltfilt(sos, data)
            else:
                data = np.apply_along_axis(lambda x: sosfiltfilt(sos, x), 1, data)

        if self.normalize:
            if data.ndim == 1:
                std = data.std()
                if std > 0:
                    data = data / std
            else:
                std = data.std(axis=1, keepdims=True)
                std[std == 0] = 1.0
                data /= std

        if self.trim_length is not None:
            ns = data.shape[-1]
            if ns > self.trim_length:
                data = data[..., :self.trim_length]
            elif ns < self.trim_length:
                pad_width = self.trim_length - ns
                if data.ndim == 1:
                    data = np.pad(data, (0, pad_width), mode="constant")
                else:
                    data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

        return Waveform(
            data=data.astype(np.float32),
            sampling_rate=waveform.sampling_rate,
            starttime=waveform.starttime,
            station=waveform.station,
            channel=waveform.channel,
        )
