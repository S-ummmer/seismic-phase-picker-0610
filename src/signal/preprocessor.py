# src/signal/preprocessor.py
# D:\Github\Mk-project\seismic-phase-picker\src\signal\preprocessor.py
"""波形信号预处理流水线。

顺序: demean → detrend → taper → bandpass → normalize → trim
"""

import numpy as np
from ..io import Waveform


class Preprocessor:
    """波形预处理流水线。

    支持选项：去均值 / 去趋势 / 尖灭 / 带通滤波 / Z-score 归一化 / 长度裁剪。
    所有操作均在单通道或多通道 Waveform 上统一执行。
    """

    def __init__(
        self,
        demean: bool = True,
        detrend: bool = True,
        taper: bool = True,
        taper_alpha: float = 0.05,
        normalize: bool = True,
        bandpass: tuple = None,       # (low, high) Hz, 如 (1, 20)
        trim_length: int = None,      # 目标采样点数
        sampling_rate: float = 100.0,
    ):
        self.demean = demean
        self.detrend = detrend
        self.taper = taper
        self.taper_alpha = taper_alpha
        self.normalize = normalize
        self.bandpass = bandpass
        self.trim_length = trim_length
        self.sampling_rate = sampling_rate

    def process(self, waveform: Waveform) -> Waveform:
        """对 Waveform 执行完整预处理链。

        Parameters
        ----------
        waveform : Waveform
            输入波形（原始 counts 或物理量）。

        Returns
        -------
        Waveform
            预处理后的波形。
        """
        data = waveform.data.copy()

        # 1. 去均值
        if self.demean:
            if data.ndim == 1:
                data = data - data.mean()
            else:
                data = data - data.mean(axis=1, keepdims=True)

        # 2. 去趋势
        if self.detrend:
            from scipy.signal import detrend as scipy_detrend
            if data.ndim == 1:
                data = scipy_detrend(data)
            else:
                data = np.apply_along_axis(scipy_detrend, 1, data)

        # 3. 尖灭 (Tukey window) — 防止滤波边界伪影
        if self.taper:
            from scipy.signal.windows import tukey
            n_samps = data.shape[-1]
            win = tukey(n_samps, alpha=self.taper_alpha)
            if data.ndim == 1:
                data = data * win
            else:
                data = data * win[np.newaxis, :]

        # 4. 带通滤波
        if self.bandpass is not None:
            from scipy.signal import butter, sosfiltfilt
            low, high = self.bandpass
            nyquist = waveform.sampling_rate / 2
            sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
            if data.ndim == 1:
                data = sosfiltfilt(sos, data)
            else:
                data = np.apply_along_axis(lambda x: sosfiltfilt(sos, x), 1, data)

        # 5. Z-score 归一化（每通道独立）
        if self.normalize:
            if data.ndim == 1:
                std = data.std()
                if std > 0:
                    data = data / std
            else:
                std = data.std(axis=1, keepdims=True)
                std[std == 0] = 1.0
                data /= std

        # 6. 长度裁剪 / 补零
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
