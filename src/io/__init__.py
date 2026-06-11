# src/io/__init__.py
# D:\Github\Mk-project\seismic-phase-picker\src\io\__init__.py
"""I/O layer: format-specific readers. Output unified Waveform objects."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Waveform:
    """波形数据容器 — 格式无关的统一表示。

    Attributes
    ----------
    data : np.ndarray
        波形数组，shape (n_channels, n_samples)。单通道为 (1, N)。
    sampling_rate : float
        采样率 (Hz)。
    starttime : float
        起始时间 (Unix timestamp)。
    station : str
        台站名。
    channel : str
        通道名 / 分量顺序 (如 "ENZ", "ZNE", "HHZ")。
    """
    data: np.ndarray
    sampling_rate: float
    starttime: float
    station: str = ""
    channel: str = ""

    def __repr__(self):
        return (f"Waveform(station={self.station}, channel={self.channel}, "
                f"shape={self.data.shape}, sr={self.sampling_rate} Hz, "
                f"start={self.starttime})")

    def time_at_index(self, idx: int) -> float:
        """返回第 idx 个采样点的绝对时间 (秒)。"""
        return self.starttime + idx / self.sampling_rate

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    @property
    def n_channels(self) -> int:
        return self.data.shape[0] if self.data.ndim > 1 else 1


from .mseed_reader import MseedReader
from .hdf5_reader import Hdf5Reader
from .label_reader import LabelReader, PhaseLabel, EventLabels

__all__ = [
    "Waveform",
    "MseedReader",
    "Hdf5Reader",
    "LabelReader",
    "PhaseLabel",
    "EventLabels",
]
