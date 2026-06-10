# src/data/reader.py — Waveform 对象 + 时间映射
# D:\Github\Mk-project\seismic-phase-picker\src\data\reader.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class Waveform:
    """地震波形数据结构。

    封装原始波形数据及其元信息，统一不同数据源的接口。

    Attributes
    ----------
    data : np.ndarray
        波形数据，shape 为 (n_channels, n_samples)。
    sampling_rate : float
        采样率 (Hz)。
    start_time : float
        波形起始时间 (Unix timestamp)。
    channel_names : tuple of str
        通道名称，如 ("Z", "N", "E")。
    station_id : Optional[str]
        台站标识。
    meta : Dict
        附加元信息 (事件ID、震中距等)。
    """
    data: np.ndarray               # (n_channels, n_samples)
    sampling_rate: float           # Hz
    start_time: float              # Unix timestamp
    channel_names: tuple = ("Z", "N", "E")
    station_id: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def time_at_index(self, index: int) -> float:
        """将采样索引映射到绝对时间 (Unix timestamp)。

        Parameters
        ----------
        index : int
            采样点索引。

        Returns
        -------
        float
            该采样点对应的绝对时间。
        """
        return self.start_time + index / self.sampling_rate

    def index_at_time(self, time: float) -> int:
        """将绝对时间映射到最近的采样索引。

        Parameters
        ----------
        time : float
            绝对时间 (Unix timestamp)。

        Returns
        -------
        int
            最近的采样点索引 (裁剪到 [0, n_samples-1])。
        """
        offset = (time - self.start_time) * self.sampling_rate
        return int(np.clip(round(offset), 0, self.n_samples - 1))

    def time_to_sample(self, relative_time: float) -> int:
        """将相对时间 (秒, 以 start_time 为 0) 映射到采样索引。

        Parameters
        ----------
        relative_time : float
            相对于波形起始的时间偏移 (秒)。

        Returns
        -------
        int
            采样点索引。
        """
        offset = relative_time * self.sampling_rate
        return int(np.clip(round(offset), 0, self.n_samples - 1))

    def sample_to_time(self, index: int) -> float:
        """将采样索引映射到相对时间 (秒)。

        Parameters
        ----------
        index : int
            采样点索引。

        Returns
        -------
        float
            相对于波形起始的时间偏移 (秒)。
        """
        return index / self.sampling_rate

    def __repr__(self) -> str:
        return (
            f"Waveform(station={self.station_id}, "
            f"shape={self.data.shape}, "
            f"sr={self.sampling_rate} Hz, "
            f"duration={self.duration:.1f}s)"
        )
