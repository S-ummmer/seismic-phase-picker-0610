# src/data/reader.py
# D:\Github\Mk-project\seismic-phase-picker\src\data\reader.py

import numpy as np
from obspy import read as obspy_read
from obspy.core.stream import Stream
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Waveform:
    """波形数据容器。

    Attributes
    ----------
    data : np.ndarray
        波形数组，shape (n_channels, n_samples)。
    sampling_rate : float
        采样率 (Hz)。
    starttime : float
        起始时间 (Unix timestamp)。
    station : str
        台站名。
    channel : str
        通道名。
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


class WaveformReader:
    """波形文件读取器 (支持 miniSEED / SAC / HDF5)。"""

    def __init__(self, prefer_channels: Tuple[str, ...] = ("Z", "N", "E")):
        """
        Parameters
        ----------
        prefer_channels : tuple
            优先保留的通道分量 (用于组合同一台站的三分量)。
        """
        self.prefer_channels = prefer_channels

    def read(self, path: str, station: Optional[str] = None) -> List[Waveform]:
        """读取波形文件，返回 Waveform 列表。

        Parameters
        ----------
        path : str
            文件路径。
        station : str, optional
            指定台站名，只返回该台站的数据。

        Returns
        -------
        List[Waveform]
        """
        st = obspy_read(path)
        return self._stream_to_waveforms(st, station=station)

    def read_multiple(self, paths: List[str]) -> List[Waveform]:
        """读取多个文件并合并。"""
        waveforms = []
        for p in paths:
            waveforms.extend(self.read(p))
        return waveforms

    def _stream_to_waveforms(self, st: Stream, station: Optional[str] = None) -> List[Waveform]:
        import obspy
        waveforms = []
        for tr in st:
            net = tr.stats.network or ""
            sta = tr.stats.station or ""
            loc = tr.stats.location or ""
            cha = tr.stats.channel or ""

            if station is not None and sta != station:
                continue

            starttime = float(tr.stats.starttime.timestamp)
            sr = float(tr.stats.sampling_rate)
            data = tr.data.astype(np.float32).copy()

            # 确保是 2D: (n_channels, n_samples)
            if data.ndim == 1:
                data = data[np.newaxis, :]

            wf = Waveform(
                data=data,
                sampling_rate=sr,
                starttime=starttime,
                station=sta,
                channel=cha,
            )
            waveforms.append(wf)
        return waveforms

    def read_event(self, event_path: str) -> Dict[str, List[Waveform]]:
        """读取单个事件文件，按台站分组。

        Returns
        -------
        Dict[str, List[Waveform]]
            {station_id: [Waveform, ...]}
        """
        st = obspy_read(event_path)
        result: Dict[str, List[Waveform]] = {}
        for wf in self._stream_to_waveforms(st):
            key = wf.station
            if key not in result:
                result[key] = []
            result[key].append(wf)
        return result
