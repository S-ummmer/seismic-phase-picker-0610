# src/io/mseed_reader.py
# D:\Github\Mk-project\seismic-phase-picker\src\io\mseed_reader.py
"""miniSEED 波形读取器。

读取 miniSEED 文件，返回 Waveform 列表。
支持 OBS 数据通道命名（H Z, S 1, SH 等）。
"""

import numpy as np
from obspy import read as obspy_read
from obspy.core.stream import Stream
from typing import Optional, Dict, List, Tuple

from . import Waveform


# 通道分量映射：OBS 通道后缀 → 标准三分量 (E, N, Z)
CHANNEL_MAP: Dict[str, Tuple[str, ...]] = {
    ("1", "2", "Z"): ("E", "N", "Z"),
    ("E", "N", "Z"): ("E", "N", "Z"),
    ("E", "N", "H"): ("E", "N", "Z"),
}


class MseedReader:
    """miniSEED 文件读取器。

    支持：
    - 标准台网 miniSEED（SAC 也可通过 ObsPy 读取）
    - OBS 实验数据（通道命名不规范，如 H Z / S 1 / SH）
    """

    def __init__(self, prefer_channels: Tuple[str, ...] = ("Z", "1", "2")):
        """
        Parameters
        ----------
        prefer_channels : tuple
            三分量合并时优先匹配的后缀顺序 (对应 E, N, Z)。
        """
        self.prefer_channels = prefer_channels

    def read(self, path: str, station: Optional[str] = None) -> List[Waveform]:
        """读取 MSEED 文件，返回所有 trace 的 Waveform 列表。

        Parameters
        ----------
        path : str
            miniSEED 文件路径。
        station : str, optional
            筛选指定台站。

        Returns
        -------
        List[Waveform]
            每个 trace 一个 Waveform 对象。
        """
        st = obspy_read(path)
        return self._stream_to_waveforms(st, station=station)

    def read_multiple(self, paths: List[str]) -> List[Waveform]:
        """批量读取多个文件并合并。"""
        waveforms = []
        for p in paths:
            waveforms.extend(self.read(p))
        return waveforms

    def read_event(self, event_path: str) -> Dict[str, List[Waveform]]:
        """读取事件文件，按台站分组。

        Returns
        -------
        Dict[str, List[Waveform]]
            {station_id: [channel_waveforms, ...]}
        """
        st = obspy_read(event_path)
        result: Dict[str, List[Waveform]] = {}
        for wf in self._stream_to_waveforms(st):
            if wf.station not in result:
                result[wf.station] = []
            result[wf.station].append(wf)
        return result

    def group_station_3ch(
        self, waveforms: List[Waveform]
    ) -> List[Waveform]:
        """将单通道 trace 按台站合并为三分量 (E, N, Z) Waveform。

        不含 3 通道的台站自动跳过。

        Parameters
        ----------
        waveforms : List[Waveform]
            单通道 trace 列表。

        Returns
        -------
        List[Waveform]
            每个台站一个 3 通道合并 Waveform (channel="ENZ")。
        """
        groups: Dict[str, List[Waveform]] = {}
        for wf in waveforms:
            groups.setdefault(wf.station, []).append(wf)

        result = []
        pref = self.prefer_channels  # ("Z", "1", "2")
        for sta, wfs in groups.items():
            # 按后缀找 Z / 1(E) / 2(N)
            z_wf = e_wf = n_wf = None
            for wf in wfs:
                ch = wf.channel.strip()
                parts = ch.split()
                suffix = parts[-1] if parts else ch
                if suffix == pref[0]:       # Z
                    z_wf = wf
                elif suffix == pref[1]:     # 1 → E
                    e_wf = wf
                elif suffix == pref[2]:     # 2 → N
                    n_wf = wf
            if z_wf is None or e_wf is None or n_wf is None:
                continue  # 三通道不全，跳过

            min_len = min(z_wf.n_samples, e_wf.n_samples, n_wf.n_samples)
            stacked = np.stack([
                e_wf.data[0, :min_len],
                n_wf.data[0, :min_len],
                z_wf.data[0, :min_len],
            ], axis=0)  # (3, N)

            result.append(Waveform(
                data=stacked.astype(np.float32),
                sampling_rate=z_wf.sampling_rate,
                starttime=z_wf.starttime,
                station=sta,
                channel="ENZ",
            ))
        return result

    # ── private ─────────────────────────────────────────

    def _stream_to_waveforms(
        self, st: Stream, station: Optional[str] = None
    ) -> List[Waveform]:
        waveforms = []
        for tr in st:
            sta = tr.stats.station or ""
            if station is not None and sta != station:
                continue
            starttime = float(tr.stats.starttime.timestamp)
            sr = float(tr.stats.sampling_rate)
            data = tr.data.astype(np.float32).copy()
            if data.ndim == 1:
                data = data[np.newaxis, :]  # (1, N)

            waveforms.append(Waveform(
                data=data,
                sampling_rate=sr,
                starttime=starttime,
                station=sta,
                channel=tr.stats.channel or "",
            ))
        return waveforms
