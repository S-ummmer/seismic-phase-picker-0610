# src/io/hdf5_reader.py
# D:\Github\Mk-project\seismic-phase-picker\src\io\hdf5_reader.py
"""HDF5 + CSV 波形读取器（STEAD / 谛听 格式）。

读取 HDF5 波形 + CSV 元数据标签，返回 Waveform 列表。
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from . import Waveform


@dataclass
class Hdf5TraceInfo:
    """单条 trace 的元信息 + 标签。"""
    trace_name: str
    station: str
    network: str
    sampling_rate: float
    component_order: str        # e.g. "ZNE"
    p_sample: Optional[int]     # P 波样本索引（None = 无标签）
    s_sample: Optional[int]     # S 波样本索引
    start_time: float           # Unix timestamp
    n_samples: int              # trace 长度
    split: str                  # train/dev/test
    # 事件信息
    origin_time: str
    source_lat: float
    source_lon: float
    source_depth: float


class Hdf5Reader:
    """STEAD 格式 HDF5+CSV 读取器。

    用法:
        reader = Hdf5Reader("waveforms.hdf5", "metadata.csv")
        wf, info = reader.read(0)           # 按 CSV 索引读取
        wf, info = reader.read_by_station("7K.BOLO")  # 按台站
        waveforms = reader.read_range(0, 100)  # 批量
    """

    def __init__(self, hdf5_path: str, csv_path: str):
        """
        Parameters
        ----------
        hdf5_path : str
            HDF5 波形文件路径。
        csv_path : str
            CSV 元数据 + 标签文件路径。
        """
        self.hdf5_path = Path(hdf5_path)
        self.csv_path = Path(csv_path)
        self._df: Optional[pd.DataFrame] = None
        self._h5: Optional[h5py.File] = None

    # ── context manager ──────────────────────────────────

    def __enter__(self):
        self._h5 = h5py.File(self.hdf5_path, "r")
        self._df = pd.read_csv(self.csv_path)
        return self

    def __exit__(self, *args):
        if self._h5:
            self._h5.close()

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    @property
    def n_traces(self) -> int:
        return len(self.df)

    # ── 读取接口 ──────────────────────────────────────────

    def read(
        self, index: int, component_order: str = "ENZ"
    ) -> Tuple[Waveform, Hdf5TraceInfo]:
        """按 CSV 行索引读取单条 trace。

        自动解析 trace_name 格式 "bucketX$idx,:channels,:n_samples"，
        从 HDF5 中取出 (3, N) 波形。

        Returns
        -------
        (Waveform, Hdf5TraceInfo)
        """
        row = self.df.iloc[index]
        info = self._parse_row(row)
        data = self._load_trace(row)
        wave = self._build_waveform(data, info, component_order)
        return wave, info

    def read_range(
        self, start: int, end: int, component_order: str = "ENZ"
    ) -> List[Tuple[Waveform, Hdf5TraceInfo]]:
        """批量读取 [start, end) 范围的 trace。"""
        results = []
        for i in range(start, min(end, self.n_traces)):
            results.append(self.read(i, component_order))
        return results

    def read_by_station(
        self, station_id: str, component_order: str = "ENZ"
    ) -> List[Tuple[Waveform, Hdf5TraceInfo]]:
        """读取指定台站的全部 trace（跨不同事件）。

        Parameters
        ----------
        station_id : str
            格式 "NETWORK.STATION"，如 "7K.BOLO"。
        """
        mask = (self.df["station_network_code"].astype(str)
                + "." + self.df["station_code"].astype(str)) == station_id
        indices = self.df[mask].index.tolist()
        return [self.read(i, component_order) for i in indices]

    def read_split(
        self, split: str, max_traces: int = 0, component_order: str = "ENZ"
    ) -> List[Tuple[Waveform, Hdf5TraceInfo]]:
        """读取指定划分 (train/dev/test) 的全部 trace。

        Parameters
        ----------
        split : str
            "train" / "dev" / "test"。
        max_traces : int
            最大读取数，0 = 全部。
        """
        mask = self.df["split"] == split
        indices = self.df[mask].index.tolist()
        if max_traces > 0:
            indices = indices[:max_traces]
        return [self.read(i, component_order) for i in indices]

    # ── 内部方法 ──────────────────────────────────────────

    def _parse_row(self, row: pd.Series) -> Hdf5TraceInfo:
        return Hdf5TraceInfo(
            trace_name=str(row["trace_name"]),
            station=str(row["station_code"]),
            network=str(row.get("station_network_code", "")),
            sampling_rate=float(row["trace_sampling_rate_hz"]),
            component_order=str(row.get("trace_component_order", "ZNE")),
            p_sample=int(row["trace_P_arrival_sample"])
            if pd.notna(row.get("trace_P_arrival_sample")) else None,
            s_sample=int(row["trace_S_arrival_sample"])
            if pd.notna(row.get("trace_S_arrival_sample")) else None,
            start_time=float(pd.Timestamp(row["trace_start_time"]).timestamp()),
            n_samples=self._parse_n_samples(row["trace_name"]),
            split=str(row.get("split", "unknown")),
            origin_time=str(row.get("source_origin_time", "")),
            source_lat=float(row.get("source_latitude_deg", 0)),
            source_lon=float(row.get("source_longitude_deg", 0)),
            source_depth=float(row.get("source_depth_km", 0)),
        )

    @staticmethod
    def _parse_n_samples(trace_name: str) -> int:
        """从 "bucket0$0,:3,:15564" 中提取 15564。"""
        try:
            parts = trace_name.split("$")
            idx_spec = parts[1]  # "0,:3,:15564"
            # 取最后一个冒号后的数字
            n_str = idx_spec.split(",")[-1].split(":")[-1]
            return int(n_str)
        except Exception:
            return 0

    def _load_trace(self, row: pd.Series) -> np.ndarray:
        """从 HDF5 加载一条 trace 的 (3, N) 波形。"""
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

        trace_name = str(row["trace_name"])
        bucket, idx_spec = trace_name.split("$")
        parts = idx_spec.split(",")
        trace_idx = int(parts[0])
        n_samples = int(parts[-1].split(":")[-1])

        bucket_data = self._h5["data"][bucket]  # (Ntraces, 3, Nsamps)
        waveform = bucket_data[trace_idx, :3, :n_samples]

        return waveform.astype(np.float32)

    def _build_waveform(
        self, data: np.ndarray, info: Hdf5TraceInfo, target_order: str = "ENZ"
    ) -> Waveform:
        """按目标分量顺序重排通道。"""
        src_order = info.component_order.upper()  # e.g. "ZNE"
        if src_order != target_order:
            perm = []
            for ch in target_order:
                try:
                    perm.append(src_order.index(ch))
                except ValueError:
                    pass
            if len(perm) == 3:
                data = data[perm, :]

        return Waveform(
            data=data,
            sampling_rate=info.sampling_rate,
            starttime=info.start_time,
            station=f"{info.network}.{info.station}",
            channel=target_order,
        )

    # ── 统计工具 ──────────────────────────────────────────

    def stats(self) -> dict:
        """返回数据集统计概览。"""
        df = self.df
        return {
            "n_traces": len(df),
            "n_events": df["source_origin_time"].nunique(),
            "n_stations": df[["station_network_code", "station_code"]]
            .drop_duplicates().shape[0],
            "p_labels": df["trace_P_arrival_sample"].notna().sum(),
            "s_labels": df["trace_S_arrival_sample"].notna().sum(),
            "both_ps": ((df["trace_P_arrival_sample"].notna())
                        & (df["trace_S_arrival_sample"].notna())).sum(),
            "sampling_rates": df["trace_sampling_rate_hz"].value_counts().to_dict(),
            "splits": df.get("split", pd.Series(["unknown"] * len(df)))
            .value_counts().to_dict(),
        }
