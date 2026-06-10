# src/data/label_reader.py
# D:\Github\Mk-project\seismic-phase-picker\src\data\label_reader.py

import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PhaseLabel:
    """单个震相标签 (拾取时刻 + 类型)。"""
    time: float       # 绝对时间 (Unix timestamp)
    phase: str        # "P" / "S" 等


@dataclass
class EventLabels:
    """单个事件的标注集合。"""
    event_id: str
    phases: List[PhaseLabel]


class LabelReader:
    """震相标签读取器。

    支持标准格式 (CSV) 及 STEAD / 谛听数据集格式。
    """

    def __init__(self, format: str = "csv"):
        """
        Parameters
        ----------
        format : str
            标签格式: "csv" / "stead" / "di_ting"。
        """
        self.format = format

    def read(self, path: str) -> List[EventLabels]:
        """读取标签文件。

        Parameters
        ----------
        path : str
            标签文件路径。

        Returns
        -------
        List[EventLabels]
            事件标签列表。
        """
        if self.format == "csv":
            return self._read_csv(path)
        elif self.format == "stead":
            return self._read_stead(path)
        elif self.format == "di_ting":
            return self._read_di_ting(path)
        else:
            raise ValueError(f"Unknown label format: {self.format}")

    def _read_csv(self, path: str) -> List[EventLabels]:
        import pandas as pd
        df = pd.read_csv(path)
        events: Dict[str, List[PhaseLabel]] = {}
        # Expected columns: event_id, phase_time, phase_type
        for _, row in df.iterrows():
            eid = row["event_id"]
            if eid not in events:
                events[eid] = []
            events[eid].append(PhaseLabel(
                time=float(row["phase_time"]),
                phase=str(row["phase_type"]),
            ))
        return [EventLabels(event_id=k, phases=v) for k, v in events.items()]

    def _read_stead(self, path: str) -> List[EventLabels]:
        raise NotImplementedError("STEAD label reader not yet implemented")

    def _read_di_ting(self, path: str) -> List[EventLabels]:
        raise NotImplementedError("谛听 label reader not yet implemented")
