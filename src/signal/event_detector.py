# src/signal/event_detector.py
# D:\Github\Mk-project\seismic-phase-picker\src\signal\event_detector.py
"""STA/LTA 事件检测器 — 从连续波形中检测地震事件窗口。

适用场景：路径 A — 连续 MSEED 波形 → 事件检测 → 截取窗口 → 震相拾取。
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..io import Waveform


@dataclass
class EventWindow:
    """检测到的事件窗口。"""
    start_idx: int      # 起始样本索引
    end_idx: int        # 结束样本索引
    trigger_idx: int    # STA/LTA 触发峰值位置
    sta_lta_max: float  # 峰值 STA/LTA 比值
    start_time: float   # 绝对起始时间 (Unix timestamp)
    duration: float     # 窗口时长 (秒)


class EventDetector:
    """STA/LTA 事件检测器。

    经典的短时窗/长时窗能量比方法。
    在连续波形上检测高于阈值的区间，合并相邻触发，提取事件窗口。

    用法:
        detector = EventDetector(sta=1.0, lta=30.0, threshold=3.0)
        windows = detector.detect(waveform)  # List[EventWindow]
        for win in windows:
            event_wf = detector.extract_window(waveform, win)
    """

    def __init__(
        self,
        sta: float = 1.0,          # 短时窗 (秒)
        lta: float = 30.0,         # 长时窗 (秒)
        threshold: float = 3.0,    # STA/LTA 触发阈值
        detrigger: float = 1.5,    # 去触发阈值
        min_event_gap: float = 5.0,     # 合并相邻事件的最小间隔 (秒)
        pre_window: float = 5.0,         # 触发前的 padding (秒)
        post_window: float = 30.0,       # 触发后的 padding (秒)
        min_duration: float = 10.0,      # 事件最小时长 (秒)
    ):
        self.sta = sta
        self.lta = lta
        self.threshold = threshold
        self.detrigger = detrigger
        self.min_event_gap = min_event_gap
        self.pre_window = pre_window
        self.post_window = post_window
        self.min_duration = min_duration

    def detect(self, waveform: Waveform) -> List[EventWindow]:
        """在连续波形上检测事件窗口。

        Parameters
        ----------
        waveform : Waveform
            连续波形数据（推荐用 Z 分量或三分量的能量）。

        Returns
        -------
        List[EventWindow]
        """
        sr = waveform.sampling_rate
        # 对所有通道取平均能量
        if waveform.data.ndim == 1:
            energy = waveform.data ** 2
        else:
            energy = np.mean(waveform.data ** 2, axis=0)

        # STA/LTA 计算
        sta_samples = max(2, int(self.sta * sr))
        lta_samples = int(self.lta * sr)
        sta_lta = self._compute_sta_lta(energy, sta_samples, lta_samples)

        # 触发区间检测
        triggers = self._find_triggers(sta_lta, sr)

        # 合并 + 扩展
        windows = self._merge_and_pad(triggers, waveform, sr)
        return windows

    def extract_window(
        self, waveform: Waveform, window: EventWindow
    ) -> Waveform:
        """从连续波形中截取事件窗口。

        Parameters
        ----------
        waveform : Waveform
            连续波形。
        window : EventWindow
            检测到的事件窗口。

        Returns
        -------
        Waveform
            截取后的窗口波形。
        """
        data = waveform.data[..., window.start_idx:window.end_idx]
        starttime = waveform.starttime + window.start_idx / waveform.sampling_rate
        return Waveform(
            data=data.copy().astype(np.float32),
            sampling_rate=waveform.sampling_rate,
            starttime=starttime,
            station=waveform.station,
            channel=waveform.channel,
        )

    # ── private ──────────────────────────────────────────

    def _compute_sta_lta(
        self, energy: np.ndarray, sta_n: int, lta_n: int
    ) -> np.ndarray:
        """递归计算 STA/LTA。"""
        n = len(energy)
        sta_lta = np.zeros(n)

        # 初始 LTA
        lta = np.mean(energy[:lta_n]) if lta_n < n else np.mean(energy)
        for i in range(lta_n, n):
            sta = np.mean(energy[max(0, i - sta_n):i])
            lta = lta + (energy[i] - energy[i - lta_n]) / lta_n
            if lta < 1e-12:
                lta = 1e-12
            sta_lta[i] = sta / lta

        return sta_lta

    def _find_triggers(
        self, sta_lta: np.ndarray, sr: float
    ) -> List[Tuple[int, int, float]]:
        """找出所有触发区间 (start_idx, end_idx, max_value)。"""
        triggered = False
        trigger_start = 0
        triggers = []

        de_samples = int(self.detrigger * sr)
        de_samples = max(de_samples, 1)

        for i in range(len(sta_lta)):
            if not triggered:
                if sta_lta[i] >= self.threshold:
                    triggered = True
                    trigger_start = i
            else:
                # 低于去触发阈值
                if sta_lta[i] < self.detrigger:
                    triggered = False
                    max_val = np.max(sta_lta[trigger_start:i])
                    triggers.append((trigger_start, i, float(max_val)))

        if triggered:
            max_val = np.max(sta_lta[trigger_start:])
            triggers.append((trigger_start, len(sta_lta) - 1, float(max_val)))

        return triggers

    def _merge_and_pad(
        self,
        triggers: List[Tuple[int, int, float]],
        waveform: Waveform,
        sr: float,
    ) -> List[EventWindow]:
        """合并时间接近的触发，加 padding，过滤太短的事件。"""
        if not triggers:
            return []

        gap_samples = int(self.min_event_gap * sr)
        merged = []
        current_start, current_end, current_max = triggers[0]

        for t_start, t_end, t_max in triggers[1:]:
            if t_start - current_end <= gap_samples:
                current_end = t_end
                current_max = max(current_max, t_max)
            else:
                merged.append((current_start, current_end, current_max))
                current_start, current_end, current_max = t_start, t_end, t_max
        merged.append((current_start, current_end, current_max))

        windows = []
        pre_n = int(self.pre_window * sr)
        post_n = int(self.post_window * sr)
        min_samples = int(self.min_duration * sr)

        for start, end, max_val in merged:
            start_pad = max(0, start - pre_n)
            end_pad = min(waveform.n_samples, end + post_n)
            if end_pad - start_pad < min_samples:
                continue

            # 找触发峰值位置
            trigger_idx = start + np.argmax(waveform.data[
                ..., start:min(end, waveform.n_samples)
            ].mean(axis=0))

            start_time = waveform.starttime + start_pad / sr
            windows.append(EventWindow(
                start_idx=start_pad,
                end_idx=end_pad,
                trigger_idx=trigger_idx,
                sta_lta_max=max_val,
                start_time=start_time,
                duration=(end_pad - start_pad) / sr,
            ))

        return windows
