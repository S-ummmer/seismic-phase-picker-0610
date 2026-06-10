# src/postprocess/peak_detector.py
# D:\Github\Mk-project\seismic-phase-picker\src\postprocess\peak_detector.py

import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PickedPhase:
    """一个拾取到的震相。"""
    time: float       # 绝对时间 (Unix timestamp)
    phase: str        # "P" / "S"
    probability: float
    index: int        # 采样索引


class PeakDetector:
    """基于概率曲线峰值检测的震相拾取器。"""

    def __init__(
        self,
        min_distance: int = 50,      # samples
        prominence: float = 0.3,
        threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        min_distance : int
            两个峰值之间的最小采样间隔。
        prominence : float
            scipy find_peaks 的 prominence 参数。
        threshold : float
            仅保留概率高于此值的峰值。
        """
        self.min_distance = min_distance
        self.prominence = prominence
        self.threshold = threshold

    def detect(
        self,
        probabilities: np.ndarray,          # (n_phases, n_samples)
        phase_labels: List[str],
        time_fn,                            # index -> absolute time
    ) -> List[PickedPhase]:
        """在所有震相通道上执行峰值检测。

        Parameters
        ----------
        probabilities : np.ndarray
            shape (n_phases, n_samples), 每通道概率曲线。
        phase_labels : List[str]
            震相标签列表, 如 ["P", "S"]。
        time_fn : callable
            f(index) -> float, 将采样索引映射为绝对时间。

        Returns
        -------
        List[PickedPhase]
            按时间排序的拾取列表。
        """
        picks = []
        for p_idx, label in enumerate(phase_labels):
            probs = probabilities[p_idx]
            peaks, props = find_peaks(
                probs,
                distance=self.min_distance,
                prominence=self.prominence,
                height=self.threshold,
            )
            for peak_idx in peaks:
                picks.append(PickedPhase(
                    time=time_fn(peak_idx),
                    phase=label,
                    probability=float(probs[peak_idx]),
                    index=int(peak_idx),
                ))
        picks.sort(key=lambda p: p.time)
        return picks
