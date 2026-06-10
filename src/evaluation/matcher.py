# src/evaluation/matcher.py
# D:\Github\Mk-project\seismic-phase-picker\src\evaluation\matcher.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ..data.label_reader import PhaseLabel


@dataclass
class MatchResult:
    """单次匹配结果。"""
    predicted_time: float
    predicted_phase: str
    predicted_prob: float
    ground_truth_time: Optional[float] = None
    ground_truth_phase: Optional[str] = None
    time_error: Optional[float] = None     # pred - gt, seconds
    is_tp: bool = False
    is_fp: bool = False
    is_fn: bool = False


@dataclass
class MatchSummary:
    """匹配汇总。"""
    tp: List[MatchResult] = field(default_factory=list)
    fp: List[MatchResult] = field(default_factory=list)
    fn: List[PhaseLabel] = field(default_factory=list)


class PhaseMatcher:
    """震相匹配器。

    将预测的震相与标注震相按时间和类型匹配。
    """

    def __init__(self, tolerance: float = 0.5):
        """
        Parameters
        ----------
        tolerance : float
            时间容差 (秒)。预测值与标注值在此范围内视为匹配。
        """
        self.tolerance = tolerance

    def match(
        self,
        predictions,          # list of (time, phase_label, prob)
        ground_truth,         # list of PhaseLabel
    ) -> MatchSummary:
        """执行匹配。

        Parameters
        ----------
        predictions : list of tuple
            [(time, phase, prob), ...]
        ground_truth : list of PhaseLabel
            标注震相列表。

        Returns
        -------
        MatchSummary
            匹配结果汇总 (TP/FP/FN)。
        """
        summary = MatchSummary()
        remaining_gt = ground_truth.copy()

        for pred_time, pred_phase, pred_prob in predictions:
            best_match = None
            best_error = float("inf")
            for i, gt in enumerate(remaining_gt):
                if gt.phase != pred_phase:
                    continue
                error = abs(pred_time - gt.time)
                if error <= self.tolerance and error < best_error:
                    best_match = i
                    best_error = error

            if best_match is not None:
                gt = remaining_gt.pop(best_match)
                result = MatchResult(
                    predicted_time=pred_time,
                    predicted_phase=pred_phase,
                    predicted_prob=pred_prob,
                    ground_truth_time=gt.time,
                    ground_truth_phase=gt.phase,
                    time_error=pred_time - gt.time,
                    is_tp=True,
                )
                summary.tp.append(result)
            else:
                result = MatchResult(
                    predicted_time=pred_time,
                    predicted_phase=pred_phase,
                    predicted_prob=pred_prob,
                    is_fp=True,
                )
                summary.fp.append(result)

        summary.fn = remaining_gt
        return summary
