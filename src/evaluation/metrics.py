# src/evaluation/metrics.py
# D:\Github\Mk-project\seismic-phase-picker\src\evaluation\metrics.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from .matcher import MatchSummary


@dataclass
class EvaluationMetrics:
    """评估指标汇总。"""
    n_predictions: int = 0
    n_ground_truth: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    mean_time_error: float = 0.0
    std_time_error: float = 0.0
    median_time_error: float = 0.0

    per_phase: Dict[str, dict] = None

    def __post_init__(self):
        if self.per_phase is None:
            self.per_phase = {}


class MetricsCalculator:
    """评估指标计算器。"""

    def compute(self, summary: MatchSummary) -> EvaluationMetrics:
        """从匹配结果计算所有指标。

        Parameters
        ----------
        summary : MatchSummary
            匹配结果。

        Returns
        -------
        EvaluationMetrics
            计算后的指标。
        """
        tp = len(summary.tp)
        fp = len(summary.fp)
        fn = len(summary.fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        time_errors = [abs(r.time_error) for r in summary.tp if r.time_error is not None]
        mean_te = float(np.mean(time_errors)) if time_errors else 0.0
        std_te = float(np.std(time_errors)) if time_errors else 0.0
        median_te = float(np.median(time_errors)) if time_errors else 0.0

        # 按震相分类统计
        per_phase = {}
        all_phase_types = set()
        for r in summary.tp:
            all_phase_types.add(r.predicted_phase)
        for r in summary.fn:
            all_phase_types.add(r.phase)
        for phase in sorted(all_phase_types):
            p_tp = sum(1 for r in summary.tp if r.predicted_phase == phase)
            p_fp = sum(1 for r in summary.fp if r.predicted_phase == phase)
            p_fn = sum(1 for r in summary.fn if r.phase == phase)
            p_prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0.0
            p_rec = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0.0
            p_f1 = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0.0
            per_phase[phase] = {
                "tp": p_tp, "fp": p_fp, "fn": p_fn,
                "precision": p_prec, "recall": p_rec, "f1": p_f1,
            }

        return EvaluationMetrics(
            n_predictions=tp + fp,
            n_ground_truth=tp + fn,
            n_tp=tp, n_fp=fp, n_fn=fn,
            precision=precision, recall=recall, f1=f1,
            mean_time_error=mean_te,
            std_time_error=std_te,
            median_time_error=median_te,
            per_phase=per_phase,
        )
