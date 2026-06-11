# src/evaluation/scorer.py
# D:\Github\Mk-project\seismic-phase-picker\src\evaluation\scorer.py
"""
震相评分系统 — 按时间误差和数量误差综合打分。

P 波规则：误差 ≤ 0.1s → 1 分；0.1~1s → 线性衰减；≥ 1s → 0 分。
S 波规则：误差 ≤ 0.2s → 1 分；0.2~2s → 线性衰减；≥ 2s → 0 分。
数量误差：预测数与真实数偏差在 5% 内不扣分；超出部分每个扣 0.5 分。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .matcher import MatchSummary, MatchResult


@dataclass
class PhaseScore:
    """单条 trace 或全局的评分结果。"""

    # ── 基础统计 ──
    n_predictions: int = 0
    n_ground_truth: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0

    # ── 分数 ──
    tp_total_score: float = 0.0        # TP 累计得分
    tp_max_possible: float = 0.0       # TP 满分 (每个 TP 满分 = 1)
    count_penalty: float = 0.0         # 数量惩罚 (正数)
    total_score: float = 0.0           # 最终得分 = tp_total_score - count_penalty

    # ── 分震相明细 ──
    per_phase: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.per_phase:
            self.per_phase = {}


class PhaseScorer:
    """比赛风格震相评分器。

    使用方式::

        scorer = PhaseScorer()
        result = scorer.score(summary)
        print(f"Score: {result.total_score:.2f}")
    """

    # P 波评分参数
    P_PERFECT = 0.1     # ≤ 此误差满分
    P_ZERO = 1.0         # ≥ 此误差零分

    # S 波评分参数
    S_PERFECT = 0.2
    S_ZERO = 2.0

    # 数量误差容忍比例
    COUNT_TOLERANCE = 0.05
    COUNT_PENALTY_PER = 0.5

    def _phase_score(self, error: float, phase: str) -> float:
        """计算单个匹配震相的得分 (0~1)。"""
        error = abs(error)
        if phase == "P":
            if error <= self.P_PERFECT:
                return 1.0
            if error >= self.P_ZERO:
                return 0.0
            return 1.0 - (error - self.P_PERFECT) / (self.P_ZERO - self.P_PERFECT)
        elif phase == "S":
            if error <= self.S_PERFECT:
                return 1.0
            if error >= self.S_ZERO:
                return 0.0
            return 1.0 - (error - self.S_PERFECT) / (self.S_ZERO - self.S_PERFECT)
        else:
            # 其他震相按 P 波规则处理
            return self._phase_score(error, "P")

    def _count_penalty(self, n_pred: int, n_gt: int) -> float:
        """计算数量误差惩罚。"""
        if n_gt == 0:
            # 无标注 — 按 "预测数为 0 才算无误差"
            if n_pred == 0:
                return 0.0
            return n_pred * self.COUNT_PENALTY_PER

        diff = abs(n_pred - n_gt)
        allowed = n_gt * self.COUNT_TOLERANCE
        if diff <= allowed:
            return 0.0
        excess = diff - allowed
        return excess * self.COUNT_PENALTY_PER

    def score(self, summary: MatchSummary) -> PhaseScore:
        """从匹配汇总计算评分。

        Parameters
        ----------
        summary : MatchSummary

        Returns
        -------
        PhaseScore
        """
        tp = summary.tp
        fp = summary.fp
        fn = summary.fn

        n_pred = len(tp) + len(fp)
        n_gt = len(tp) + len(fn)

        result = PhaseScore(
            n_predictions=n_pred,
            n_ground_truth=n_gt,
            n_tp=len(tp),
            n_fp=len(fp),
            n_fn=len(fn),
        )

        # ── TP 得分 ──
        tp_scores = []
        p_scores = []
        s_scores = []

        for r in tp:
            if r.time_error is None:
                continue
            s = self._phase_score(r.time_error, r.predicted_phase)
            tp_scores.append(s)
            if r.predicted_phase == "P":
                p_scores.append((abs(r.time_error), s))
            elif r.predicted_phase == "S":
                s_scores.append((abs(r.time_error), s))

        result.tp_total_score = sum(tp_scores)
        result.tp_max_possible = float(len(tp))  # 每个 TP 满分 1

        # ── 数量惩罚 ──
        result.count_penalty = self._count_penalty(n_pred, n_gt)
        result.total_score = result.tp_total_score - result.count_penalty

        # ── 分震相 ──
        for label, scores in [("P", p_scores), ("S", s_scores)]:
            if not scores:
                continue
            errors = [e for e, _ in scores]
            vals = [v for _, v in scores]
            result.per_phase[label] = {
                "n_matched": len(scores),
                "score_sum": sum(vals),
                "score_mean": float(np.mean(vals)),
                "score_median": float(np.median(vals)),
                "perfect_ratio": sum(1 for v in vals if v >= 0.999) / len(vals),
                "mean_error_s": float(np.mean(errors)),
                "median_error_s": float(np.median(errors)),
            }

        return result

    def score_per_event(self, all_summaries: List[Tuple[str, MatchSummary]]) -> List[Tuple[str, PhaseScore]]:
        """返回每个事件的逐个评分列表。

        Parameters
        ----------
        all_summaries : list of (name, MatchSummary)

        Returns
        -------
        list of (name, PhaseScore)
        """
        return [(name, self.score(summary)) for name, summary in all_summaries]
