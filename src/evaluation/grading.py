# src/evaluation/grading.py
# D:\Github\Mk-project\seismic-phase-picker\src\evaluation\grading.py

from dataclasses import dataclass
from enum import Enum
from .matcher import MatchSummary
from .metrics import EvaluationMetrics


class Grade(Enum):
    """评估等级。

    仅基于 TP 的拾取精度 (时间误差)评定，FP/FN 不纳入分级。
    """
    A = "A"  # 极佳
    B = "B"  # 良好
    C = "C"  # 一般
    D = "D"  # 较差


@dataclass
class GradeResult:
    grade: Grade
    grade_label: str
    mean_error: float
    median_error: float
    tp_count: int


class EventGrader:
    """震相拾取质量分级器。

    分级仅考虑正确拾取 (TP) 的时间误差，不受 FP/FN 影响。
    FP/FN 由 metrics.py 单独统计。
    """

    def __init__(
        self,
        thresholds: dict = None,
    ):
        """
        Parameters
        ----------
        thresholds : dict
            分级阈值，默认:
            A: median_error <= 0.05s
            B: median_error <= 0.10s
            C: median_error <= 0.20s
            D: median_error >  0.20s
        """
        self.thresholds = thresholds or {
            "A": 0.05,
            "B": 0.10,
            "C": 0.20,
        }

    def grade(self, summary: MatchSummary) -> GradeResult:
        """对匹配结果进行分级。

        Parameters
        ----------
        summary : MatchSummary
            匹配结果。

        Returns
        -------
        GradeResult
            分级结果。
        """
        tp = summary.tp
        if not tp:
            return GradeResult(
                grade=Grade.D,
                grade_label="无正确拾取 (No TP)",
                mean_error=0.0,
                median_error=0.0,
                tp_count=0,
            )

        errors = [abs(r.time_error) for r in tp if r.time_error is not None]
        mean_err = sum(errors) / len(errors)
        median_err = sorted(errors)[len(errors) // 2]

        if median_err <= self.thresholds["A"]:
            grade = Grade.A
            label = f"极佳 (median <= {self.thresholds['A']}s)"
        elif median_err <= self.thresholds["B"]:
            grade = Grade.B
            label = f"良好 (median <= {self.thresholds['B']}s)"
        elif median_err <= self.thresholds["C"]:
            grade = Grade.C
            label = f"一般 (median <= {self.thresholds['C']}s)"
        else:
            grade = Grade.D
            label = f"较差 (median > {self.thresholds['C']}s)"

        return GradeResult(
            grade=grade,
            grade_label=label,
            mean_error=mean_err,
            median_error=median_err,
            tp_count=len(tp),
        )
