# tests/test_grading.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_grading.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.grading import EventGrader, Grade
from src.evaluation.matcher import MatchSummary, MatchResult


class TestEventGrader:
    """震相分级器单元测试。"""

    def test_grade_a(self):
        summary = MatchSummary()
        summary.tp = [
            MatchResult(predicted_time=10.0, predicted_phase="P", predicted_prob=0.95,
                        ground_truth_time=10.0, time_error=0.0, is_tp=True),
            MatchResult(predicted_time=20.0, predicted_phase="S", predicted_prob=0.90,
                        ground_truth_time=20.03, time_error=0.03, is_tp=True),
        ]
        grader = EventGrader()
        result = grader.grade(summary)
        assert result.grade == Grade.A
        assert result.tp_count == 2

    def test_grade_d_no_tp(self):
        summary = MatchSummary()
        grader = EventGrader()
        result = grader.grade(summary)
        assert result.grade == Grade.D
        assert result.tp_count == 0

    def test_fp_fn_do_not_affect_grade(self):
        """验证 FP/FN 不影响分级结果。"""
        summary = MatchSummary()
        summary.tp = [
            MatchResult(predicted_time=10.0, predicted_phase="P", predicted_prob=0.95,
                        ground_truth_time=10.0, time_error=0.0, is_tp=True),
        ]
        summary.fp = [MatchResult(predicted_time=99.0, predicted_phase="P", predicted_prob=0.7, is_fp=True)] * 100
        # 大量 FP 但 median_error 仍是 0 -> Grade A
        grader = EventGrader()
        result = grader.grade(summary)
        assert result.grade == Grade.A
