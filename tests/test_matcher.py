# tests/test_matcher.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_matcher.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.matcher import PhaseMatcher
from src.data.label_reader import PhaseLabel


class TestPhaseMatcher:
    """震相匹配器单元测试。"""

    def setup_method(self):
        self.matcher = PhaseMatcher(tolerance=0.5)

    def test_perfect_match(self):
        predictions = [(10.0, "P", 0.9), (15.0, "S", 0.85)]
        gt = [PhaseLabel(time=10.0, phase="P"), PhaseLabel(time=15.0, phase="S")]
        summary = self.matcher.match(predictions, gt)
        assert len(summary.tp) == 2
        assert len(summary.fp) == 0
        assert len(summary.fn) == 0

    def test_false_positive(self):
        predictions = [(10.0, "P", 0.9), (99.0, "P", 0.7)]
        gt = [PhaseLabel(time=10.0, phase="P")]
        summary = self.matcher.match(predictions, gt)
        assert len(summary.tp) == 1
        assert len(summary.fp) == 1
        assert len(summary.fn) == 0

    def test_false_negative(self):
        predictions = [(10.0, "P", 0.9)]
        gt = [PhaseLabel(time=10.0, phase="P"), PhaseLabel(time=20.0, phase="S")]
        summary = self.matcher.match(predictions, gt)
        assert len(summary.tp) == 1
        assert len(summary.fp) == 0
        assert len(summary.fn) == 1
        assert summary.fn[0].phase == "S"

    def test_tolerance_boundary(self):
        predictions = [(10.49, "P", 0.9)]
        gt = [PhaseLabel(time=10.0, phase="P")]
        summary = self.matcher.match(predictions, gt)
        assert len(summary.tp) == 1  # within tolerance

    def test_outside_tolerance(self):
        predictions = [(10.51, "P", 0.9)]
        gt = [PhaseLabel(time=10.0, phase="P")]
        summary = self.matcher.match(predictions, gt)
        assert len(summary.tp) == 0
        assert len(summary.fp) == 1
