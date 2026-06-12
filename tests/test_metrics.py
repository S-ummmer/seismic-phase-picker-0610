# tests/test_metrics.py
# D:\Github\Mk-project\seismic-phase-picker\tests\test_metrics.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.evaluation.matcher import PhaseMatcher, MatchSummary, MatchResult
from src.evaluation.metrics import MetricsCalculator
from src.io.label_reader import PhaseLabel


class TestMetricsCalculator:
    """指标计算器单元测试。"""

    def test_perfect_metrics(self):
        predictions = [(10.0, "P", 0.95), (15.0, "S", 0.90)]
        gt = [PhaseLabel(time=10.0, phase="P"), PhaseLabel(time=15.0, phase="S")]
        matcher = PhaseMatcher(tolerance=0.5)
        summary = matcher.match(predictions, gt)

        calc = MetricsCalculator()
        metrics = calc.compute(summary)

        assert metrics.n_tp == 2
        assert metrics.n_fp == 0
        assert metrics.n_fn == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_partial_metrics(self):
        predictions = [(10.0, "P", 0.95)]
        gt = [PhaseLabel(time=10.0, phase="P"), PhaseLabel(time=20.0, phase="S")]
        matcher = PhaseMatcher(tolerance=0.5)
        summary = matcher.match(predictions, gt)

        calc = MetricsCalculator()
        metrics = calc.compute(summary)

        assert metrics.n_tp == 1
        assert metrics.n_fp == 0
        assert metrics.n_fn == 1
        assert metrics.precision == 1.0
        assert metrics.recall == 0.5
        assert metrics.f1 == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_per_phase_stats(self):
        predictions = [(10.0, "P", 0.95), (15.0, "S", 0.90)]
        gt = [PhaseLabel(time=10.0, phase="P"), PhaseLabel(time=15.0, phase="S")]
        matcher = PhaseMatcher(tolerance=0.5)
        summary = matcher.match(predictions, gt)

        calc = MetricsCalculator()
        metrics = calc.compute(summary)

        assert "P" in metrics.per_phase
        assert "S" in metrics.per_phase
        assert metrics.per_phase["P"]["tp"] == 1
        assert metrics.per_phase["S"]["tp"] == 1

    def test_time_error_stats(self):
        predictions = [(10.0, "P", 0.95), (15.0, "S", 0.90)]
        gt = [PhaseLabel(time=10.1, phase="P"), PhaseLabel(time=14.9, phase="S")]
        matcher = PhaseMatcher(tolerance=0.5)
        summary = matcher.match(predictions, gt)

        calc = MetricsCalculator()
        metrics = calc.compute(summary)

        assert abs(metrics.mean_time_error - 0.1) < 1e-6
        assert abs(metrics.median_time_error - 0.1) < 1e-6
