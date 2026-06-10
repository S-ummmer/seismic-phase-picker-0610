# scripts/evaluate_folder.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\evaluate_folder.py

"""
批量评估：对文件夹内所有波形执行推理并与标签比对。

用法:
    python scripts/evaluate_folder.py --config config.yaml --data data/raw/ --labels data/labels/
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import SeismicPipeline
from src.data.reader import Waveform
from src.data.label_reader import LabelReader
from src.evaluation.matcher import PhaseMatcher
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.grading import EventGrader

import numpy as np
import h5py


def load_waveform_hdf5(path: str) -> Waveform:
    with h5py.File(path, "r") as f:
        data = f["data"][:]
        sr = f["data"].attrs.get("sampling_rate", 100.0)
        start_time = f["data"].attrs.get("start_time", 0.0)
    return Waveform(data=data, sampling_rate=sr, start_time=start_time)


def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Directory with waveform files")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    args = parser.parse_args()

    pipeline = SeismicPipeline(args.config)
    matcher = PhaseMatcher(tolerance=pipeline.tolerance)
    calculator = MetricsCalculator()
    grader = EventGrader()

    # 加载所有标签
    label_reader = LabelReader(format="csv")
    all_labels = label_reader.read(args.labels)
    label_map = {e.event_id: e for e in all_labels}

    data_dir = Path(args.data)
    h5_files = sorted(data_dir.glob("*.h5")) or sorted(data_dir.glob("*.hdf5"))
    print(f"Found {len(h5_files)} waveform files")

    all_summaries = []
    for h5f in h5_files:
        event_id = h5f.stem
        wf = load_waveform_hdf5(str(h5f))
        picks = pipeline.run_inference(wf)

        gt = label_map.get(event_id)
        if gt is None:
            print(f"  [{event_id}] No labels found, skipping")
            continue

        pred_tuples = [(p.time, p.phase, p.probability) for p in picks]
        summary = matcher.match(pred_tuples, gt.phases)
        metrics = calculator.compute(summary)
        grade = grader.grade(summary)

        print(f"  [{event_id}] TP={metrics.n_tp} FP={metrics.n_fp} FN={metrics.n_fn} "
              f"F1={metrics.f1:.3f} Grade={grade.grade.value}")
        all_summaries.append(summary)

    # 全局评估
    from src.evaluation.matcher import MatchSummary
    global_summary = MatchSummary()
    for s in all_summaries:
        global_summary.tp.extend(s.tp)
        global_summary.fp.extend(s.fp)
        global_summary.fn.extend(s.fn)

    global_metrics = calculator.compute(global_summary)
    global_grade = grader.grade(global_summary)

    print(f"\n=== Global Results ===")
    print(f"  TP={global_metrics.n_tp} FP={global_metrics.n_fp} FN={global_metrics.n_fn}")
    print(f"  Precision={global_metrics.precision:.4f} Recall={global_metrics.recall:.4f} F1={global_metrics.f1:.4f}")
    print(f"  Mean Error={global_metrics.mean_time_error:.4f}s Median Error={global_metrics.median_time_error:.4f}s")
    print(f"  Grade: {global_grade.grade.value} - {global_grade.grade_label}")

    for phase, stats in global_metrics.per_phase.items():
        print(f"  {phase}: Prec={stats['precision']:.4f} Rec={stats['recall']:.4f} F1={stats['f1']:.4f}")


if __name__ == "__main__":
    main()
