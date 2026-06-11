#!/usr/bin/env python
# scripts/evaluate_stead.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\evaluate_stead.py
"""
STEAD 数据集完整评估：推理 + 指标计算。

用法:
    python scripts/evaluate_stead.py                    # 默认 test split, 全部 traces
    python scripts/evaluate_stead.py --max 500          # 限制 500 条
    python scripts/evaluate_stead.py --split dev         # dev split
    python scripts/evaluate_stead.py --threshold 0.3    # 调阈值
"""

import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.hdf5_reader import Hdf5Reader
from src.signal.resampler import Resampler
from src.signal.preprocessor import Preprocessor
from src.models.wrapper import ModelWrapper
from src.postprocess.peak_detector import PeakDetector
from src.evaluation.matcher import PhaseMatcher, MatchSummary
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.grading import EventGrader
from src.evaluation.scorer import PhaseScorer, PhaseScore


def _sliding_window_inference(
    model: ModelWrapper,
    data: np.ndarray,
    window_size: int = 3001,
    step_size: int = 1500,
    batch_size: int = 32,
) -> np.ndarray:
    """滑动窗口推理：将长波形切分为重叠窗口，批量推理后平均重叠区域。

    Parameters
    ----------
    data : (C, N) ndarray
    window_size : int
    step_size : int

    Returns
    -------
    (3, N) ndarray — softmax probability for each sample
    """
    C, N = data.shape

    # 生成窗口起始位置
    starts = list(range(0, max(1, N - window_size + 1), step_size))
    if not starts or starts[-1] + window_size < N:
        starts.append(max(0, N - window_size))

    # 累积器
    prob_accum = np.zeros((3, N), dtype=np.float64)
    count_accum = np.zeros(N, dtype=np.float64)

    # 批量处理
    for batch_start in range(0, len(starts), batch_size):
        batch_ends = starts[batch_start:batch_start + batch_size]
        batch_windows = []
        for s in batch_ends:
            win = data[:, s:s + window_size]
            batch_windows.append(win)
        batch_arr = np.stack(batch_windows)  # (B, C, window_size)

        outputs = model.predict_prob_batch(batch_arr)  # (B, 3, window_size)

        for i, s in enumerate(batch_ends):
            prob_accum[:, s:s + window_size] += outputs[i]
            count_accum[s:s + window_size] += 1.0

    count_accum[count_accum == 0] = 1.0
    return (prob_accum / count_accum).astype(np.float32)


def evaluate_once(
    hdf5_path: str, csv_path: str,
    model: ModelWrapper, detector: PeakDetector,
    resampler: Resampler, preprocessor: Preprocessor,
    matcher: PhaseMatcher, calculator: MetricsCalculator, grader: EventGrader,
    scorer: PhaseScorer,
    split: str = "test", max_traces: int = 0,
) -> dict:
    """主评估流程。"""

    # ── 推理 ──────────────────────────────────────
    print(f"Loading {split} split from STEAD ...")
    all_rows = []
    n_processed = 0
    n_ps_labeled = 0      # 同时有 P 和 S 标签的 trace
    n_p_only = 0
    n_s_only = 0
    n_no_label = 0

    with Hdf5Reader(hdf5_path, csv_path) as reader:
        traces = reader.read_split(split, max_traces)
        n_total = len(traces)
        print(f"  Total traces: {n_total}")
        print(f"  Running inference ...")

        t0 = datetime.now()
        for wf, info in traces:
            # 统计标签
            has_p = info.p_sample is not None
            has_s = info.s_sample is not None
            if has_p and has_s:
                n_ps_labeled += 1
            elif has_p:
                n_p_only += 1
            elif has_s:
                n_s_only += 1
            else:
                n_no_label += 1

            # 推理链（滑动窗口覆盖完整 trace）
            wf = resampler.resample(wf)
            wf = preprocessor.process(wf)
            probs = _sliding_window_inference(
                model, wf.data,
                window_size=model.expected_length,
                step_size=model.expected_length // 2,
                batch_size=32,
            )
            picks = detector.detect(
                probabilities=probs[1:],
                phase_labels=["P", "S"],
                time_fn=wf.time_at_index,
            )

            # 格式化预测
            pred_tuples = [(p.time, p.phase, p.probability) for p in picks]

            # 格式化标签（绝对时间 = start_time + sample/sr）
            gt_labels = []
            if has_p:
                gt_labels.append((
                    info.start_time + info.p_sample / info.sampling_rate,
                    "P",
                ))
            if has_s:
                gt_labels.append((
                    info.start_time + info.s_sample / info.sampling_rate,
                    "S",
                ))

            all_rows.append({
                "trace_name": info.trace_name,
                "station": f"{info.network}.{info.station}",
                "picks": picks,
                "pred_tuples": pred_tuples,
                "gt_tuples": gt_labels,
                "has_p": has_p,
                "has_s": has_s,
            })

            n_processed += 1
            if n_processed % 500 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                speed = n_processed / elapsed
                eta = (n_total - n_processed) / speed
                print(f"  [{n_processed}/{n_total}] {speed:.1f} traces/s, ETA {eta:.0f}s")

    total_sec = (datetime.now() - t0).total_seconds()
    print(f"  Done in {total_sec:.1f}s ({n_total / total_sec:.1f} traces/s)")

    # ── 评估 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Label distribution:")
    print(f"  P+S: {n_ps_labeled}  P-only: {n_p_only}  S-only: {n_s_only}  No label: {n_no_label}")
    print(f"{'='*60}")

    # 全局匹配
    global_summary = MatchSummary()
    per_trace_metrics = []

    # 需要用 PhaseLabel for matching
    from src.data.label_reader import PhaseLabel

    for row in all_rows:
        gt_phases = [PhaseLabel(time=t, phase=ph) for t, ph in row["gt_tuples"]]
        summary = matcher.match(row["pred_tuples"], gt_phases)
        metrics = calculator.compute(summary)
        g = grader.grade(summary)

        per_trace_metrics.append({
            "trace_name": row["trace_name"],
            **metrics.__dict__,
        })

        global_summary.tp.extend(summary.tp)
        global_summary.fp.extend(summary.fp)
        global_summary.fn.extend(summary.fn)

    # 全局指标
    global_metrics = calculator.compute(global_summary)
    global_grade = grader.grade(global_summary)

    # 按震相统计
    phase_stats = _per_phase_stats(global_summary)

    # ── 输出 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GLOBAL METRICS ({split} split, {n_total} traces)")
    print(f"{'='*60}")
    print(f"  Predictions:  {global_metrics.n_predictions}")
    print(f"  Ground Truth: {global_metrics.n_ground_truth}")
    print(f"  TP={global_metrics.n_tp}  FP={global_metrics.n_fp}  FN={global_metrics.n_fn}")
    print()
    print(f"  Precision:  {global_metrics.precision:.4f}")
    print(f"  Recall:     {global_metrics.recall:.4f}")
    print(f"  F1 Score:   {global_metrics.f1:.4f}")
    print()
    print(f"  Time Error (TP only):")
    print(f"    Mean:   {global_metrics.mean_time_error:.4f}s")
    print(f"    Median: {global_metrics.median_time_error:.4f}s")
    print(f"    Std:    {global_metrics.std_time_error:.4f}s")
    print()
    print(f"  Grade: {global_grade.grade.value} — {global_grade.grade_label}")
    print(f"    TP count: {global_grade.tp_count}")
    print(f"    Mean err: {global_grade.mean_error:.4f}s")
    print(f"    Median err: {global_grade.median_error:.4f}s")
    print()

    # ── 按震相 ──────────────────────────────────────
    print(f"  Per-Phase Metrics:")
    print(f"  {'Phase':<6} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*56}")
    for ph in ["P", "S"]:
        if ph in global_metrics.per_phase:
            s = global_metrics.per_phase[ph]
            print(f"  {ph:<6} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5} "
                  f"{s['precision']:>10.4f} {s['recall']:>10.4f} {s['f1']:>10.4f}")
    print()

    # ── 时间误差分布 ────────────────────────────────
    p_errors = [abs(r.time_error) for r in global_summary.tp
                if r.predicted_phase == "P" and r.time_error is not None]
    s_errors = [abs(r.time_error) for r in global_summary.tp
                if r.predicted_phase == "S" and r.time_error is not None]

    if p_errors:
        p_errors = sorted(p_errors)
        print(f"  P-wave time error percentiles (n={len(p_errors)}):")
        for pct in [50, 75, 90, 95, 99]:
            idx = int(len(p_errors) * pct / 100)
            print(f"    P{pct}: {p_errors[min(idx, len(p_errors)-1)]:.4f}s")

    if s_errors:
        s_errors = sorted(s_errors)
        print(f"  S-wave time error percentiles (n={len(s_errors)}):")
        for pct in [50, 75, 90, 95, 99]:
            idx = int(len(s_errors) * pct / 100)
            print(f"    P{pct}: {s_errors[min(idx, len(s_errors)-1)]:.4f}s")

    # ── 比赛评分 ──────────────────────────────────────
    score_result = scorer.score(global_summary)
    print(f"\n{'='*60}")
    print(f"COMPETITION SCORE")
    print(f"{'='*60}")
    print(f"  Predictions: {score_result.n_predictions}  Ground Truth: {score_result.n_ground_truth}")
    print(f"  TP: {score_result.n_tp}  FP: {score_result.n_fp}  FN: {score_result.n_fn}")
    print()
    print(f"  TP Score Sum:    {score_result.tp_total_score:6.2f} / {score_result.tp_max_possible:.0f} (max possible)")
    print(f"  Count Penalty:   {score_result.count_penalty:6.2f}")
    if score_result.count_penalty > 0:
        allowed = score_result.n_ground_truth * PhaseScorer.COUNT_TOLERANCE
        excess = abs(score_result.n_predictions - score_result.n_ground_truth) - allowed
        print(f"    (diff={abs(score_result.n_predictions-score_result.n_ground_truth)}, "
              f"allowed=±{allowed:.0f}, excess={excess:.0f} × {PhaseScorer.COUNT_PENALTY_PER})")
    print(f"  {'─'*40}")
    print(f"  FINAL SCORE:     {score_result.total_score:6.2f}")
    print()
    for ph in ["P", "S"]:
        if ph in score_result.per_phase:
            s = score_result.per_phase[ph]
            print(f"  {ph}-wave: matched={s['n_matched']}, "
                  f"score_sum={s['score_sum']:.2f}, "
                  f"score_mean={s['score_mean']:.4f}, "
                  f"perfect_ratio={s['perfect_ratio']:.1%}")
            print(f"           mean_err={s['mean_error_s']:.4f}s, "
                  f"median_err={s['median_error_s']:.4f}s")

    # ── 保存 CSV ──────────────────────────────────────
    out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 预测 CSV
    pred_csv = out_dir / f"predictions_{split}_{ts}.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trace_name", "station", "phase", "time_s", "probability",
                     "gt_time_s", "error_s", "is_tp"])
        for row in all_rows:
            for p in row["picks"]:
                w.writerow([row["trace_name"], row["station"],
                            p.phase, f"{p.time:.4f}", f"{p.probability:.4f}",
                            "", "", ""])

    # 汇总 JSON（用 CSV 代替 JSON 方便查看）
    summary_csv = out_dir / f"summary_{split}_{ts}.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_traces", n_total])
        w.writerow(["n_predictions", global_metrics.n_predictions])
        w.writerow(["n_ground_truth", global_metrics.n_ground_truth])
        w.writerow(["TP", global_metrics.n_tp])
        w.writerow(["FP", global_metrics.n_fp])
        w.writerow(["FN", global_metrics.n_fn])
        w.writerow(["precision", f"{global_metrics.precision:.4f}"])
        w.writerow(["recall", f"{global_metrics.recall:.4f}"])
        w.writerow(["f1", f"{global_metrics.f1:.4f}"])
        w.writerow(["mean_time_error_s", f"{global_metrics.mean_time_error:.4f}"])
        w.writerow(["median_time_error_s", f"{global_metrics.median_time_error:.4f}"])
        w.writerow(["std_time_error_s", f"{global_metrics.std_time_error:.4f}"])
        w.writerow(["grade", global_grade.grade.value])
        w.writerow(["grade_label", global_grade.grade_label])
        w.writerow(["score_tp_sum", f"{score_result.tp_total_score:.2f}"])
        w.writerow(["score_tp_max", f"{score_result.tp_max_possible:.0f}"])
        w.writerow(["score_count_penalty", f"{score_result.count_penalty:.2f}"])
        w.writerow(["score_final", f"{score_result.total_score:.2f}"])
        for ph in ["P", "S"]:
            if ph in score_result.per_phase:
                s = score_result.per_phase[ph]
                w.writerow([f"score_{ph}_matched", s["n_matched"]])
                w.writerow([f"score_{ph}_sum", f"{s['score_sum']:.2f}"])
                w.writerow([f"score_{ph}_mean", f"{s['score_mean']:.4f}"])
                w.writerow([f"score_{ph}_perfect_ratio", f"{s['perfect_ratio']:.4f}"])
        for ph in ["P", "S"]:
            if ph in global_metrics.per_phase:
                s = global_metrics.per_phase[ph]
                w.writerow([f"{ph}_tp", s["tp"]])
                w.writerow([f"{ph}_fp", s["fp"]])
                w.writerow([f"{ph}_fn", s["fn"]])
                w.writerow([f"{ph}_precision", f"{s['precision']:.4f}"])
                w.writerow([f"{ph}_recall", f"{s['recall']:.4f}"])
                w.writerow([f"{ph}_f1", f"{s['f1']:.4f}"])

    print(f"  Predictions saved: {pred_csv}")
    print(f"  Summary saved:     {summary_csv}")
    print()

    return {
        "metrics": global_metrics,
        "grade": global_grade,
        "per_phase": global_metrics.per_phase,
    }


def _per_phase_stats(summary: MatchSummary) -> dict:
    """手工按震相统计（独立于 MetricsCalculator）。"""
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "errors": []})
    for r in summary.tp:
        stats[r.predicted_phase]["tp"] += 1
        if r.time_error is not None:
            stats[r.predicted_phase]["errors"].append(abs(r.time_error))
    for r in summary.fp:
        stats[r.predicted_phase]["fp"] += 1
    for r in summary.fn:
        stats[r.phase]["fn"] += 1

    result = {}
    for ph, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        errors = s["errors"]
        result[ph] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_err": float(np.mean(errors)) if errors else 0.0,
            "median_err": float(np.median(errors)) if errors else 0.0,
            "n_errors": len(errors),
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STEAD Evaluation")
    parser.add_argument("--hdf5", default="data/raw/stead/waveforms.hdf5")
    parser.add_argument("--csv", default="data/raw/stead/metadata.csv")
    parser.add_argument("--model", default="models/phasenet.jit")
    parser.add_argument("--info", default="models/model_info.json")
    parser.add_argument("--split", default="test",
                        help="train/dev/test")
    parser.add_argument("--max", type=int, default=0,
                        help="Max traces (0=all)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Peak detection threshold")
    parser.add_argument("--prominence", type=float, default=0.1,
                        help="Peak detection prominence")
    parser.add_argument("--min-distance", type=int, default=25,
                        help="Peak detection min distance (samples)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Matching tolerance (seconds)")
    args = parser.parse_args()

    hdf5_path = PROJECT_ROOT / args.hdf5
    csv_path = PROJECT_ROOT / args.csv
    if not hdf5_path.exists():
        print(f"ERROR: HDF5 not found: {hdf5_path}")
        sys.exit(1)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"HDF5: {hdf5_path}")
    print(f"CSV:  {csv_path}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Threshold: {args.threshold}  Prominence: {args.prominence}  MinDist: {args.min_distance}")
    print(f"Tolerance: {args.tolerance}s")
    print()

    # 初始化组件
    model = ModelWrapper(model_path=args.model, info_path=args.info)
    detector = PeakDetector(min_distance=args.min_distance, prominence=args.prominence, threshold=args.threshold)
    resampler = Resampler(target_sr=model.expected_sampling_rate)
    preprocessor = Preprocessor(demean=True, detrend=True, taper=True, normalize=True)
    matcher = PhaseMatcher(tolerance=args.tolerance)
    calculator = MetricsCalculator()
    grader = EventGrader()
    scorer = PhaseScorer()

    result = evaluate_once(
        hdf5_path=str(hdf5_path),
        csv_path=str(csv_path),
        model=model, detector=detector,
        resampler=resampler, preprocessor=preprocessor,
        matcher=matcher, calculator=calculator, grader=grader,
        scorer=scorer,
        split=args.split,
        max_traces=args.max,
    )


if __name__ == "__main__":
    main()
