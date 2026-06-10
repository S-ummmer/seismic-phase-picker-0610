#!/usr/bin/env python
# scripts/run_pipeline.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\run_pipeline.py
"""
批量震相拾取流水线 — miniSEED 输入 → CSV 输出。

用法:
    # 批量处理 2021 年全部月份
    python scripts/run_pipeline.py --data_dir data/raw/2021/

    # 只处理指定月份
    python scripts/run_pipeline.py --data_dir data/raw/2021/01

    # 只处理前 3 个文件 (快速测试)
    python scripts/run_pipeline.py --data_dir data/raw/2021/ --max_files 3
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reader import WaveformReader, Waveform
from src.data.resampler import Resampler
from src.data.preprocessor import Preprocessor
from src.models.wrapper import ModelWrapper
from src.postprocess.peak_detector import PeakDetector, PickedPhase


# ---------------------------------------------------------------------------
# 三分量合并
# ---------------------------------------------------------------------------

def group_station_3ch(waveforms):
    """按台站合并三分量 (ENZ)，返回合并后的 Waveform 列表。"""
    groups = {}
    for wf in waveforms:
        sta = wf.station
        groups.setdefault(sta, []).append(wf)

    result = []
    for sta, wfs in groups.items():
        # 找 Z / 1(E) / 2(N)
        z_wf = e_wf = n_wf = None
        for wf in wfs:
            ch = wf.channel.strip()
            parts = ch.split()[-1]
            if parts == "Z":
                z_wf = wf
            elif parts == "1":
                e_wf = wf
            elif parts == "2":
                n_wf = wf
        if z_wf is None or e_wf is None or n_wf is None:
            continue

        min_len = min(z_wf.n_samples, e_wf.n_samples, n_wf.n_samples)
        stacked = np.stack([
            e_wf.data[0, :min_len],
            n_wf.data[0, :min_len],
            z_wf.data[0, :min_len],
        ], axis=0)

        result.append(Waveform(
            data=stacked.astype(np.float32),
            sampling_rate=z_wf.sampling_rate,
            starttime=z_wf.starttime,
            station=sta,
            channel="ENZ",
        ))
    return result


# ---------------------------------------------------------------------------
# 单文件推理
# ---------------------------------------------------------------------------

def process_file(
    filepath: Path,
    reader: WaveformReader,
    resampler: Resampler,
    preprocessor: Preprocessor,
    model: ModelWrapper,
    detector: PeakDetector,
    phase_labels: list,
) -> tuple:
    """对单个 miniSEED 文件执行完整推理链。

    Returns
    -------
    (filename, list_of_picks, n_stations, n_skipped)
    """
    waveforms = reader.read(str(filepath))

    # 合并三分量
    stations = group_station_3ch(waveforms)

    all_picks = []
    for wf in stations:
        wf = resampler.resample(wf)
        wf = preprocessor.process(wf)

        probs = model.predict_prob(wf.data)  # (3, N): Noise/P/S
        picks = detector.detect(
            probabilities=probs[1:],  # (2, N) — 跳 Noise，只取 P/S
            phase_labels=phase_labels,
            time_fn=wf.time_at_index,
        )
        for p in picks:
            all_picks.append((wf.station, p))

    n_stations = len(stations)
    n_skipped = len(waveforms) - len(stations) * 3
    return filepath.name, all_picks, n_stations, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Seismic Phase Picker — Batch Inference")
    parser.add_argument("--data_dir", default="data/raw/2021",
                        help="Data root directory (default: data/raw/2021)")
    parser.add_argument("--model", default="models/phasenet.jit",
                        help="TorchScript model path")
    parser.add_argument("--info", default="models/model_info.json",
                        help="Model info JSON path")
    parser.add_argument("--output_dir", default="outputs/predictions",
                        help="Predictions output directory")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Pick probability threshold (default: 0.5)")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max files to process (0=all)")
    parser.add_argument("--skip_station_stats", action="store_true",
                        help="Skip per-station verbose output")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    # 收集所有文件
    if data_dir.is_dir():
        # 检查是 month dirs 还是 flat files
        month_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        all_files = []
        for md in month_dirs:
            all_files.extend(sorted(md.glob("*")))
        if not all_files:
            # 可能 flat 单层
            all_files = sorted(data_dir.glob("*"))
            all_files = [f for f in all_files if f.is_file()]
    else:
        all_files = [data_dir]

    if not all_files:
        print(f"ERROR: no files found under {data_dir}")
        sys.exit(1)

    if args.max_files > 0:
        all_files = all_files[:args.max_files]

    print(f"Files to process: {len(all_files)}")
    print(f"Threshold: {args.threshold}")
    print(f"Model: {args.model}")
    print()

    # 初始化组件
    model = ModelWrapper(model_path=args.model, info_path=args.info)
    reader = WaveformReader()
    resampler = Resampler(target_sr=model.expected_sampling_rate)
    preprocessor = Preprocessor(demean=True, normalize=True)
    detector = PeakDetector(
        min_distance=50, prominence=0.3, threshold=args.threshold,
    )
    phase_labels = ["P", "S"]

    # 准备输出
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"picks_{timestamp}.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["filename", "station", "phase", "time", "probability", "index"])

    # 统计
    total_picks = 0
    total_stations = 0
    total_skipped = 0
    files_with_picks = 0

    t0 = datetime.now()
    for i, fpath in enumerate(all_files, 1):
        fname, picks, n_sta, n_skip = process_file(
            fpath, reader, resampler, preprocessor, model, detector, phase_labels,
        )
        for sta, p in picks:
            csv_w.writerow([fname, sta, p.phase,
                            f"{p.time:.4f}", f"{p.probability:.4f}", p.index])

        total_picks += len(picks)
        total_stations += n_sta
        total_skipped += n_skip
        if picks:
            files_with_picks += 1

        # 进度输出
        pflag = "+" if picks else "-"
        elapsed = (datetime.now() - t0).total_seconds()
        rate = elapsed / i
        eta = rate * (len(all_files) - i)
        print(f"  [{i:3d}/{len(all_files)}] {pflag} {fname[:50]:50s}  "
              f"{len(picks):2d} picks, {n_sta:2d} sta  "
              f"ETA {eta:5.0f}s"
              + (f"  e.g. {picks[0][1].phase}@{picks[0][1].probability:.2f}" if picks else ""))

    csv_f.close()

    # 汇总
    total_sec = (datetime.now() - t0).total_seconds()
    print()
    print("=" * 60)
    print(f"Done in {total_sec:.1f}s")
    print(f"  Files processed:  {len(all_files)}")
    print(f"  Files with picks: {files_with_picks}")
    print(f"  Total picks:      {total_picks}")
    print(f"  Stations:         {total_stations} (skipped traces: {total_skipped})")
    print(f"  Predictions CSV:  {csv_path}")


if __name__ == "__main__":
    main()
