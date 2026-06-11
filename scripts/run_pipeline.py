#!/usr/bin/env python
# scripts/run_pipeline.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\run_pipeline.py
"""
震相拾取流水线 — 自动识别格式 (MSEED / HDF5+CSV)。

用法:
    # MSEED 路径 A（连续波形）
    python scripts/run_pipeline.py --data_dir data/raw/2021/

    # HDF5+CSV 路径 B（预截取窗口）
    python scripts/run_pipeline.py --data_dir data/raw/stead/

    # 限制数量 / 调阈值
    python scripts/run_pipeline.py --data_dir data/raw/stead/ --max_files 100 --threshold 0.3
"""

import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.mseed_reader import MseedReader, Waveform
from src.io.hdf5_reader import Hdf5Reader
from src.signal.resampler import Resampler
from src.signal.preprocessor import Preprocessor
from src.models.wrapper import ModelWrapper
from src.postprocess.peak_detector import PeakDetector


# ---------------------------------------------------------------------------
# 格式检测
# ---------------------------------------------------------------------------

def detect_format(data_dir: Path) -> str:
    """自动检测数据格式。

    Returns
    -------
    str
        "mseed" / "hdf5" / "unknown"
    """
    if (data_dir / "waveforms.hdf5").exists() and (data_dir / "metadata.csv").exists():
        return "hdf5"
    mseed_files = list(data_dir.rglob("*.mseed"))
    if mseed_files:
        return "mseed"
    # 无拓展名的文件可能是 miniSEED
    raw_files = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix == ""]
    if raw_files:
        return "mseed"
    return "unknown"


def list_mseed_files(data_dir: Path) -> list:
    """收集所有 miniSEED 文件。"""
    month_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    all_files = []
    if month_dirs:
        for md in month_dirs:
            sub = sorted(md.glob("*"))
            all_files.extend([f for f in sub if f.is_file()])
    else:
        all_files = sorted([f for f in data_dir.glob("*") if f.is_file()])
    return all_files


# ---------------------------------------------------------------------------
# 路径 A: MSEED 推理
# ---------------------------------------------------------------------------

def process_mseed(
    filepath: Path, reader: MseedReader,
    resampler: Resampler, preprocessor: Preprocessor,
    model: ModelWrapper, detector: PeakDetector,
) -> tuple:
    """对单个 MSEED 文件推理，返回 (filename, picks, n_sta, n_skip)。"""
    traces = reader.read(str(filepath))
    stations = reader.group_station_3ch(traces)

    all_picks = []
    for wf in stations:
        wf = resampler.resample(wf)
        wf = preprocessor.process(wf)
        probs = model.predict_prob(wf.data)
        picks = detector.detect(
            probabilities=probs[1:], phase_labels=["P", "S"],
            time_fn=wf.time_at_index,
        )
        for p in picks:
            all_picks.append((wf.station, p))

    return filepath.name, all_picks, len(stations), len(traces) - len(stations) * 3


# ---------------------------------------------------------------------------
# 路径 B: HDF5 推理
# ---------------------------------------------------------------------------

def process_hdf5(
    hdf5_path: str, csv_path: str,
    resampler: Resampler, preprocessor: Preprocessor,
    model: ModelWrapper, detector: PeakDetector,
    split: str = "test", max_traces: int = 0,
) -> list:
    """对 HDF5+CSV 数据集批量推理。

    Returns
    -------
    list of (trace_name, station, picks, labels_dict)
    """
    results = []
    with Hdf5Reader(hdf5_path, csv_path) as reader:
        traces = reader.read_split(split, max_traces)
        for wf, info in traces:
            wf = resampler.resample(wf)
            wf = preprocessor.process(wf)
            probs = model.predict_prob(wf.data)
            picks = detector.detect(
                probabilities=probs[1:], phase_labels=["P", "S"],
                time_fn=wf.time_at_index,
            )
            results.append({
                "trace_name": info.trace_name,
                "station": wf.station,
                "picks": picks,
                "p_sample": info.p_sample,
                "s_sample": info.s_sample,
            })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Seismic Phase Picker")
    parser.add_argument("--data_dir", default="data/raw/2021",
                        help="Data directory (MSEED or HDF5)")
    parser.add_argument("--model", default="models/phasenet.jit")
    parser.add_argument("--info", default="models/model_info.json")
    parser.add_argument("--output_dir", default="outputs/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--split", default="test", help="HDF5 split (train/dev/test)")
    parser.add_argument("--csv", help="HDF5 CSV path (override auto-detect)")
    parser.add_argument("--hdf5", help="HDF5 path (override auto-detect)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    fmt = detect_format(data_dir)
    if fmt == "unknown":
        print(f"ERROR: cannot detect data format under {data_dir}")
        print("  MSEED: expect *.mseed or no-extension files")
        print("  HDF5:  expect waveforms.hdf5 + metadata.csv")
        sys.exit(1)

    print(f"Detected format: {fmt}")
    print(f"Data dir:  {data_dir}")
    print(f"Threshold: {args.threshold}")
    print()

    # 初始化组件
    model = ModelWrapper(model_path=args.model, info_path=args.info)
    resampler = Resampler(target_sr=model.expected_sampling_rate)
    preprocessor = Preprocessor(demean=True, detrend=True, taper=True, normalize=True)
    detector = PeakDetector(min_distance=50, prominence=0.3, threshold=args.threshold)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 路径 A: MSEED ─────────────────────────────────
    if fmt == "mseed":
        all_files = list_mseed_files(data_dir)
        if not all_files:
            print(f"ERROR: no files found under {data_dir}")
            sys.exit(1)
        if args.max_files > 0:
            all_files = all_files[:args.max_files]
        print(f"Files to process: {len(all_files)}")

        reader = MseedReader()
        csv_path = out_dir / f"picks_{timestamp}.csv"
        csv_f = open(csv_path, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["filename", "station", "phase", "time", "probability", "index"])

        total_picks = total_sta = total_skip = files_with = 0
        t0 = datetime.now()
        for i, fpath in enumerate(all_files, 1):
            fname, picks, n_sta, n_skip = process_mseed(
                fpath, reader, resampler, preprocessor, model, detector,
            )
            for sta, p in picks:
                csv_w.writerow([fname, sta, p.phase,
                                f"{p.time:.4f}", f"{p.probability:.4f}", p.index])
            total_picks += len(picks)
            total_sta += n_sta
            total_skip += n_skip
            if picks:
                files_with += 1
            elapsed = (datetime.now() - t0).total_seconds()
            eta = elapsed / i * (len(all_files) - i)
            pflag = "+" if picks else "-"
            print(f"  [{i:3d}/{len(all_files)}] {pflag} {fname[:50]:50s}  "
                  f"{len(picks):2d}p  {n_sta:2d}sta  ETA {eta:5.0f}s")
        csv_f.close()

        total_sec = (datetime.now() - t0).total_seconds()
        print()
        print("=" * 60)
        print(f"Done in {total_sec:.1f}s")
        print(f"  Files processed:  {len(all_files)}")
        print(f"  Files with picks: {files_with}")
        print(f"  Total picks:      {total_picks}")
        print(f"  Stations:         {total_sta} (skipped traces: {total_skip})")
        print(f"  Predictions CSV:  {csv_path}")

    # ── 路径 B: HDF5 ─────────────────────────────────
    else:
        hdf5_path = args.hdf5 or str(data_dir / "waveforms.hdf5")
        csv_path = args.csv or str(data_dir / "metadata.csv")
        if not Path(hdf5_path).exists():
            print(f"ERROR: HDF5 not found: {hdf5_path}")
            sys.exit(1)

        print(f"HDF5: {hdf5_path}")
        print(f"CSV:  {csv_path}")
        print(f"Split: {args.split}")
        print()

        results = process_hdf5(
            hdf5_path, csv_path, resampler, preprocessor, model, detector,
            split=args.split, max_traces=args.max_files,
        )

        # 输出 CSV
        pred_csv = out_dir / f"picks_hdf5_{timestamp}.csv"
        csv_f = open(pred_csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["trace_name", "station", "phase", "time", "probability", "index"])
        total_picks = 0
        for r in results:
            for p in r["picks"]:
                csv_w.writerow([r["trace_name"], r["station"],
                                p.phase, f"{p.time:.4f}",
                                f"{p.probability:.4f}", p.index])
                total_picks += 1
        csv_f.close()

        traces_with_picks = sum(1 for r in results if r["picks"])
        print()
        print("=" * 60)
        print(f"Processed {len(results)} traces ({args.split} split)")
        print(f"  Traces with picks: {traces_with_picks}")
        print(f"  Total picks:       {total_picks}")
        print(f"  Predictions CSV:   {pred_csv}")

    print()


if __name__ == "__main__":
    main()
