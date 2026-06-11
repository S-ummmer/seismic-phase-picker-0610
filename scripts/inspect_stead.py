#!/usr/bin/env python
# scripts/inspect_stead.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\inspect_stead.py
"""
STEAD 数据集查看脚本 — 查看 HDF5 波形 + CSV 标签的结构和内容。

用法:
    python scripts/inspect_stead.py              # 概览
    python scripts/inspect_stead.py --detail     # 查看第 1 条 trace 细节
    python scripts/inspect_stead.py --index 99   # 查看第 99 条 trace 细节
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="STEAD dataset inspector")
    parser.add_argument("--detail", action="store_true", help="Show first trace waveform details")
    parser.add_argument("--index", type=int, default=0, help="Trace index to inspect (default: 0)")
    args = parser.parse_args()

    data_root = Path("data/raw/stead")
    csv_path = data_root / "metadata.csv"
    h5_path = data_root / "waveforms.hdf5"

    # ── 1. 元数据概览 ────────────────────────────────────
    print("=" * 65)
    print("  STEAD 元数据概览")
    print("=" * 65)
    df = pd.read_csv(csv_path)
    print(f"  CSV 行数:      {len(df):,}")
    print(f"  列名:          {', '.join(df.columns.tolist())}")
    print()
    print(f"  独立事件数:     {df['source_origin_time'].nunique()}")
    print(f"  独立台站数:     {df[['station_network_code','station_code']].drop_duplicates().shape[0]}")
    print(f"  P 波标签:       {df['trace_P_arrival_sample'].notna().sum():,}")
    print(f"  S 波标签:       {df['trace_S_arrival_sample'].notna().sum():,}")
    print(f"  同时有 P+S:      {((df['trace_P_arrival_sample'].notna()) & (df['trace_S_arrival_sample'].notna())).sum():,}")
    print(f"  采样率:         {df['trace_sampling_rate_hz'].value_counts().to_dict()}")
    print(f"  分量顺序:       {df['trace_component_order'].value_counts().to_dict()}")
    print(f"  数据划分:       {df['split'].value_counts().to_dict()}")
    print()

    # ── 2. HDF5 结构 ────────────────────────────────────
    print("=" * 65)
    print("  HDF5 文件结构")
    print("=" * 65)
    with h5py.File(h5_path, "r") as f_h5:
        print(f"  顶层 keys:      {list(f_h5.keys())}")
        data_g = f_h5["data"]
        buckets = sorted(data_g.keys(), key=lambda x: int(x.replace("bucket", "")))
        print(f"  Bucket 数量:    {len(buckets)}")
        total_traces = 0
        for b in buckets:
            ds = data_g[b]
            total_traces += ds.shape[0]
            print(f"    {b}: {ds.shape} (={ds.shape[0]} traces, {ds.shape[1]}ch, max {ds.shape[2]} samples)")
        print(f"  总 trace 数:    {total_traces:,}")
        print(f"  dtype:          {data_g[buckets[0]].dtype}")
        print()

        # ── 3. 单条 trace 详情 ─────────────────────────────
        if args.detail or args.index > 0:
            idx = args.index
            print("=" * 65)
            print(f"  Trace #{idx} 详情")
            print("=" * 65)
            row = df.iloc[idx]

            # 解析 trace_name → bucket + 切片
            trace_name = row["trace_name"]
            bucket_name, idx_str = trace_name.split("$")
            parts = idx_str.split(",")
            trace_idx = int(parts[0])
            n_samples = int(parts[2].split(":")[1])

            # 读取波形
            bucket_data = f_h5["data"][bucket_name]
            waveform = bucket_data[trace_idx, :, :n_samples]  # (3, N)

            # 标签
            p_samp = row["trace_P_arrival_sample"]
            s_samp = row["trace_S_arrival_sample"]

            print(f"  事件:           {row['source_origin_time']}")
            print(f"  台站:           {row['station_network_code']}.{row['station_code']}")
            print(f"  trace_name:     {trace_name}")
            print(f"  波形:           {waveform.shape} (3ch × {n_samples} samples @ {row['trace_sampling_rate_hz']}Hz)")
            print(f"  时长:           {n_samples / row['trace_sampling_rate_hz']:.1f}s")
            print(f"  划分:           {row['split']}")
            print()
            print(f"  ┌─ 标签 ──────────────────────────────────────")
            if pd.notna(p_samp):
                print(f"  │  P 波: sample={int(p_samp)} → {(waveform[2, int(p_samp)]):.2f} (Z分量振幅)")
            else:
                print(f"  │  P 波: (无)")
            if pd.notna(s_samp):
                print(f"  │  S 波: sample={int(s_samp)} → {(waveform[2, int(s_samp)]):.2f} (Z分量振幅)")
                if pd.notna(p_samp):
                    print(f"  │  Δt(P→S): {(s_samp - p_samp) / row['trace_sampling_rate_hz']:.2f}s")
            else:
                print(f"  │  S 波: (无)")
            print(f"  └─────────────────────────────────────────────")
            print()
            print(f"  各分量统计:")
            for i, ch in enumerate(["E", "N", "Z"]):
                c = waveform[i]
                print(f"    {ch}: mean={c.mean():8.2f}, std={c.std():8.2f}, "
                      f"min={c.min():8.2f}, max={c.max():8.2f}")
            print()
            print(f"  P 波附近 +-50 samples (Z分量):")
            p_win = waveform[2, int(p_samp)-50:int(p_samp)+51] if pd.notna(p_samp) else np.array([])
            if len(p_win) > 0:
                p_win_abs = np.abs(p_win)
                print(f"    绝对振幅 max={p_win_abs.max():.2f} @ +{p_win_abs.argmax()-50}")
            print(f"  S 波附近 +-50 samples (Z分量):")
            s_win = waveform[2, int(s_samp)-50:int(s_samp)+51] if pd.notna(s_samp) else np.array([])
            if len(s_win) > 0:
                s_win_abs = np.abs(s_win)
                print(f"    绝对振幅 max={s_win_abs.max():.2f} @ +{s_win_abs.argmax()-50}")

            # ── 完整 CSV 列打印 ─────────────────────────────
            print()
            print(f"  ┌─ 完整 CSV 列 ({len(df.columns)} 列) ──────────────────")
            for i, col in enumerate(df.columns):
                print(f"  │  [{i:2d}] {col}: {row[col]}")
            print(f"  └─────────────────────────────────────────────")


if __name__ == "__main__":
    main()
