#!/usr/bin/env python
# scripts/inspect_mseed.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\inspect_mseed.py
"""
miniSEED 数据集查看脚本 — 查看 M170 波形文件的完整元信息和各 trace 头段。

用法:
    python scripts/inspect_mseed.py                                # 查看 01/ 下第一个文件
    python scripts/inspect_mseed.py --file data/raw/2021/01/xxx    # 指定文件
    python scripts/inspect_mseed.py --file xxx --index 2           # 只看第 2 条 trace
"""

import sys
import argparse
from pathlib import Path
from obspy import read, UTCDateTime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="miniSEED dataset inspector")
    parser.add_argument("--file", type=str, default=None,
                        help="MSEED file path (default: first file under data/raw/2021/01/)")
    parser.add_argument("--index", type=int, default=-1,
                        help="Show only this trace index (0-based); omit for all")
    args = parser.parse_args()

    # 自动找第一个文件
    if args.file is None:
        raw_dir = Path("data/raw/2021/01")
        files = sorted(raw_dir.glob("*"))
        if not files:
            print(f"未找到数据: {raw_dir}")
            return
        args.file = str(files[0])

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return

    print("=" * 72)
    print(f"  miniSEED 文件: {file_path.name}")
    print(f"  路径: {file_path}")
    print(f"  大小: {file_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 72)

    # 读取
    st = read(str(file_path))
    print(f"  Trace 总数: {len(st)}")
    print(f"  时间范围:   {st[0].stats.starttime} ~ {st[-1].stats.endtime}")
    print()

    # 概览表格
    print(f"  {'#':>3}  {'台站':<12} {'通道':<6} {'采样率':>6} {'样本数':>7} "
          f"{'起始时间':>28} {'数据dtype':<8} {'min':>12} {'max':>12}")
    print("  " + "-" * 135)

    for i, tr in enumerate(st):
        s = tr.stats
        print(f"  {i:>3}  {s.station:<12} {s.channel:<6} {s.sampling_rate:>6.1f} "
              f"{len(tr.data):>7} {str(s.starttime):>28} {str(tr.data.dtype):<8} "
              f"{tr.data.min():>12.2f} {tr.data.max():>12.2f}")

    print()
    print(f"  汇总: {len(st)} traces, "
          f"台站数={len(set(tr.stats.station for tr in st))}, "
          f"通道数={len(set(tr.stats.channel for tr in st))}")
    print()

    # ── Trace 详情 ──
    indices = range(len(st)) if args.index < 0 else [min(args.index, len(st)-1)]

    for idx in indices:
        tr = st[idx]
        s = tr.stats
        print("=" * 72)
        print(f"  Trace #{idx} 完整头段")
        print("=" * 72)
        print(f"  文件:     {file_path.name}")
        print(f"  trace id: {tr.id}")
        print()

        # 完整 stats 字典
        print(f"  ┌─ Stats 字段 ({len(s.__dict__.keys())} 项) ────────────────────────────")
        for key in sorted(s.__dict__.keys()):
            val = s.__dict__[key]
            print(f"  │  {key:<30} = {val}")
        print(f"  └{'─'*70}")

        # 波形数据摘要
        print()
        print(f"  ┌─ 波形数据 ───────────────────────────────────────")
        print(f"  │  dtype:     {tr.data.dtype}")
        print(f"  │  样本数:    {len(tr.data):,}")
        print(f"  │  时长:      {len(tr.data)/s.sampling_rate:.2f}s")
        print(f"  │  mean:      {tr.data.mean():.6f}")
        print(f"  │  std:       {tr.data.std():.6f}")
        print(f"  │  min:       {tr.data.min():.6f}")
        print(f"  │  max:       {tr.data.max():.6f}")
        print(f"  │  前 10 点:  {list(tr.data[:10].round(2))}")
        print(f"  └{'─'*70}")
        print()

    # 如果正在查看全部 trace，显示台站-通道汇总
    if args.index < 0:
        print("=" * 72)
        print("  台站 × 通道 矩阵")
        print("=" * 72)
        stations = sorted(set(tr.stats.station for tr in st))
        channels = sorted(set(tr.stats.channel for tr in st))
        header = f"  {'台站':<8} " + " ".join(f"{ch:>6}" for ch in channels)
        print(header)
        print("  " + "-" * len(header))
        for sta in stations:
            row = f"  {sta:<8} "
            for ch in channels:
                has = any(tr.stats.station == sta and tr.stats.channel == ch for tr in st)
                row += f"  {'X' if has else '·':>4} "
            print(row)


if __name__ == "__main__":
    main()
