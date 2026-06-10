#!/usr/bin/env python
# scripts/test_read_data.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\test_read_data.py
"""
测试数据读取和预处理流程。
读取 data/raw/2021/01/ 下一个事件文件，验证 reader + resampler + preprocessor 工作正常。
"""
import sys
from pathlib import Path

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.reader import WaveformReader, Waveform
from src.data.resampler import Resampler
from src.data.preprocessor import Preprocessor


def main():
    # 找测试文件
    raw_dir = Path("D:/Github/Mk-project/seismic-phase-picker/data/raw/2021/01")
    files = sorted(raw_dir.glob("*"))
    if not files:
        print("未找到测试数据！")
        return
    test_file = str(files[0])
    print(f"测试文件: {test_file}")

    # 1. 读取
    reader = WaveformReader()
    waveforms = reader.read(test_file)
    print(f"\n[1] 读取完成: {len(waveforms)} 条 waveform")
    for wf in waveforms[:3]:
        print(f"  {wf}")

    # 2. 重采样到 100 Hz (PhaseNet 要求)
    target_sr = 100.0
    resampler = Resampler(target_sr=target_sr)
    resampled = [resampler.resample(wf) for wf in waveforms]
    print(f"\n[2] 重采样到 {target_sr} Hz 完成")
    for wf in resampled[:3]:
        print(f"  {wf}")

    # 3. 预处理
    preprocessor = Preprocessor(demean=True, normalize=True)
    processed = [preprocessor.process(wf) for wf in resampled]
    print(f"\n[3] 预处理完成")
    for wf in processed[:3]:
        d = wf.data
        print(f"  {wf.station}/{wf.channel}: "
              f"mean={d.mean():.6f}, std={d.std():.6f}, "
              f"min={d.min():.6f}, max={d.max():.6f}")

    print("\n✅ 数据读取+预处理流程跑通！")
    return waveforms, resampled, processed


if __name__ == "__main__":
    main()
