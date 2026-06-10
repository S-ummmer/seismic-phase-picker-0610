#!/usr/bin/env python
# scripts/test_inference.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\test_inference.py
"""
完整推理流程测试：read → grouping 3-channel → resample → preprocess → model → peak detection
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.reader import WaveformReader, Waveform
from src.data.resampler import Resampler
from src.data.preprocessor import Preprocessor
from src.models.wrapper import ModelWrapper
from src.postprocess.peak_detector import PeakDetector


def group_station_3ch(waveforms, prefer_channels=("Z", "1", "2", "H")):
    """按台站合并通道为三分量 Waveform。

    返回满足 3 通道条件的台站。
    """
    # 按台站分组
    groups = {}
    for wf in waveforms:
        sta = wf.station
        if sta not in groups:
            groups[sta] = []
        groups[sta].append(wf)

    result = []
    for sta, wfs in groups.items():
        if len(wfs) < 3:
            continue  # 不足 3 通道，跳过

        # 找出 Z / 1 / 2 或等效通道
        z_wf = e_wf = n_wf = None
        for wf in wfs:
            ch = wf.channel.strip()
            parts = ch.split()[-1]  # 取最后一部分 (如 "Z", "1", "H")
            if parts == "Z":
                z_wf = wf
            elif parts == "1":
                e_wf = wf
            elif parts == "2":
                n_wf = wf

        if z_wf is None or e_wf is None or n_wf is None:
            continue

        # 对齐长度 (取最小)
        min_len = min(z_wf.n_samples, e_wf.n_samples, n_wf.n_samples)
        stacked = np.stack([
            e_wf.data[0, :min_len],
            n_wf.data[0, :min_len],
            z_wf.data[0, :min_len],
        ], axis=0)  # (3, N)

        result.append(Waveform(
            data=stacked.astype(np.float32),
            sampling_rate=z_wf.sampling_rate,
            starttime=z_wf.starttime,
            station=sta,
            channel="ENZ",
        ))

    return result


def main():
    # 1. 加载模型
    wrapper = ModelWrapper(
        model_path="models/phasenet.jit",
        info_path="models/model_info.json",
    )
    print(f"[模型] PhaseNet loaded, input: {wrapper.expected_channels}ch x {wrapper.expected_length}samples")

    # 2. 读取数据并按台站合并三分量
    raw_dir = Path("data/raw/2021/01")
    files = sorted(raw_dir.glob("*"))
    if not files:
        print("No data found!")
        return
    test_file = str(files[0])
    print(f"\n[数据] Reading: {Path(test_file).name}")

    reader = WaveformReader()
    waveforms = reader.read(test_file)
    print(f"       {len(waveforms)} raw traces")

    stations = group_station_3ch(waveforms)
    print(f"       {len(stations)} stations with 3 channels (ENZ)")

    if not stations:
        print("No 3-channel stations found! Available channels:")
        for wf in waveforms[:20]:
            print(f"  station={wf.station}, channel={wf.channel}")
        return

    # 3. 重采样 + 预处理
    resampler = Resampler(target_sr=100.0)
    preprocessor = Preprocessor(demean=True, normalize=True)
    detector = PeakDetector(min_distance=50, prominence=0.3, threshold=0.5)

    total_picks = 0
    phase_labels = ["P", "S"]

    for i, wf in enumerate(stations[:5]):
        wf = resampler.resample(wf)
        wf = preprocessor.process(wf)

        # 推理
        probs = wrapper.predict_prob(wf.data)  # (3, N)
        picks = detector.detect(
            probabilities=probs[1:],  # (2, N) — P and S
            phase_labels=phase_labels,
            time_fn=wf.time_at_index,
        )

        if picks:
            total_picks += len(picks)
            print(f"\n  [{i}] station={wf.station}: {len(picks)} picks")
            for p in picks[:5]:
                print(f"        {p.phase} @ t={p.time:.2f}s, prob={p.probability:.4f}")
        else:
            print(f"  [{i}] station={wf.station}: no picks above threshold")

    print(f"\nFull pipeline test complete! ({total_picks} total picks)")


if __name__ == "__main__":
    main()
