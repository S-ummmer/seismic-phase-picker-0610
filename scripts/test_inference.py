#!/usr/bin/env python
# scripts/test_inference.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\test_inference.py
"""
完整推理流程测试 — MSEED 路径：read → 3ch grouping → resample → preprocess → model → peak detection
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io.mseed_reader import MseedReader, Waveform
from src.signal.resampler import Resampler
from src.signal.preprocessor import Preprocessor
from src.models.wrapper import ModelWrapper
from src.postprocess.peak_detector import PeakDetector


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

    reader = MseedReader()
    waveforms = reader.read(test_file)
    print(f"       {len(waveforms)} raw traces")

    stations = reader.group_station_3ch(waveforms)
    print(f"       {len(stations)} stations with 3 channels (ENZ)")

    if not stations:
        print("No 3-channel stations found! Available channels:")
        for wf in waveforms[:20]:
            print(f"  station={wf.station}, channel={wf.channel}")
        return

    # 3. 重采样 + 预处理
    resampler = Resampler(target_sr=100.0)
    preprocessor = Preprocessor(demean=True, detrend=True, taper=True, normalize=True)
    detector = PeakDetector(min_distance=50, prominence=0.3, threshold=0.5)

    total_picks = 0

    for i, wf in enumerate(stations[:5]):
        wf = resampler.resample(wf)
        wf = preprocessor.process(wf)

        probs = wrapper.predict_prob(wf.data)  # (3, N)
        picks = detector.detect(
            probabilities=probs[1:],  # (2, N) — P and S
            phase_labels=["P", "S"],
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
