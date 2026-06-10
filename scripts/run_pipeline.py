# scripts/run_pipeline.py
# D:\Github\Mk-project\seismic-phase-picker\scripts\run_pipeline.py

"""
运行单个事件或批量事件的震相拾取流水线。

用法:
    python scripts/run_pipeline.py --config config.yaml --data data/raw/event_001.h5
    python scripts/run_pipeline.py --config config.yaml --data data/raw/ --batch
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import SeismicPipeline
from src.data.reader import Waveform

import numpy as np
import h5py


def load_waveform_hdf5(path: str) -> Waveform:
    """从 HDF5 文件加载波形。"""
    with h5py.File(path, "r") as f:
        data = f["data"][:]
        sr = f["data"].attrs.get("sampling_rate", 100.0)
        start_time = f["data"].attrs.get("start_time", 0.0)
    return Waveform(
        data=data,
        sampling_rate=sr,
        start_time=start_time,
    )


def main():
    parser = argparse.ArgumentParser(description="Seismic Phase Picker Pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Path to waveform file or directory")
    parser.add_argument("--batch", action="store_true", help="Process all files in directory")
    args = parser.parse_args()

    pipeline = SeismicPipeline(args.config)

    data_path = Path(args.data)

    if args.batch and data_path.is_dir():
        h5_files = sorted(data_path.glob("*.h5"))
        if not h5_files:
            h5_files = sorted(data_path.glob("*.hdf5"))
        print(f"Found {len(h5_files)} waveform files")
        for h5f in h5_files:
            print(f"\nProcessing: {h5f.name}")
            wf = load_waveform_hdf5(str(h5f))
            picks = pipeline.run_inference(wf)
            for p in picks:
                print(f"  {p.phase} at {p.time:.3f}s (prob={p.probability:.3f})")
    else:
        wf = load_waveform_hdf5(str(data_path))
        picks = pipeline.run_inference(wf)
        print(f"\nDetected {len(picks)} phases:")
        for p in picks:
            print(f"  {p.phase} at {p.time:.3f}s (prob={p.probability:.3f})")


if __name__ == "__main__":
    main()
