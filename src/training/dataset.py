# src/training/dataset.py
# D:\Github\Mk-project\seismic-phase-picker\src\training\dataset.py
"""STEAD PyTorch Dataset — 加载 HDF5 波形 + 生成三分类掩码。

用法:
    dataset = SteadDataset(hdf5_path, csv_path, split="train")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.hdf5_reader import Hdf5Reader, Hdf5TraceInfo
from src.io import Waveform
from src.signal.resampler import Resampler
from src.signal.preprocessor import Preprocessor
from src.training.label_generator import generate_phase_mask


class SteadDataset(Dataset):
    """STEAD 训练/验证/测试 Dataset。

    每条样本返回:
      - waveform: (3, window_size) float32 tensor
      - mask:     (window_size,) int64 tensor       (0=噪声, 1=P, 2=S)
      - info:     Hdf5TraceInfo (元信息，测试用)
    """

    def __init__(
        self,
        hdf5_path: str,
        csv_path: str,
        split: str = "train",
        window_size: int = 6001,
        target_sr: float = 100.0,
        p_buffer: int = 20,
        s_buffer: int = 20,
        p_offset_sec: float = 10.0,
        s_offset_sec: float = 50.0,
        demean: bool = True,
        detrend: bool = True,
        taper: bool = True,
        normalize: bool = True,
        require_both_ps: bool = False,
        max_traces: int = 0,
    ):
        """
        Parameters
        ----------
        hdf5_path : str
        csv_path : str
        split : str
            "train" / "dev" / "test"
        window_size : int
            PhaseNet 输入长度。
            6001 = 60s @100Hz（震前10s + 震后50s）
            3001 = 30s @100Hz（原始 PhaseNet 默认）
        target_sr : float
            目标采样率 Hz。
        p_buffer, s_buffer : int
            P/S 标记半窗口（样本数）。
        p_offset_sec : float
            窗口起始到 P 到时的目标秒数（默认 10s，
            对应 STEAD 格式"震前10秒"）。
        s_offset_sec : float
            窗口起始到 S 到时的目标秒数（默认 50s，
            对应"震后50秒"）。实际 S 位置由数据决定，
            此值仅用于滑动窗口定位。
        require_both_ps : bool
            True 时只保留同时有 P 和 S 标签的 trace。
        max_traces : int
            最大加载数，0 = 全部。
        """
        self.hdf5_path = hdf5_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.target_sr = target_sr
        self.p_buffer = p_buffer
        self.s_buffer = s_buffer
        self.p_offset_samples = int(p_offset_sec * target_sr)
        self.s_offset_samples = int(s_offset_sec * target_sr)
        self.split = split
        self.require_both_ps = require_both_ps

        # 信号处理组件
        self.resampler = Resampler(target_sr=target_sr)
        self.preprocessor = Preprocessor(
            demean=demean, detrend=detrend,
            taper=taper, normalize=normalize,
        )

        # 加载元数据
        self._reader = Hdf5Reader(hdf5_path, csv_path)
        self._df = self._reader.df

        # 按 split 过滤
        split_mask = self._df["split"] == split
        self._indices = self._df[split_mask].index.tolist()

        # 可选：只保留有 P+S 标签的
        if require_both_ps:
            self._indices = [
                i for i in self._indices
                if (pd_notna(self._df.iloc[i].get("trace_P_arrival_sample"))
                    and pd_notna(self._df.iloc[i].get("trace_S_arrival_sample")))
            ]

        if max_traces > 0:
            self._indices = self._indices[:max_traces]

        # 打开 HDF5（读索引时不反复开关）
        self._h5 = None

    # ── context manager ──────────────────────────────────

    def __enter__(self):
        self._reader.__enter__()
        self._h5 = self._reader._h5
        return self

    def __exit__(self, *args):
        self._reader.__exit__(*args)
        self._h5 = None

    # ── Dataset 接口 ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        csv_idx = self._indices[idx]
        row = self._df.iloc[csv_idx]

        # 加载波形
        data = self._load_waveform(row)  # (3, N)
        info = self._reader._parse_row(row)

        # 重采样 + 预处理
        wf = Waveform(
            data=data,
            sampling_rate=info.sampling_rate,
            starttime=info.start_time,
            station=info.station,
            channel="ENZ",
        )
        wf = self.resampler.resample(wf)
        wf = self.preprocessor.process(wf)

        # 重采样后 P/S 位置
        sr_ratio = self.target_sr / info.sampling_rate
        p_sample = int(info.p_sample * sr_ratio) if info.p_sample is not None else None
        s_sample = int(info.s_sample * sr_ratio) if info.s_sample is not None else None

        # ── 智能窗口定位 ──
        # 目标（STEAD 格式）：震前 10s + 震后 50s（共 60s）
        #   - P 到时在窗口内 10s 处（p_offset_samples）
        #   - S 到时尽量落在窗口内
        # 策略（优先级）：
        #   1) P 固定在窗口 p_offset_samples 处
        #   2) S 超出窗口 → 滑动窗口使 S 接近 s_offset_samples 处，
        #      但必须保证 P 不掉出窗口
        #   3) S-P > 窗口容量 → 无法同时捕获，保留 P
        data_len = wf.data.shape[1]
        window_size = self.window_size

        if p_sample is not None:
            # P 目标位置：窗口内 10s 处
            p_target = self.p_offset_samples
            win_start = max(0, p_sample - p_target)

            # S 超出窗口？尝试滑动使 S 接近 s_offset_samples 处
            if s_sample is not None and s_sample >= win_start + window_size:
                s_target = self.s_offset_samples
                alt_start = s_sample - s_target
                # 只有 P 不掉出窗口才接受
                if p_sample >= alt_start:
                    win_start = max(0, alt_start)

            # 不超出 trace 末尾
            win_start = min(win_start, max(0, data_len - window_size))
        elif s_sample is not None:
            # 只有 S：S 在窗口 s_offset_samples 处
            s_target = self.s_offset_samples
            win_start = max(0, s_sample - s_target)
            win_start = min(win_start, max(0, data_len - window_size))
        else:
            win_start = 0

        # 截取窗口
        win_end = min(win_start + window_size, data_len)
        waveform = self._trim_or_pad(wf.data[:, win_start:win_end], window_size)

        # 调整 P/S 的相对位置
        p_sample_rel = (p_sample - win_start) if p_sample is not None else None
        s_sample_rel = (s_sample - win_start) if s_sample is not None else None

        # 过滤越界的标签
        if p_sample_rel is not None and not (0 <= p_sample_rel < window_size):
            p_sample_rel = None
        if s_sample_rel is not None and not (0 <= s_sample_rel < window_size):
            s_sample_rel = None

        # 生成掩码
        mask = generate_phase_mask(
            n_samples=window_size,
            p_sample=p_sample_rel,
            s_sample=s_sample_rel,
            p_buffer=self.p_buffer,
            s_buffer=self.s_buffer,
        )

        meta = {
            "trace_name": info.trace_name,
            "station": info.station,
            "p_sample": p_sample_rel if p_sample_rel is not None else -1,
            "s_sample": s_sample_rel if s_sample_rel is not None else -1,
            "has_p": p_sample_rel is not None,
            "has_s": s_sample_rel is not None,
            "win_start": win_start,
        }

        return (
            torch.from_numpy(waveform.copy()).float(),
            torch.from_numpy(mask.copy()).long(),
            meta,
        )

    # ── 内部方法 ──────────────────────────────────────────

    def _load_waveform(self, row) -> np.ndarray:
        """从 HDF5 加载一条 trace 的 (3, N) 波形。"""
        return self._reader._load_trace(row)

    @staticmethod
    def _trim_or_pad(data: np.ndarray, target_len: int) -> np.ndarray:
        """截取或零填充到 target_len。"""
        C, N = data.shape
        if N >= target_len:
            return data[:, :target_len].astype(np.float32)
        else:
            padded = np.zeros((C, target_len), dtype=np.float32)
            padded[:, :N] = data
            return padded

    # ── 信息 ──────────────────────────────────────────────

    def class_weights(self) -> torch.Tensor:
        """估算三类样本权重 (noise, P, S)。

        遍历全部数据统计掩码分布，返回 1/频率 的归一化权重。
        """
        counts = np.array([0, 0, 0], dtype=np.float64)
        for idx in range(len(self)):
            _, mask, _ = self[idx]
            for c in range(3):
                counts[c] += (mask == c).sum().item()

        # 避免除零
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * 3.0  # 归一化到总和=3
        return torch.tensor(weights, dtype=torch.float32)


def pd_notna(val) -> bool:
    """跨版本 pd.notna 安全调用。"""
    try:
        import pandas as pd
        return pd.notna(val)
    except Exception:
        return val is not None and val == val
