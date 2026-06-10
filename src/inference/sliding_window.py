# src/inference/sliding_window.py
# D:\Github\Mk-project\seismic-phase-picker\src\inference\sliding_window.py

import numpy as np
import torch
from typing import List, Tuple
from src.data.reader import Waveform
from src.models.wrapper import ModelWrapper


class SlidingWindowInference:
    """滑动窗口推理引擎。

    将一个长波形切分为多个重叠窗口分别推理，再拼接为完整概率序列。
    不包含 time_aligner — 时间映射由 Waveform.time_at_index 负责。
    """

    def __init__(
        self,
        model: ModelWrapper,
        window_length: float = 30.0,      # seconds
        step_size: float = 15.0,          # seconds
        batch_size: int = 32,
        threshold: float = 0.5,
    ):
        self.model = model
        self.window_length = window_length
        self.step_size = step_size
        self.batch_size = batch_size
        self.threshold = threshold

    def run(self, waveform: Waveform) -> np.ndarray:
        """对波形执行滑动窗口推理。

        Parameters
        ----------
        waveform : Waveform
            输入波形。

        Returns
        -------
        np.ndarray
            shape (n_phases, n_samples) — 每个采样点每个震相的概率。
        """
        sr = waveform.sampling_rate
        window_samples = int(self.window_length * sr)
        step_samples = int(self.step_size * sr)
        n_phases = len(self.model.phase_labels)
        total_samples = waveform.n_samples

        # 累积概率矩阵
        prob_accum = np.zeros((n_phases, total_samples), dtype=np.float64)
        count_accum = np.zeros(total_samples, dtype=np.float64)

        # 生成窗口起始点
        starts = list(range(0, total_samples - window_samples + 1, step_samples))
        if starts[-1] + window_samples < total_samples:
            starts.append(total_samples - window_samples)

        # 批量处理
        for batch_start in range(0, len(starts), self.batch_size):
            batch_starts = starts[batch_start:batch_start + self.batch_size]
            batch = []
            for s in batch_starts:
                window = waveform.data[:, s:s + window_samples]
                batch.append(window)
            batch_tensor = torch.from_numpy(np.stack(batch)).float()  # (B, C, T)

            outputs = self.model.predict(batch_tensor).numpy()  # (B, P, T)

            for i, s in enumerate(batch_starts):
                window_out = outputs[i]  # (P, T)
                prob_accum[:, s:s + window_samples] += window_out
                count_accum[s:s + window_samples] += 1.0

        # 平均重叠部分
        count_accum[count_accum == 0] = 1.0
        prob_averaged = prob_accum / count_accum

        return prob_averaged.astype(np.float32)

    def detect_phases(self, waveform: Waveform) -> Tuple[np.ndarray, np.ndarray]:
        """推理并提取高于阈值的震相时刻。

        Parameters
        ----------
        waveform : Waveform
            输入波形。

        Returns
        -------
        times : np.ndarray
            震相绝对时间数组 (Unix timestamp)。
        phases : np.ndarray
            震相类型字符串数组。
        labels : np.ndarray
            对应 phase_labels 的整数索引。
        probs : np.ndarray
            对应概率值。
        """
        probs = self.run(waveform)
        phase_labels = self.model.phase_labels

        times_list, phases_list, labels_list, probs_list = [], [], [], []
        for p_idx in range(probs.shape[0]):
            phase_probs = probs[p_idx]
            above = np.where(phase_probs >= self.threshold)[0]
            for idx in above:
                times_list.append(waveform.time_at_index(idx))
                phases_list.append(phase_labels[p_idx])
                labels_list.append(p_idx)
                probs_list.append(phase_probs[idx])

        return (
            np.array(times_list),
            np.array(phases_list),
            np.array(labels_list),
            np.array(probs_list),
        )
