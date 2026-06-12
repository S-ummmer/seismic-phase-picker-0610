# src/training/augmentation.py
# D:\Github\Mk-project\seismic-phase-picker\src\training\augmentation.py
"""PhaseNet 数据增强模块（On-the-fly）。

实现 Mousavi et al. 2020 (Domain Invariant Hierarchical Learning)
中描述的四项增强：
  1. random_time_shift    ±10s，同步平移波形和 P/S 标签
  2. random_gaussian_noise  SNR 15-30dB
  3. random_channel_dropout  随机丢弃/交换通道
  4. random_sign_flip      随机符号反转

所有增强均为**可调用对象**，接受 (waveform, p_sample, s_sample)
并返回增强后的 (waveform, p_sample, s_sample)，
可直接在 `SteadDataset.__getitem__()` 中调用。
"""

import random
from typing import Optional, Tuple

import numpy as np


class RandomTimeShift:
    """随机时间平移。

    将波形循环平移 ±max_shift 秒，同时平移 P/S 标签位置。
    若平移后 P 或 S 掉出窗口，则拒绝该样本（返回原样）。

    Parameters
    ----------
    max_shift : float
        最大平移量（秒），默认 10s。
    sampling_rate : float
        采样率 Hz，默认 100Hz。
    """

    def __init__(self, max_shift: float = 10.0, sampling_rate: float = 100.0):
        self.max_shift = max_shift
        self.sr = sampling_rate

    def __call__(
        self,
        waveform: np.ndarray,
        p_sample: Optional[int],
        s_sample: Optional[int],
    ) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
        shift_sec = random.uniform(-self.max_shift, self.max_shift)
        shift_samples = int(round(shift_sec * self.sr))
        if shift_samples == 0:
            return waveform, p_sample, s_sample

        n_samples = waveform.shape[1]
        shifted = np.zeros_like(waveform)

        if shift_samples > 0:
            # 向右移：左边的样本补零
            shifted[:, shift_samples:] = waveform[:, :n_samples - shift_samples]
        else:
            # 向左移：右边的样本补零
            shifted[:, :n_samples + shift_samples] = waveform[:, -shift_samples:]

        # 平移标签
        new_p = (p_sample + shift_samples) if p_sample is not None else None
        new_s = (s_sample + shift_samples) if s_sample is not None else None

        # 拒绝：标签掉出窗口
        if (new_p is not None and (new_p < 0 or new_p >= n_samples) or
                new_s is not None and (new_s < 0 or new_s >= n_samples)):
            return waveform, p_sample, s_sample

        return shifted, new_p, new_s


class RandomGaussianNoise:
    """随机高斯噪声。

    按目标 SNR（信噪比）添加高斯白噪声。
    SNR = 10*log10(P_signal / P_noise)。

    Parameters
    ----------
    snr_range : tuple(float, float)
        目标 SNR 范围 (min_db, max_db)，默认 (15, 30)。
    """

    def __init__(self, snr_range: Tuple[float, float] = (15.0, 30.0)):
        self.snr_min, self.snr_max = snr_range

    def __call__(
        self,
        waveform: np.ndarray,
        p_sample: Optional[int],
        s_sample: Optional[int],
    ) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
        snr_db = random.uniform(self.snr_min, self.snr_max)
        # 按通道独立计算信号功率
        signal_power = np.mean(waveform ** 2, axis=1, keepdims=True)
        # SNR = 10*log10(P_signal / P_noise) → P_noise = P_signal / 10^(SNR/10)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.randn(*waveform.shape) * np.sqrt(noise_power)
        return waveform + noise.astype(waveform.dtype), p_sample, s_sample


class RandomChannelDropout:
    """随机通道丢弃或交换。

    两项操作二选一（50% 概率）：
    - Dropout：随机将一个通道置零（模拟传感器故障）
    - Swap：随机交换两个通道（模拟接线错误）

    Parameters
    ----------
    dropout_prob : float
        执行 dropout 的概率（其余情况执行 swap 或保持不变）。
    """

    def __init__(self, dropout_prob: float = 0.5):
        self.dropout_prob = dropout_prob

    def __call__(
        self,
        waveform: np.ndarray,
        p_sample: Optional[int],
        s_sample: Optional[int],
    ) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
        n_channels = waveform.shape[0]
        if n_channels < 2:
            return waveform, p_sample, s_sample

        augmented = waveform.copy()
        r = random.random()

        if r < self.dropout_prob:
            # 随机丢弃一个通道
            ch = random.randint(0, n_channels - 1)
            augmented[ch, :] = 0.0
        elif r < self.dropout_prob + 0.25:
            # 随机交换两个通道
            i, j = random.sample(range(n_channels), 2)
            augmented[[i, j], :] = augmented[[j, i], :]

        # 其余情况：保持不变（模拟无增强）
        return augmented, p_sample, s_sample


class RandomSignFlip:
    """随机符号反转。

    以 50% 概率将整个波形乘以 -1（极性反转）。
    标签位置不变（仅振幅符号变化）。
    """

    def __call__(
        self,
        waveform: np.ndarray,
        p_sample: Optional[int],
        s_sample: Optional[int],
    ) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
        if random.random() < 0.5:
            return -waveform, p_sample, s_sample
        return waveform, p_sample, s_sample


class Compose:
    """组合多个增强变换。

    用法:
        aug = Compose([
            RandomTimeShift(max_shift=10.0, sampling_rate=100.0),
            RandomGaussianNoise(snr_range=(15, 30)),
            RandomChannelDropout(),
            RandomSignFlip(),
        ])
        waveform, p_sample, s_sample = aug(waveform, p_sample, s_sample)
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        waveform: np.ndarray,
        p_sample: Optional[int],
        s_sample: Optional[int],
    ) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
        for t in self.transforms:
            waveform, p_sample, s_sample = t(waveform, p_sample, s_sample)
        return waveform, p_sample, s_sample

    def __repr__(self):
        names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({names})"
