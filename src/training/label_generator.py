# src/training/label_generator.py
# D:\Github\Mk-project\seismic-phase-picker\src\training\label_generator.py
"""STEAD P/S 到时 → PhaseNet 三分类掩码 (0=Noise, 1=P, 2=S)。

策略说明：
- P 波：p_sample ± p_buffer 范围标记为 1
- S 波：s_sample ± s_buffer 范围标记为 2
- P/S 窗口重叠时，S 波优先（更晚到达，特征更分散）
- S 波不存在时（无标注），仅标记 P
- 其余全部标记为 0（噪声）
"""

import numpy as np
from typing import Optional


def generate_phase_mask(
    n_samples: int,
    p_sample: Optional[int],
    s_sample: Optional[int],
    p_buffer: int = 20,
    s_buffer: int = 20,
) -> np.ndarray:
    """将 P/S 到时转为 PhaseNet 三分类掩码。

    Parameters
    ----------
    n_samples : int
        波形总样本数。
    p_sample : int or None
        P 波到达样本索引（None = 无 P 标签）。
    s_sample : int or None
        S 波到达样本索引（None = 无 S 标签）。
    p_buffer : int
        P 波标记半窗口大小（样本数），默认 20（200ms @100Hz）。
    s_buffer : int
        S 波标记半窗口大小（样本数），默认 20（200ms @100Hz）。

    Returns
    -------
    mask : ndarray of int64, shape (n_samples,)
        0 = 噪声, 1 = P 波, 2 = S 波
    """
    mask = np.zeros(n_samples, dtype=np.int64)

    # ── S 波先标记（重叠时 S 优先） ──
    if s_sample is not None and 0 <= s_sample < n_samples:
        s_start = max(0, int(s_sample - s_buffer))
        s_end = min(n_samples, int(s_sample + s_buffer))
        mask[s_start:s_end] = 2

    # ── P 波标记（不与 S 重叠的区域） ──
    if p_sample is not None and 0 <= p_sample < n_samples:
        p_start = max(0, int(p_sample - p_buffer))
        p_end = min(n_samples, int(p_sample + p_buffer))
        # 只在 mask==0 的位置写 P（S 波已占用则跳过）
        mask[p_start:p_end] = np.where(mask[p_start:p_end] == 0, 1, mask[p_start:p_end])

    return mask


def mask_to_onehot(mask: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """掩码 → one-hot (C, N) 浮点数组，供加权 loss 使用。

    Parameters
    ----------
    mask : (N,) int64
    n_classes : int

    Returns
    -------
    (n_classes, N) float32
    """
    onehot = np.zeros((n_classes, len(mask)), dtype=np.float32)
    for c in range(n_classes):
        onehot[c, mask == c] = 1.0
    return onehot


def mask_class_distribution(mask: np.ndarray) -> dict:
    """统计三类分布比例。

    Returns
    -------
    {"noise": 0.85, "P": 0.10, "S": 0.05}
    """
    total = len(mask)
    return {
        "noise": (mask == 0).sum() / total,
        "P": (mask == 1).sum() / total,
        "S": (mask == 2).sum() / total,
    }
