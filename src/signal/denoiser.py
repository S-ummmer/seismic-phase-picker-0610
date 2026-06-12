# src/signal/denoiser.py
# D:\Github\Mk-project\seismic-phase-picker\src\signal\denoiser.py
"""
DeepDenoiser 地震信号去噪封装。

基于 seisbench DeepDenoiser（Zhu et al., 2019），
在预处理前对三分量波形进行深度学习去噪。

架构：
    原始波形 (C, N)
        │
        ▼ 切分为 3000-sample 滑动窗口
    窗口 1 → STFT → U-Net → mask → ISTFT → 去噪窗口 1
    窗口 2 → STFT → U-Net → mask → ISTFT → 去噪窗口 2
        ...
        │
        ▼ 重叠区域余弦加权平均
    去噪波形 (C, N)

DeepDenoiser 原生输入为 3000 样本 @100Hz（30s）。
长波形通过滑动窗口 + 重叠重建处理。

Reference:
    Zhu, W., Mousavi, S. M., & Beroza, G. C. (2019).
    Seismic signal denoising and decomposition using deep neural networks.
    IEEE TGRS, 57(11), 9476-9488.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class DeepDenoiser:
    """DeepDenoiser 去噪器封装。

    用法:
        denoiser = DeepDenoiser()
        denoised = denoiser.denoise(data)          # data: (3, N) ndarray
        denoised = denoiser.denoise(data, overlap=0.5)
    """

    # 原生输入参数
    NATIVE_SAMPLES = 3000   # DeepDenoiser 固定输入长度
    NATIVE_SR = 100          # 采样率 Hz

    def __init__(self, pretrained: str = "original", device: str = "cpu"):
        """
        Parameters
        ----------
        pretrained : str
            seisbench 预训练版本: "original" | "urban"
        device : str
            "cpu" | "cuda"
        """
        import seisbench.models as sbm

        self.pretrained = pretrained
        self.device = device
        self._model: Optional[sbm.DeepDenoiser] = None
        self._loaded = False

    # ── 懒加载 ──────────────────────────────────────────

    @property
    def model(self):
        """懒加载 seisbench DeepDenoiser 模型（按需下载权重）。"""
        if self._model is None:
            import seisbench.models as sbm
            logger.info(
                f"Loading DeepDenoiser '{self.pretrained}' "
                f"(first call may download weights ~30MB)..."
            )
            try:
                self._model = sbm.DeepDenoiser.from_pretrained(self.pretrained)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Failed to download DeepDenoiser weights. "
                    f"seisbench stores weights at ~/.seisbench/models/.\n"
                    f"Ensure network access and write permissions, "
                    f"then retry. Original error: {e}"
                ) from e
            if self.device != "cpu":
                self._model.to(self.device)
            self._model.eval()
            self._loaded = True
            logger.info("DeepDenoiser loaded.")
        return self._model

    # ── 公有 API ────────────────────────────────────────

    def denoise(
        self,
        data: np.ndarray,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """去噪三分量波形。

        Parameters
        ----------
        data : (C, N) ndarray, float32/float64
            三分量波形数据，采样率必须为 100Hz。
        overlap : float
            滑动窗口重叠比例，默认 0.5。

        Returns
        -------
        (C, N) ndarray
            去噪后的波形，shape 与输入一致。
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array (C, N), got shape {data.shape}")
        C, N = data.shape

        # 短波形：单次去噪
        if N <= self.NATIVE_SAMPLES:
            return self._denoise_single(data)

        # 长波形：滑动窗口
        return self._denoise_sliding(data, overlap=overlap)

    # ── 单窗口去噪 ───────────────────────────────────────

    def _denoise_single(self, data: np.ndarray) -> np.ndarray:
        """单窗口波形去噪（内部处理 ≤3000 样本）。"""
        from obspy import Trace, Stream

        C, N = data.shape

        # numpy → ObsPy Stream
        stream = Stream()
        for i in range(C):
            tr = Trace(data=data[i].copy())
            tr.stats.sampling_rate = self.NATIVE_SR
            tr.stats.channel = f"HH{i}"
            stream.append(tr)

        # DeepDenoiser annotate（原地修改 stream）
        self.model.annotate(stream)

        # ObsPy Stream → numpy
        denoised = np.stack([tr.data for tr in stream], axis=0)
        return denoised.astype(np.float32)

    # ── 滑动窗口去噪 ─────────────────────────────────────

    def _denoise_sliding(
        self,
        data: np.ndarray,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """滑动窗口去噪 + 余弦加权重叠重建。"""
        C, N = data.shape
        window_size = self.NATIVE_SAMPLES
        step = int(window_size * (1.0 - overlap))
        step = max(1, step)

        # 生成窗口起始位置
        starts = list(range(0, max(1, N - window_size + 1), step))
        if not starts or starts[-1] + window_size < N:
            starts.append(max(0, N - window_size))

        # 累积器
        accum = np.zeros((C, N), dtype=np.float64)
        weight_accum = np.zeros(N, dtype=np.float64)

        # 余弦窗口（重叠区域平滑过渡）
        cos_win = self._cosine_window(window_size)

        for s in starts:
            end = min(s + window_size, N)
            wlen = end - s

            # 截取
            win = data[:, s:end]

            # 如果窗口不足，右侧补零
            if wlen < window_size:
                padded = np.zeros((C, window_size), dtype=np.float32)
                padded[:, :wlen] = win
                denoised_full = self._denoise_single(padded)
                denoised = denoised_full[:, :wlen]
                win_weight = cos_win[:wlen]
            else:
                denoised = self._denoise_single(win)
                win_weight = cos_win

            accum[:, s:end] += denoised * win_weight
            weight_accum[s:end] += win_weight

        # 归一化
        weight_accum[weight_accum == 0] = 1.0
        result = accum / weight_accum
        return result.astype(np.float32)

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _cosine_window(length: int) -> np.ndarray:
        """余弦渐变窗口（两端平滑过渡）。"""
        t = np.arange(length, dtype=np.float64)
        win = np.ones(length, dtype=np.float64)
        taper_len = min(length // 4, 150)  # taper 长度最多 150 样本
        if taper_len > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_len) / taper_len))
            win[:taper_len] = ramp
            win[-taper_len:] = ramp[::-1]
        return win

    # ── 批量去噪（用于评估管线） ──────────────────────────

    def denoise_waveforms(
        self,
        waveforms: np.ndarray,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """批量去噪（每个 waveform 独立处理）。

        Parameters
        ----------
        waveforms : (B, C, N) ndarray
        overlap : float

        Returns
        -------
        (B, C, N) ndarray
        """
        results = []
        for i in range(len(waveforms)):
            denoised = self.denoise(waveforms[i], overlap=overlap)
            results.append(denoised)
        return np.stack(results, axis=0)


# ── 模块级便捷函数 ───────────────────────────────────────

def create_denoiser(
    enabled: bool = False,
    pretrained: str = "original",
    device: str = "cpu",
) -> Optional[DeepDenoiser]:
    """工厂函数：根据配置创建 DeepDenoiser（或返回 None）。

    Parameters
    ----------
    enabled : bool
        True 时加载模型，False 返回 None。
    pretrained : str
        预训练版本名。
    device : str

    Returns
    -------
    DeepDenoiser or None
    """
    if not enabled:
        return None
    return DeepDenoiser(pretrained=pretrained, device=device)
