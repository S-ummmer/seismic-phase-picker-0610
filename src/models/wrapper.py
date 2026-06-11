# src/models/wrapper.py
# D:\Github\Mk-project\seismic-phase-picker\src\models\wrapper.py

import json
import torch
import numpy as np
from typing import Optional, List
from pathlib import Path


class ModelWrapper:
    """模型加载与推理封装。

    加载 TorchScript 模型，提供统一的 predict() 接口。
    """

    def __init__(self, model_path: str, device: str = "cpu", info_path: Optional[str] = None):
        """
        Parameters
        ----------
        model_path : str
            TorchScript .jit 模型文件路径。
        device : str
            "cpu" / "cuda" / "mps"。
        info_path : str, optional
            模型信息 JSON 文件路径。
        """
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # 读取模型元信息
        if info_path is None:
            info_path = str(Path(model_path).parent / "model_info.json")
        if Path(info_path).exists():
            with open(info_path) as f:
                self.info = json.load(f)
        else:
            self.info = {}

        self.expected_sampling_rate = self.info.get("sampling_rate", 100.0)
        self.expected_length = self.info.get("input_shape", [1, 3, 3001])[-1]
        self.expected_channels = self.info.get("input_channels", 3)
        self.phase_labels = self.info.get("phase_labels", ["Noise", "P", "S"])

        # 尝试从模型中读取 labels 属性
        if hasattr(self.model, "labels"):
            self.phase_labels = list(self.model.labels)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """对输入波形进行推理。

        Parameters
        ----------
        data : np.ndarray
            波形数据，shape (n_channels, n_samples)，采样率需为 self.expected_sampling_rate。

        Returns
        -------
        np.ndarray
            各震相的概率序列，shape (n_classes, n_samples)。
        """
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # 裁剪/填充到期望长度
        n_samples = data.shape[-1]
        if n_samples > self.expected_length:
            data = data[:, :self.expected_length]
        elif n_samples < self.expected_length:
            pad = self.expected_length - n_samples
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")

        tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # (1, C, N)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)  # (1, classes, N)

        return output.squeeze(0).cpu().numpy()  # (classes, N)

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """批量推理 (B, C, N) → (B, classes, N)。

        用于滑动窗口等需要一次处理多个窗口的场景。
        """
        tensor = torch.from_numpy(batch.astype(np.float32)).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)  # (B, classes, N)
        return output.cpu().numpy()

    def predict_prob(self, data: np.ndarray) -> np.ndarray:
        """返回 softmax 归一化后的概率（0~1），单条 (C, N)。"""
        probs = self.predict(data)
        exp = np.exp(probs - probs.max(axis=0, keepdims=True))
        return exp / exp.sum(axis=0, keepdims=True)

    def predict_prob_batch(self, batch: np.ndarray) -> np.ndarray:
        """批量 softmax 概率 (B, C, N) → (B, classes, N)。"""
        probs = self.predict_batch(batch)
        exp = np.exp(probs - probs.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
