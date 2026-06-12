# src/models/wrapper.py
# D:\Github\Mk-project\seismic-phase-picker\src\models\wrapper.py

import json
import torch
import numpy as np
from typing import Optional, List
from pathlib import Path


class ModelWrapper:
    """模型加载与推理封装。

    支持两种格式：
    - .jit: TorchScript 序列化模型（原始 phasenet.jit）
    - .pt: PyTorch 训练 checkpoint（含 model_state_dict）
    """

    def __init__(self, model_path: str, device: str = "cpu", info_path: Optional[str] = None):
        """
        Parameters
        ----------
        model_path : str
            模型文件路径，支持 .jit（TorchScript）和 .pt（训练 checkpoint）。
        device : str
            "cpu" / "cuda" / "mps"。
        info_path : str, optional
            模型信息 JSON 文件路径。
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        suffix = self.model_path.suffix.lower()

        if suffix == ".jit":
            self._load_jit(model_path)
        elif suffix == ".pt" or suffix == ".pth":
            self._load_checkpoint(model_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}, expected .jit or .pt")

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

        # model_info 中的 labels 优先于默认值，但模型自身的 labels 最高优先级
        if not hasattr(self, "phase_labels") or not self.phase_labels:
            self.phase_labels = self.info.get("phase_labels", ["Noise", "P", "S"])

    def _load_jit(self, model_path: str):
        """加载 TorchScript .jit 模型。"""
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # 尝试从模型中读取 labels 属性
        if hasattr(self.model, "labels"):
            self.phase_labels = list(self.model.labels)

    def _load_checkpoint(self, model_path: str):
        """加载训练 checkpoint (.pt)，包含 model_state_dict。"""
        from seisbench.models import PhaseNet

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # checkpoint 可能包含额外的训练状态
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # 兼容纯 state_dict
            state_dict = checkpoint

        # 推断 phases：从 state_dict 的 weights 标签数反推
        # 默认 "NPS" (Noise/P/S)，也兼容 "PS"
        weight_shape = state_dict.get("conv.weight", state_dict.get("conv1.weight"))
        if weight_shape is not None and weight_shape.shape[0] > 3:
            n_phases = weight_shape.shape[0]
        else:
            n_phases = 3

        phase_map = {3: "NPS", 2: "PS", 1: "N"}
        phases = phase_map.get(n_phases, "NPS")

        self.model = PhaseNet(phases=phases).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.phase_labels = list(phases)

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
