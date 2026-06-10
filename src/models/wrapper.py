# src/models/wrapper.py
# D:\Github\Mk-project\seismic-phase-picker\src\models\wrapper.py

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional


class ModelWrapper:
    """模型封装器。

    加载 JIT 模型或标准 PyTorch checkpoint，提供统一的推理接口。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        info_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_path : str
            TorchScript (.jit) 或 .pt checkpoint 路径。
        device : str
            运行设备 ("cpu" / "cuda" / "mps")。
        info_path : str, optional
            模型元信息 JSON 路径。
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.info_path = Path(info_path) if info_path else None
        self.model_info = self._load_info()
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        suffix = self.model_path.suffix
        if suffix in (".jit", ".pt"):
            model = torch.jit.load(str(self.model_path), map_location=self.device)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
        model.eval()
        return model.to(self.device)

    def _load_info(self) -> dict:
        if self.info_path and self.info_path.exists():
            with open(self.info_path, "r") as f:
                return json.load(f)
        return {}

    @property
    def phase_labels(self) -> List[str]:
        """模型输出的震相类型标签，如 ["P", "S"]。"""
        return self.model_info.get("phase_labels", ["P", "S"])

    @property
    def expected_sampling_rate(self) -> float:
        return self.model_info.get("sampling_rate", 100.0)

    @property
    def expected_length(self) -> int:
        return self.model_info.get("input_length", 3000)

    def predict(self, waveform: torch.Tensor) -> torch.Tensor:
        """执行推理。

        Parameters
        ----------
        waveform : torch.Tensor
            shape (batch, channels, samples)。

        Returns
        -------
        torch.Tensor
            shape (batch, n_phases, samples) — 每个采样点的概率。
        """
        with torch.no_grad():
            waveform = waveform.to(self.device)
            output = self.model(waveform)
        return output.cpu()
