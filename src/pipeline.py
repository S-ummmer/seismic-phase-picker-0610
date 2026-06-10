# src/pipeline.py — 主流程编排
# D:\Github\Mk-project\seismic-phase-picker\src\pipeline.py

import yaml
import logging
from pathlib import Path
from typing import List, Optional

from .data.reader import Waveform
from .data.resampler import Resampler
from .data.preprocessor import Preprocessor
from .data.label_reader import LabelReader, EventLabels
from .models.wrapper import ModelWrapper
from .inference.sliding_window import SlidingWindowInference
from .postprocess.peak_detector import PeakDetector, PickedPhase

logger = logging.getLogger(__name__)


class SeismicPipeline:
    """地震震相拾取全流程编排器。"""

    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            YAML 配置文件路径。
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._init_components()

    def _init_components(self):
        cfg = self.config

        # 模型
        model_cfg = cfg["model"]
        self.model = ModelWrapper(
            model_path=model_cfg["jit_path"],
            device=model_cfg.get("device", "cpu"),
            info_path=model_cfg.get("info_path"),
        )

        # 推理
        inf_cfg = cfg["inference"]
        self.inference = SlidingWindowInference(
            model=self.model,
            window_length=inf_cfg["window_length"],
            step_size=inf_cfg["step_size"],
            batch_size=inf_cfg.get("batch_size", 32),
            threshold=inf_cfg.get("threshold", 0.5),
        )

        # 预处理
        self.resampler = Resampler(target_sr=self.model.expected_sampling_rate)
        self.preprocessor = Preprocessor(
            demean=True,
            normalize=True,
            sampling_rate=self.model.expected_sampling_rate,
        )

        # 后处理
        pp_cfg = cfg.get("postprocess", {})
        self.peak_detector = PeakDetector(
            min_distance=pp_cfg.get("min_distance", 50),
            prominence=pp_cfg.get("prominence", 0.3),
            threshold=pp_cfg.get("threshold", 0.5),
        )

        # 标签读取
        data_cfg = cfg.get("data", {})
        self.label_reader = LabelReader(format=data_cfg.get("format", "csv"))

        # 评估 (惰性初始化)
        ev_cfg = cfg.get("evaluation", {})
        self.tolerance = ev_cfg.get("tolerance", 0.5)
        self.target_phases = ev_cfg.get("phases", ["P", "S"])

    def run_inference(self, waveform: Waveform) -> List[PickedPhase]:
        """完整推理流程: 预处理 -> 滑动窗口推理 -> 峰值拾取。

        Parameters
        ----------
        waveform : Waveform
            原始波形。

        Returns
        -------
        List[PickedPhase]
            拾取到的震相列表。
        """
        # 预处理
        wf = self.resampler.resample(waveform)
        wf = self.preprocessor.process(wf)

        # 推理
        logger.info(f"Running inference on waveform: {wf}")
        probs = self.inference.run(wf)

        # 峰值拾取
        picks = self.peak_detector.detect(
            probabilities=probs,
            phase_labels=self.model.phase_labels,
            time_fn=wf.time_at_index,
        )

        logger.info(f"Detected {len(picks)} phases.")
        return picks
