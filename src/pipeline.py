# src/pipeline.py — 主流程编排
# D:\Github\Mk-project\seismic-phase-picker\src\pipeline.py
"""
双路径编排器：
- 路径 A (连续波形): MSEED → EventDetector → Preprocessor → Model → Picks
- 路径 B (预截取窗口): HDF5+CSV → Preprocessor → Model → Picks → Evaluation
"""

import yaml
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from .io import Waveform
from .io.mseed_reader import MseedReader
from .io.hdf5_reader import Hdf5Reader, Hdf5TraceInfo
from .io.label_reader import LabelReader, EventLabels, PhaseLabel
from .signal.preprocessor import Preprocessor
from .signal.resampler import Resampler
from .signal.event_detector import EventDetector, EventWindow
from .models.wrapper import ModelWrapper
from .inference.sliding_window import SlidingWindowInference
from .postprocess.peak_detector import PeakDetector, PickedPhase

logger = logging.getLogger(__name__)


class SeismicPipeline:
    """地震震相拾取全流程编排器。

    使用方法:
        pipe = SeismicPipeline("config.yaml")

        # 路径 A：连续波形 (MSEED)
        picks = pipe.process_continuous("data/raw/2021/01/file.mseed")

        # 路径 B：预截取窗口 (HDF5)
        results = pipe.process_windows("data/raw/stead/")
    """

    def __init__(self, config_path: str):
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
            demean=True, detrend=True, taper=True,
            normalize=True, bandpass=None,
            sampling_rate=self.model.expected_sampling_rate,
        )

        # 后处理
        pp_cfg = cfg.get("postprocess", {})
        self.peak_detector = PeakDetector(
            min_distance=pp_cfg.get("min_distance", 50),
            prominence=pp_cfg.get("prominence", 0.3),
            threshold=pp_cfg.get("threshold", 0.5),
        )

        # 事件检测（路径 A）
        self.event_detector = EventDetector(
            sta=1.0, lta=30.0, threshold=3.0,
        )

        # 标签读取
        data_cfg = cfg.get("data", {})
        self.label_reader = LabelReader(format=data_cfg.get("format", "csv"))

        # 评估参数
        ev_cfg = cfg.get("evaluation", {})
        self.tolerance = ev_cfg.get("tolerance", 0.5)
        self.target_phases = ev_cfg.get("phases", ["P", "S"])

    # ── 路径 A: 连续波形 ──────────────────────────────

    def process_continuous(
        self, filepath: str
    ) -> List[Tuple[str, List[PickedPhase]]]:
        """处理连续 MSEED 波形：事件检测 → 截取 → 震相拾取。

        Parameters
        ----------
        filepath : str
            miniSEED 文件路径。

        Returns
        -------
        List[Tuple[str, List[PickedPhase]]]
            [(station_id, [picks, ...]), ...]
        """
        reader = MseedReader()
        traces = reader.read(filepath)
        stations = reader.group_station_3ch(traces)

        all_results = []
        for wf in stations:
            # 事件检测
            windows = self.event_detector.detect(wf)
            if not windows:
                continue

            for win in windows:
                event_wf = self.event_detector.extract_window(wf, win)
                picks = self._run_inference(event_wf)
                all_results.append((wf.station, picks))

        return all_results

    def process_continuous_no_detect(
        self, filepath: str
    ) -> List[Tuple[str, List[PickedPhase]]]:
        """处理连续 MSEED 但不做事件检测（短波形直推）。"""
        reader = MseedReader()
        traces = reader.read(filepath)
        stations = reader.group_station_3ch(traces)

        results = []
        for wf in stations:
            picks = self._run_inference(wf)
            results.append((wf.station, picks))
        return results

    # ── 路径 B: 预截取窗口 ──────────────────────────────

    def process_windows(
        self, hdf5_path: str, csv_path: str,
        split: str = "test", max_traces: int = 0,
    ) -> List[dict]:
        """处理预截取窗口 (HDF5+CSV)：推理 + 评估。

        Returns
        -------
        List[dict]
            每条 trace 的结果: {trace_name, picks, labels, metrics}
        """
        results = []
        with Hdf5Reader(hdf5_path, csv_path) as reader:
            traces = reader.read_split(split, max_traces)
            for wf, info in traces:
                picks = self._run_inference(wf)
                row = {
                    "trace_name": info.trace_name,
                    "station": wf.station,
                    "picks": picks,
                    "labels": self._build_labels(info),
                }
                results.append(row)
        return results

    # ── 公用推理 ───────────────────────────────────────

    def _run_inference(self, waveform: Waveform) -> List[PickedPhase]:
        """完整推理链: 重采样 → 预处理 → 模型 → 峰值拾取。"""
        wf = self.resampler.resample(waveform)
        wf = self.preprocessor.process(wf)
        probs = self.model.predict_prob(wf.data)
        picks = self.peak_detector.detect(
            probabilities=probs[1:],  # (2, N) — skip Noise
            phase_labels=["P", "S"],
            time_fn=wf.time_at_index,
        )
        return picks

    def _build_labels(self, info: Hdf5TraceInfo) -> List[PhaseLabel]:
        labels = []
        if info.p_sample is not None:
            labels.append(PhaseLabel(
                time=info.p_sample / info.sampling_rate,
                phase="P",
            ))
        if info.s_sample is not None:
            labels.append(PhaseLabel(
                time=info.s_sample / info.sampling_rate,
                phase="S",
            ))
        return labels

    # ── 快捷方法 ───────────────────────────────────────

    def run_inference_mseed(
        self, filepath: str
    ) -> List[Tuple[str, List[PickedPhase]]]:
        """MSEED 快捷推理（不做事件检测，适合已截取的短波形）。"""
        reader = MseedReader()
        traces = reader.read(filepath)
        return [(wf.station, self._run_inference(wf)) for wf in traces]
