# seismic-phase-picker

地震震相拾取项目，基于 PhaseNet / EQTransformer 的端到端震相检测与评估流程。

---

## 项目结构

```
seismic-phase-picker/
│
├── README.md              ← 本文件
├── requirements.txt       ← Python 依赖
├── config.yaml            ← 全流程配置文件
│
├── data/                  ← 数据层
├── models/                ← 模型仓库
├── src/                   ← 核心源码包
├── scripts/               ← 可执行脚本
├── notebooks/             ← 探索分析
├── tests/                 ← 单元测试
└── outputs/               ← 输出结果
```

---

## 一、data/ — 数据层

**职责：** 管理所有数据的生命周期，从原始波形到标签答案。

| 子目录 | 作用 | 存放内容 |
|--------|------|----------|
| `raw/` | 原始波形数据（只读） | MSEED 文件（.mseed），三分量地震波形记录 |
| `labels/` | 真实震相答案 | CSV/PHASE 文件，包含文件名、震相类型（P/S）、到时时间 |
| `processed/` | 预处理缓存（可选） | 预处理后的 .npy 或 .h5 中间文件，加速重复实验 |

**数据流：** `raw/` → 经 `src/io/` 模块读取 → `src/signal/` 处理 → 模型输入；`labels/` → 经 `src/io/label_reader.py` 解析 → 评估模块

---

## 二、models/ — 模型仓库

**职责：** 存放预训练模型文件及其自描述元数据，实现模型热插拔。

| 文件 | 作用 | 详细说明 |
|------|------|----------|
| `phasenet.jit` | PhaseNet 模型权重（TorchScript） | 跨平台序列化模型，可在无 Python 源码环境运行。可替换为 `eqt.jit` |
| `model_info.json` | 模型元信息 | 记录输入形状、采样率、预处理是否内置、输出通道含义。`wrapper.py` 据此决定外部预处理策略 |

**关键设计：** 模型文件与代码完全解耦。切换模型只需更换这两个文件 + 修改 `config.yaml`，无需改动任何业务代码。

---

## 三、src/ — 核心源码包

### 3.1 `src/__init__.py`

将 `src` 标记为 Python 包，可定义包级别版本号。

---

### 3.2 `src/io/` — 格式相关 IO 层

**职责：** 隔离文件格式差异，输出统一的 `Waveform` 对象。

| 文件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `mseed_reader.py` | 读取 miniSEED 波形 + 三分量合并 | MSEED 文件路径 | `Waveform` 列表（单通道或 ENZ 三分量） |
| `hdf5_reader.py` | 读取 HDF5+CSV 格式 (STEAD/谛听) | HDF5 文件 + CSV 元数据 | `Waveform` + `Hdf5TraceInfo` (含标签) |
| `label_reader.py` | 读取震相标签（CSV/STEAD/谛听） | CSV/PHASE 文件 | `{trace_name: [PhaseLabel, ...]}` |

**关键设计：** IO 层只负责"读到什么"，不做信号处理。不同格式的读取策略各不相同，但一旦输出 `Waveform`，后续所有代码完全通用。

**Waveform 对象**（定义在 `src/io/__init__.py`）：
- `data`: `(N_channels, N_samples)` ndarray
- `sampling_rate`, `starttime`, `station`, `channel`
- `time_at_index(idx)`: 采样点 → 绝对时间

---

### 3.3 `src/signal/` — 格式无关信号处理层

**职责：** 对 `Waveform` 做纯信号处理，不关心数据来源。

| 文件 | 职责 | 详细说明 |
|------|------|----------|
| `resampler.py` | 重采样到目标频率 | 独立可测；含抗混叠滤波；适配不同台站原始采样率 |
| `preprocessor.py` | 信号预处理流水线 | demean → detrend → taper → bandpass → normalize → trim；顺序固定，每步可独立开关 |
| `event_detector.py` | STA/LTA 事件检测 | 从连续波形中检测地震事件窗口；仅路径 A 使用 |

**处理链：**
```
Waveform → resampler.py → preprocessor.py → 模型输入
```

> **注意：** 对于预截取窗口（路径 B，如 STEAD），跳过 `event_detector.py`，直接 preprocess。

---

### 3.4 `src/models/` — 模型封装模块

| 文件 | 职责 | 详细说明 |
|------|------|----------|
| `wrapper.py` | 模型加载与推理接口 | 加载 JIT/PyTorch 模型；读取 `model_info.json` 校验输入形状；封装 `predict(waveform) -> probs` 统一接口 |

**关键功能：**
- 根据 `model_info.json` 决定是否跳过外部预处理
- 统一不同模型的输入/输出格式
- 支持 PhaseNet、EQTransformer 等多模型热插拔

---

### 3.5 `src/inference/` — 推理引擎模块

| 文件 | 职责 | 详细说明 |
|------|------|----------|
| `sliding_window.py` | 滑动窗口推理 | 将长波形切分为固定窗口（如 60s）；逐窗推理；处理窗口重叠区域去重拼接 |

**为什么需要：** 大多数模型只接受固定长度输入，实际波形往往长达数分钟甚至小时。滑动窗口可处理任意长度，重叠设计避免边界震相漏检。

> **注意：** `time_aligner.py` 已合并进 `reader.py`，时间对齐是数据读取时完成的元信息提取。

---

### 3.6 `src/postprocess/` — 后处理模块

| 文件 | 职责 | 详细说明 |
|------|------|----------|
| `peak_detector.py` | 概率序列 → 离散震相列表 | 阈值筛选 → 峰值检测 → 合并重复震相（极小窗口内只保留最高概率）→ 输出 `[(phase_type, time_s, confidence)]` |

**关键参数（均可在 `config.yaml` 中配置）：**
- 概率阈值（默认 0.5）
- 合并窗口（默认 0.2s）

---

### 3.6 `src/evaluation/` — 评估模块

| 文件 | 职责 | 详细说明 |
|------|------|----------|
| `matcher.py` | 震相事件匹配 | 基于容忍窗口的匈牙利/贪心匹配；P/S 分别匹配；返回 TP/FP/FN |
| `metrics.py` | 基础指标计算 | Precision, Recall, F1, MAE, RMSE |
| `grading.py` | 分级评估 | 按到时误差将 TP 分为"完好/破坏/毁坏"；计算各级别 IoU（Jaccard 指数） |

**评估流程：**
```
预测震相 + 真实震相
    ↓
matcher.py（容忍窗口匹配）
    ↓
metrics.py（基础指标）
    ↓
grading.py（分级 IoU）
```

> **IoU 计算公式：** `IoU = TP / (TP + FP + FN)`，与 F1 单调正相关

---

### 3.7 `src/pipeline.py` — 双路径主流程编排

**职责：** 将上述所有模块串联。自动识别数据格式分派到两条路径。

**路径 A（连续波形 — MSEED）：**
```
MSEED 文件
    → MseedReader.read()
    → MseedReader.group_station_3ch()
    → EventDetector.detect()        ← STA/LTA 事件检测
    → EventDetector.extract_window()
    → Resampler.resample()
    → Preprocessor.process()
    → ModelWrapper.predict_prob()
    → PeakDetector.detect()
    → 输出震相列表
```

**路径 B（预截取窗口 — HDF5+CSV）：**
```
HDF5 + CSV
    → Hdf5Reader.read()
    → Resampler.resample()
    → Preprocessor.process()
    → ModelWrapper.predict_prob()
    → PeakDetector.detect()
    → 输出震相列表 + 评估分数
```

**核心原则：** 格式相关的放 `src/io/`，格式无关的放 `src/signal/`。

---

## 四、scripts/ — 可执行脚本

| 文件 | 作用 | 详细说明 |
|------|------|----------|
| `run_pipeline.py` | 批量推理 | 自动识别 MSEED/HDF5 格式，运行完整流程，输出 CSV 预测震相 |
| `evaluate_folder.py` | 批量评估 | 对整个测试集运行，汇总所有文件的评估指标 |
| `api_server.py` | API 服务 | 比赛要求的 HTTP API 接口：接收文件路径 → 返回 JSON 震相列表 |
| `inspect_mseed.py` | 数据审视 | 查看 miniSEED 文件结构和头段信息 |
| `inspect_stead.py` | 数据审视 | 查看 STEAD HDF5 波形和 CSV 标签详情 |
| `test_read_data.py` | 单元测试 | 验证 MSEED 数据读取+预处理链路 |
| `test_inference.py` | 单元测试 | 验证 MSEED 完整推理链路（读取→推理→拾取） |

---

## 五、notebooks/ — 探索分析

| 文件 | 作用 |
|------|------|
| `exploratory_analysis.ipynb` | 数据探索、可视化调试、原型验证 |

> 与 `src/` 的区别：notebook 是"实验台"，`src/` 是"生产线"。

---

## 六、tests/ — 单元测试

| 文件 | 测试对象 | 详细说明 |
|------|----------|----------|
| `fixtures/` | 测试数据 | 人工合成的小型波形数据和假答案，确保测试快速、独立、可复现 |
| `test_reader.py` | `mseed_reader.py` | 测试 MSEED 读取、头段解析、三分量合并 |
| `test_resampler.py` | `resampler.py` | 测试重采样精度、抗混叠效果 |
| `test_preprocessor.py` | `preprocessor.py` | 测试 demean/detrend/taper/filter/normalize |
| `test_event_detector.py` | `event_detector.py` | 测试 STA/LTA 事件检测正确性 |
| `test_matcher.py` | `matcher.py` | 测试容忍窗口匹配正确性 |
| `test_metrics.py` | `metrics.py` | 测试 Precision/Recall/F1 计算 |
| `test_grading.py` | `grading.py` | 测试完好/破坏/毁坏分级逻辑 |

---

## 七、outputs/ — 输出结果

| 子目录 | 作用 | 存放内容 |
|--------|------|----------|
| `predictions/` | 预测结果 | 每个文件的预测震相列表 CSV（`{basename}_picks.csv`），列：phase_type, time_s, confidence |
| `logs/` | 运行日志 | 每次运行的详细日志文件 |

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试（验证数据读取+预处理）
python scripts/test_read_data.py

# 数据结构预览
# （HDF5+CSV）
python scripts/inspect_stead.py --index 1234
# MSEED
python scripts/inspect_mseed.py --index 1

# MSEED 路径（自动识别连续波形）
python scripts/run_pipeline.py --data_dir data/raw/2021/

# HDF5+CSV 路径（自动识别预截取窗口）
python scripts/run_pipeline.py --data_dir data/raw/stead/

# 限制数量 + 调阈值
python scripts/run_pipeline.py --data_dir data/raw/stead/ --max_files 100 --threshold 0.3

python scripts/evaluate_stead.py --split test --threshold 0.2 --prominence 0.1 --min-distance 25

# 批量评估
python scripts/evaluate_folder.py --config config.yaml --data_dir data/raw/2021/01/
```

---

## 配置文件说明

参见 `config.yaml`，主要配置项：

- `data.sampling_rate`：目标采样率（默认 100 Hz）
- `inference.threshold`：震相检测概率阈值
- `evaluation.tolerance`：匹配容忍窗口（秒）
- `model.type`：模型类型（phasenet / eqtransformer）
