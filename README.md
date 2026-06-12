# seismic-phase-picker

地震震相拾取项目 — 基于 PhaseNet 的端到端震相检测、训练与评估流程。

---

## 项目结构

```
seismic-phase-picker/
│
├── README.md              ← 本文件
├── requirements.txt       ← Python 依赖
├── config.yaml            ← 全流程配置文件（预处理/模型/推理/训练/评估参数）
│
├── data/                  ← 数据目录
│   ├── raw/               ← 原始波形（只读）
│   │   ├── stead/         ←   STEAD 数据集: waveforms.hdf5 + metadata.csv
│   │   └── mseed/         ←   miniSEED 连续波形
│   ├── labels/            ← 震相标签（CSV/PHASE 格式）
│   └── processed/         ← 预处理缓存（.npy/.h5 中间产物）
│
├── models/                ← 预训练模型仓库（热插拔）
│   ├── phasenet.jit       ←   PhaseNet TorchScript 模型
│   └── model_info.json    ←   模型元信息（采样率/输入形状/输出通道）
│
├── src/                   ← 核心源码包
│   ├── pipeline.py        ←   主流程编排器（路径 A/B）
│   ├── io/                ←   格式相关 IO 层
│   ├── signal/            ←   格式无关信号处理
│   ├── models/            ←   模型封装（wrapper）
│   ├── inference/         ←   推理引擎（滑动窗口）
│   ├── postprocess/       ←   后处理（峰值拾取）
│   ├── training/          ←   训练管线
│   └── evaluation/        ←   评估系统
│
├── scripts/               ← 可执行脚本
│   ├── train_phasenet.py  ←   训练入口
│   ├── evaluate_stead.py  ←   STEAD 评估（含 A/B 对比）
│   ├── evaluate_folder.py ←   文件夹批量评估
│   ├── run_pipeline.py    ←   批量推理
│   ├── api_server.py      ←   HTTP API 服务
│   ├── inspect_stead.py   ←   STEAD 数据探查
│   └── inspect_mseed.py   ←   MSEED 数据探查
│
├── tests/                 ← 单元测试
└── outputs/               ← 输出结果
    ├── checkpoints/       ←   训练检查点
    ├── predictions/       ←   预测结果 CSV
    ├── evaluation/        ←   评估报告 CSV
    └── logs/              ←   运行日志
```

---

## 一、数据层 — `data/`

### 1.1 数据放置规范

| 目录 | 用途 | 存放内容 |
|------|------|----------|
| `data/raw/stead/` | STEAD 数据集 | `waveforms.hdf5`（波形）+ `metadata.csv`（标签） |
| `data/raw/mseed/` | 连续波形 | miniSEED 文件（`.mseed`），三分量 ENZ 地震记录 |
| `data/labels/` | 震相标签 | CSV/PHASE 文件，含文件名、震相类型（P/S）、到时时间 |
| `data/processed/` | 预处理缓存 | 预处理后的 `.npy` / `.h5` 中间文件 |

### 1.2 数据格式与读取

**STEAD (HDF5+CSV)：** 预截取时间窗口，每条 trace 约 60s 三分量波形，CSV 中记录 P/S 到时。

> 读取模块：`src/io/hdf5_reader.py` → `Hdf5Reader`

输出 `Waveform` 对象 + `Hdf5TraceInfo`（含 P/S 采样点索引）：
```python
from src.io.hdf5_reader import Hdf5Reader

with Hdf5Reader("data/raw/stead/waveforms.hdf5",
                "data/raw/stead/metadata.csv") as reader:
    for wf, info in reader.read_split("test"):
        print(wf.data.shape)       # (3, 6000)
        print(info.p_sample)       # P 波采样点索引
```

**MSEED (miniSEED)：** 连续三分量地震记录，需经事件检测截取。

> 读取模块：`src/io/mseed_reader.py` → `MseedReader`

```python
from src.io.mseed_reader import MseedReader

reader = MseedReader()
traces = reader.read("data/raw/mseed/2021/01/file.mseed")
stations = reader.group_station_3ch(traces)   # → [Waveform(ENZ), ...]
```

---

## 二、信号处理层 — `src/signal/`

**设计原则：** 格式无关的信号处理，输入 `Waveform` → 输出 `Waveform`。

### 2.1 预处理 — `preprocessor.py`

> 文件：`src/signal/preprocessor.py`

**流水线顺序（固定）：**

```
demean → detrend → taper → bandpass → normalize → trim
```

| 步骤 | 方法 | 参数 | 说明 |
|------|------|------|------|
| 去均值 | `data - data.mean()` | `demean: true` | 每通道独立减均值 |
| 去趋势 | `scipy.signal.detrend` | `detrend: true` | 移除线性趋势 |
| 尖灭 | Tukey window | `taper_alpha: 0.05` | 抑制滤波边界伪影 |
| 带通滤波 | 4 阶 Butterworth | `bandpass: [1, 20]` (Hz) | 可选，默认关闭 |
| Z-score 归一化 | `data / std` | `normalize: true` | 每通道独立标准化 |
| 长度裁剪 | 截断/补零 | `trim_length: N` | 可选，适配模型输入长度 |

### 2.2 重采样 — `resampler.py`

> 文件：`src/signal/resampler.py`

将波形重采样到目标频率（默认 100 Hz），含抗混叠滤波。

### 2.3 去噪 — `denoiser.py`（DeepDenoiser）

> 文件：`src/signal/denoiser.py`

基于 seisbench DeepDenoiser（Zhu et al., 2019），**在预处理前执行**。

```
原始波形 (C, N)
    │ 切分为 3000-sample 滑动窗口
    ▼
窗口 1 → STFT → U-Net → mask → ISTFT → 去噪窗口 1
窗口 2 → STFT → U-Net → mask → ISTFT → 去噪窗口 2
    ...
    │ 重叠区域余弦加权平均
    ▼
去噪波形 (C, N)
```

**配置（`config.yaml`）：**
```yaml
denoiser:
  enabled: false          # 设为 true 启用
  pretrained: "original"  # "original" | "urban"
  device: "cpu"
  overlap: 0.5
```

**首次使用需下载 ~30MB 权重到 `~/.seisbench/models/`。**

### 2.4 事件检测 — `event_detector.py`（仅路径 A）

> 文件：`src/signal/event_detector.py`

STA/LTA 算法从连续波形中检测地震事件窗口：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| STA | 1.0 s | 短时窗 |
| LTA | 30.0 s | 长时窗 |
| threshold | 3.0 | STA/LTA 触发阈值 |
| pre_window | 5.0 s | 触发前保留 |
| post_window | 30.0 s | 触发后保留 |

---

## 三、训练管线 — `src/training/`

### 3.1 数据集 — `dataset.py`

> 文件：`src/training/dataset.py` → `SteadDataset`

PyTorch Dataset，加载 STEAD HDF5 波形并生成三分类掩码。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 6001 | PhaseNet 输入长度（60s @100Hz） |
| `p_offset_sec` | 10.0 | P 到时在窗口内 10s 处（震前 10s） |
| `s_offset_sec` | 50.0 | S 到时目标在窗口 50s 处（震后最大 50s） |
| `p_buffer` | 20 | P 标记半窗口（样本） |
| `s_buffer` | 20 | S 标记半窗口（样本） |
| `require_both_ps` | false | 仅保留同时有 P+S 标签的 trace |

**智能窗口定位策略：**
1. P 固定在窗口 `p_offset_samples` 处
2. 若 S 超出窗口 → 滑动使 S 接近 `s_offset_samples`
3. 必须保证 P 不掉出窗口
4. S-P > 窗口容量时，仅保留 P

每条样本返回：`(waveform_3x6001, mask_6001, meta_dict)`

**标签方案 — `label_generator.py`：**

> 文件：`src/training/label_generator.py` → `generate_phase_mask()`

生成三分类逐点掩码：
- **0** = Noise（噪声）
- **1** = P 波（p_sample ± p_buffer）
- **2** = S 波（s_sample ± s_buffer）

### 3.2 数据增强 — `augmentation.py`

> 文件：`src/training/augmentation.py`

四类 On-the-fly 增强，基于 Mousavi et al. 2020（Domain Invariant Hierarchical Learning）：

| 增强 | 类 | 参数 | 说明 |
|------|-----|------|------|
| 随机时间平移 | `RandomTimeShift` | `max_shift=10s` | 循环平移波形，同步平移标签；越界则拒绝 |
| 随机高斯噪声 | `RandomGaussianNoise` | `snr_range=[15, 30]` dB | 按通道功率独立添加噪声 |
| 随机通道丢弃/交换 | `RandomChannelDropout` | `drop_prob=0.5` | 模拟传感器故障或接线错误 |
| 随机符号反转 | `RandomSignFlip` | `prob=0.5` | 极性反转 |

**使用方式：**
```python
from src.training.augmentation import Compose, RandomTimeShift, RandomGaussianNoise

aug = Compose([
    RandomTimeShift(max_shift=10.0, sampling_rate=100.0),
    RandomGaussianNoise(snr_range=(15, 30)),
    RandomChannelDropout(),
    RandomSignFlip(),
])
waveform, p_sample, s_sample = aug(waveform, p_sample, s_sample)
```

### 3.3 训练器 — `trainer.py`

> 文件：`src/training/trainer.py` → `PhaseNetTrainer`

**训练策略：**

| 组件 | 配置 | 说明 |
|------|------|------|
| 损失函数 | `CrossEntropyLoss` | 带类别权重（自动从数据集估算） |
| 优化器 | `Adam` | `lr=1e-3`, `weight_decay=1e-5` |
| 学习率调度 | `CosineAnnealingLR` | `T_max = epochs` |
| EarlyStopping | `patience=10` | 连续 10 epoch val_loss 无改善则停止 |
| 模型保存 | `outputs/checkpoints/` | `best_model.pt` + `last_model.pt` |
| 训练/验证 | STEAD train/dev split | |

**训练记录：**
```python
trainer.history = {
    "train_loss": [...], "val_loss": [...],
    "train_acc":  [...], "val_acc":  [...],
    "lr":         [...],
}
```

### 3.4 训练入口 — `scripts/train_phasenet.py`

```bash
# 小样本试跑
python scripts/train_phasenet.py --epochs 2 --max-train 200 --max-val 50

# 完整训练（60s窗口 + EarlyStopping）
python scripts/train_phasenet.py --epochs 50 --batch-size 32 --lr 1e-3

# 从检查点恢复
python scripts/train_phasenet.py --resume outputs/checkpoints/last_model.pt --epochs 10

# 禁用 EarlyStopping
python scripts/train_phasenet.py --patience 0

# 调整窗口大小（原始 PhaseNet 30s）
python scripts/train_phasenet.py --window-size 3001
```

---

## 四、模型层 — `src/models/`

> 文件：`src/models/wrapper.py` → `ModelWrapper`

### 4.1 模型加载与推理

**支持的模型格式：**

| 格式 | 后缀 | 加载方式 | 说明 |
|------|------|----------|------|
| TorchScript | `.jit` | `torch.jit.load()` | 预编译的 PhaseNet，跨平台运行 |
| PyTorch Checkpoint | `.pt` / `.pth` | 重建 PhaseNet + `load_state_dict()` | 训练产出，含 model_state_dict |

**推理接口：**
```python
model = ModelWrapper("models/phasenet.jit")

probs = model.predict(data)        # (C, N) → (3, N) softmax 前
probs = model.predict_prob(data)   # (C, N) → (3, N) softmax 概率

batch_out = model.predict_prob_batch(batch)  # (B, C, N) → (B, 3, N)
```

**模型元信息（`models/model_info.json`）：**
```json
{
  "sampling_rate": 100.0,
  "input_shape": [1, 3, 3001],
  "input_channels": 3,
  "phase_labels": ["Noise", "P", "S"]
}
```

---

## 五、推理引擎 — `src/inference/`

> 文件：`src/inference/sliding_window.py` → `SlidingWindowInference`

### 5.1 滑动窗口推理

将长波形切分为固定窗口逐窗推理，重叠区域取平均。

```
|=======|                  窗口 1: 0 → 30s
    |=======|              窗口 2: 15s → 45s  (50% 重叠)
        |=======|          窗口 3: 30s → 60s
      ↓ 批量推理
      重叠区域平均
      ↓
  完整概率序列 (3, N)
```

**配置参数（`config.yaml`）：**
```yaml
inference:
  window_length: 30.0    # 秒
  step_size: 15.0        # 秒 (50% overlap)
  batch_size: 32
  threshold: 0.5         # 震相拾取最低概率阈值
```

---

## 六、后处理 — `src/postprocess/`

> 文件：`src/postprocess/peak_detector.py` → `PeakDetector`

### 6.1 峰值拾取

将概率序列 `(2, N)` → 离散震相列表：

```
阈值筛选 → 峰值检测 → 合并重复震相（极小窗口内只保留最高概率）
```

**参数（`config.yaml`）：**
```yaml
postprocess:
  method: "peak_detection"
  min_distance: 50       # 信号峰值最小间距（样本）
  prominence: 0.3        # 峰值显著性
```

**输出格式：**
```python
PickedPhase(phase="P", time=123.45, probability=0.87, ...)
```

---

## 七、评估系统 — `src/evaluation/`

### 7.1 震相匹配 — `matcher.py`

> 文件：`src/evaluation/matcher.py` → `PhaseMatcher`

按震相类型在容忍窗口内贪心匹配预测 ↔ 真实标签。

| 参数 | 默认 | 说明 |
|------|------|------|
| `tolerance` | 0.5 s | 预测值与此范围内标注值视为匹配 |

**输出：** `MatchSummary`（TP / FP / FN 三列表）

### 7.2 基础指标 — `metrics.py`

> 文件：`src/evaluation/metrics.py` → `MetricsCalculator`

从 `MatchSummary` 计算：

| 指标 | 公式 | 说明 |
|------|------|------|
| Precision | `TP / (TP + FP)` | 预测正确的比例 |
| Recall | `TP / (TP + FN)` | 真实震相被检出的比例 |
| F1 | `2·P·R / (P + R)` | 综合指标 |
| Mean/Median/Std Time Error | 仅 TP 的误差统计 | 拾取精度 |

**同时计算 per-phase（P/S 分别统计）。**

### 7.3 震相质量分级 — `grading.py`

> 文件：`src/evaluation/grading.py` → `EventGrader`

仅基于 TP 的时间误差中位数（median_error）分级：

| 等级 | 阈值 | 含义 |
|------|------|------|
| A | ≤ 0.05 s | 极佳 |
| B | ≤ 0.10 s | 良好 |
| C | ≤ 0.20 s | 一般 |
| D | > 0.20 s | 较差 |

### 7.4 比赛评分 — `scorer.py`

> 文件：`src/evaluation/scorer.py` → `PhaseScorer`

**时间误差打分（每个 TP 0~1 分）：**

| 震相 | ≤ 满分阈值 | 满分→零分区间 | ≥ 零分阈值 | 说明 |
|------|-----------|-------------|-----------|------|
| P | ≤ 0.1s → 1 分 | 0.1~1s → 线性衰减 | ≥ 1s → 0 分 | P 波要求更高精度 |
| S | ≤ 0.2s → 1 分 | 0.2~2s → 线性衰减 | ≥ 2s → 0 分 | S 波容差更大 |

**数量惩罚：** 预测数量与真实数量偏差在 5% 内不扣分；超出部分每个扣 0.5 分。

```
最终得分 = Σ TP 得分 - 数量惩罚
```

---

## 八、全流程编排 — `src/pipeline.py`

> 文件：`src/pipeline.py` → `SeismicPipeline`

### 路径 A：连续波形（MSEED）

```
MSEED 文件
    → MseedReader.read()
    → MseedReader.group_station_3ch()
    → EventDetector.detect()         ← STA/LTA 事件检测
    → EventDetector.extract_window()
    → Resampler.resample()           ← 重采样到 100Hz
    → DeepDenoiser.denoise()         ← [可选] 去噪
    → Preprocessor.process()         ← 预处理
    → ModelWrapper.predict_prob()    ← PhaseNet 推理
    → PeakDetector.detect()          ← 峰值拾取
    → 输出震相列表
```

### 路径 B：预截取窗口（HDF5+CSV / STEAD）

```
HDF5 + CSV
    → Hdf5Reader.read()
    → Resampler.resample()           ← 重采样到 100Hz
    → DeepDenoiser.denoise()         ← [可选] 去噪
    → Preprocessor.process()         ← 预处理
    → SlidingWindowInference.run()   ← 滑动窗口推理
    → PeakDetector.detect()          ← 峰值拾取
    → PhaseMatcher.match()           ← 震相匹配
    → MetricsCalculator.compute()    ← 基础指标
    → EventGrader.grade()            ← 质量分级
    → PhaseScorer.score()            ← 比赛评分
    → 输出震相列表 + 评估报告
```

### A/B 对比模式（DeepDenoiser 效果评估）

> 脚本：`scripts/evaluate_stead.py --compare`

同一条波形并行推理两条路径：
```
原始波形 ──→ Preprocessor → PhaseNet → Picks_A  ┐
   │                                             ├→ 对比报告
   └─→ DeepDenoiser → Preprocessor → PhaseNet → Picks_B
```

---

## 九、脚本说明

| 脚本 | 作用 | 关键参数 |
|------|------|----------|
| `scripts/train_phasenet.py` | PhaseNet 训练 | `--epochs`, `--batch-size`, `--window-size`, `--patience`, `--resume` |
| `scripts/evaluate_stead.py` | STEAD 评估 | `--split`, `--max`, `--threshold`, `--denoise`, `--compare`, `--tolerance` |
| `scripts/evaluate_folder.py` | 批量文件夹评估 | `--config`, `--data_dir` |
| `scripts/run_pipeline.py` | 批量推理 | `--data_dir`, `--max_files`, `--threshold` |
| `scripts/api_server.py` | HTTP API 服务 | 接收文件路径 → 返回 JSON 震相列表 |
| `scripts/inspect_stead.py` | STEAD 数据探查 | `--index` 查看特定 trace |
| `scripts/inspect_mseed.py` | MSEED 数据探查 | `--index` 查看特定文件 |

---

## 十、测试

| 测试文件 | 测试对象 |
|----------|----------|
| `tests/test_reader.py` | MSEED/HDF5 读取 |
| `tests/test_resampler.py` | 重采样精度 |
| `tests/test_preprocessor.py` | 预处理流水线 |
| `tests/test_denoiser.py` | DeepDenoiser 去噪 |
| `tests/test_matcher.py` | 震相匹配 |
| `tests/test_metrics.py` | 指标计算 |
| `tests/test_grading.py` | 质量分级 |
| `tests/test_training.py` | 训练管线 |

```bash
cd seismic-phase-picker
python -m pytest tests/ -v
```

---

## 十一、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据探查
python scripts/inspect_stead.py --index 1234
python scripts/inspect_mseed.py --index 1

# 3. 训练（小样本验证）
python scripts/train_phasenet.py --epochs 2 --max-train 200 --max-val 50

# 4. 训练（完整）
python scripts/train_phasenet.py --epochs 50 --batch-size 32 --lr 1e-3

# 5. 评估
python scripts/evaluate_stead.py --split test --max 500 --threshold 0.2

# 6. 评估 + 去噪
python scripts/evaluate_stead.py --denoise --split test --max 500

# 7. A/B 对比（去噪 vs 不去噪）
python scripts/evaluate_stead.py --compare --split test --max 500

# 8. 推理
python scripts/run_pipeline.py --data_dir data/raw/stead/
python scripts/run_pipeline.py --data_dir data/raw/mseed/
```

---

## 十二、配置文件说明

全流程配置集中在 `config.yaml`，8 个配置段：

| 配置段 | 说明 | 关键参数 |
|--------|------|----------|
| `data` | 数据路径与格式 | `raw_path`, `labels_path`, `format: auto` |
| `preprocessing` | 预处理开关 | `demean`, `detrend`, `taper`, `bandpass`, `normalize`, `sampling_rate` |
| `denoiser` | DeepDenoiser 去噪 | `enabled`, `pretrained`, `device`, `overlap` |
| `model` | 模型选择 | `type: phasenet`, `jit_path`, `device` |
| `inference` | 推理参数 | `window_length`, `step_size`, `batch_size`, `threshold` |
| `event_detection` | STA/LTA 参数 | `sta`, `lta`, `threshold` |
| `postprocess` | 峰值拾取 | `min_distance`, `prominence` |
| `evaluation` | 评估参数 | `tolerance`, `phases: [P, S]` |
| `training` | 训练参数 | `epochs`, `batch_size`, `lr`, `window_size`, `patience` |
