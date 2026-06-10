# Seismic Phase Picker

基于深度学习的自动地震震相拾取系统，支持 PhaseNet / EQTransformer 等模型。

## 项目结构

```
seismic-phase-picker/
├── data/              # 数据目录 (raw / labels / processed)
├── models/            # 模型文件 (JIT / 权重)
├── src/               # 核心源码
│   ├── data/          # 数据读取与预处理
│   ├── models/        # 模型封装
│   ├── inference/     # 推理引擎 (滑动窗口)
│   ├── postprocess/   # 拾取后处理 (峰检测)
│   ├── evaluation/    # 评估 (匹配/指标/分级)
│   └── pipeline.py    # 主流程编排
├── scripts/           # 启动脚本
├── notebooks/         # 探索性分析
├── tests/             # 单元测试
└── outputs/           # 预测结果与日志
```

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config config.yaml
```

## 数据格式

默认支持 STEAD、谛听等标准地震数据集格式。
原始波形放置于 `data/raw/`，标注文件放置于 `data/labels/`。
