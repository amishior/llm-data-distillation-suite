# SFT Dataset Distillation Pipeline

一个基于多 LLM 自动生成、评估和筛选 SFT（Supervised Fine-Tuning）训练数据的数据蒸馏流水线。

## 功能简介

* 从领域知识切片（JSONL）自动生成指令-回复对
* 使用多个 LLM 对生成样本进行多维度质量评估
* 基于评分进行筛选与语义去重
* 输出符合 ShareGPT 格式的 `train.jsonl` / `val.jsonl` 以及元数据 `metadata.json`

适用于：

* 从领域知识库蒸馏高质量 SFT 训练数据
* 为领域大模型微调自动构造训练 / 评测集
* 搭建可扩展、可替换 LLM 的数据生产流水线

---

## 项目结构

```text
sft-distillation/
├─ sft_distill/
│  ├─ __init__.py
│  ├─ config.py          # 配置模版 & 默认参数
│  ├─ models.py          # Pydantic 数据模型
│  ├─ prompts.py         # 生成 / 评估 Prompt 模版
│  ├─ llm_client.py      # 通用 LLM API Client（异步）
│  ├─ utils.py           # JSON 抽取等通用工具
│  ├─ generator.py       # SFT 数据生成器
│  ├─ evaluator.py       # 质量评估器
│  ├─ filtering.py       # 筛选与去重逻辑
│  ├─ data_io.py         # JSONL 读写等 I/O
│  ├─ pipeline.py        # DatasetDistillationPipeline 主工作流
├─ scripts/
│  └─ run_pipeline.py    # 命令行入口
├─ requirements.txt
├─ README.md
```

---

## 环境与安装

### 环境要求

* Python 3.10+

### 安装依赖

```bash
pip install -r requirements.txt
```

你也可以使用可编辑安装的方式（可选）：

```bash
pip install -e .
```

或者直接使用：

```bash
PYTHONPATH=. python scripts/run_pipeline.py --input ./knowledge_slices.jsonl
```

---

## 配置说明

默认配置位于 `sft_distill/config.py` 中的 `CONFIG_TEMPLATE`：

```python
CONFIG_TEMPLATE = {
    "generator": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
        "model": "qwen3-max",
        "max_concurrent": 5,
    },
    "evaluators": {
        "primary": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "Moonshot-Kimi-K2-Instruct",
            "max_concurrent": 3,
        },
        "secondary": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "",
            "model": "deepseek-v3.2-exp",
            "max_concurrent": 3,
        },
    },
    "samples_per_slice": 3,
    "min_score_threshold": 7.0,
    "min_votes": 2,
    "train_val_split": 0.9,
    "enable_dedup": True,
    "output_dir": "./distilled_dataset",
}
```

你可以通过 `config.json` 覆盖这些配置（浅合并）：

```jsonc
{
  "generator": {
    "api_key": "YOUR_QWEN_API_KEY"
  },
  "evaluators": {
    "primary": {
      "api_key": "YOUR_KIMI_API_KEY"
    },
    "secondary": {
      "api_key": "YOUR_DEEPSEEK_API_KEY"
    }
  },
  "output_dir": "./distilled_dataset_insurance",
  "samples_per_slice": 5,
  "min_score_threshold": 7.5
}
```

运行时指定：

```bash
python scripts/run_pipeline.py \
  --input ./knowledge_slices.jsonl \
  --output ./distilled_dataset \
  --config ./config.json
```

---

## 输入数据格式

输入文件是 JSONL，每行一个知识切片：

```json
{"id": "slice_1", "content": "数千到数万字的领域知识文本...", "metadata": {"domain": "insurance"}}
```

字段说明：

* `id`：知识切片 ID，可选，不填则自动生成
* `content`：长文本内容（条款、文档片段等），必填
* `metadata`：可选元数据，例如来源、领域标签等

---

## 输出数据格式

运行结束后，`output_dir`（默认 `./distilled_dataset`）下会包含：

* `train.jsonl`：训练集（ShareGPT 格式）
* `val.jsonl`：验证集（ShareGPT 格式）
* `generated_raw.jsonl`：原始生成的 SFT 样本（未筛选）
* `metadata.json`：元信息（样本数量、平均得分等）

### ShareGPT 样本示例

```json
{
  "id": "slice_1_gen0_20250301010101",
  "conversations": [
    {
      "from": "human",
      "value": "指令文本\n\n可选 input 内容"
    },
    {
      "from": "gpt",
      "value": "高质量回答文本"
    }
  ],
  "avg_score": 8.2,
  "dimension_scores": {
    "Moonshot-Kimi-K2-Instruct": {
      "accuracy": 8.5,
      "completeness": 8.0,
      "logicality": 8.0,
      "helpfulness": 8.5,
      "safety": 9.0
    },
    "deepseek-v3.2-exp": {
      "accuracy": 8.0,
      "completeness": 8.0,
      "logicality": 8.5,
      "helpfulness": 8.0,
      "safety": 8.5
    }
  }
}
```

---

## 运行示例

1. 准备知识切片：

```bash
cat > knowledge_slices.jsonl << 'EOF'
{"id": "slice_1", "content": "......", "metadata": {}}
{"id": "slice_2", "content": "......", "metadata": {}}
EOF
```

2. 填写 API Key（在 `config.json` 或 `config.py` 中）

3. 运行流水线：

```bash
python scripts/run_pipeline.py \
  --input ./knowledge_slices.jsonl \
  --output ./distilled_dataset \
  --config ./config.json
```

---

## 高级用法：语义去重

`DataFilter` 支持传入 `embed_fn` 做句向量去重。例如：

```python
from typing import Iterable
import numpy as np

from sft_distill.filtering import DataFilter


def my_embed_fn(texts: Iterable[str]) -> np.ndarray:
    # 使用你自己的 embedding 模型
    return your_embedder.encode(list(texts))

config = {
    # ... 其他配置
    "embed_fn": my_embed_fn,
}
```

当 `embed_fn` 存在时，将基于 **余弦相似度** 做去重；否则使用简单的 **Jaccard 文本相似度**。

---

## TODO

* 支持断点续跑（只重跑评估或去重阶段）
* 支持输出更多格式（Alpaca、ChatML 等）
* 引入更细粒度的评分维度和加权策略

---
