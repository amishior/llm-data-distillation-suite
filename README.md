# SFT Dataset Distillation Pipeline

A data distillation pipeline that automates the generation, evaluation, and filtering of SFT (Supervised Fine-Tuning) training data using multiple LLMs.

## Features

* Auto-generate instruction-response pairs from domain knowledge slices (JSONL)
* Multi-dimensional quality evaluation of generated samples using multiple LLMs
* Filtering and semantic deduplication based on scores
* Outputs `train.jsonl` / `val.jsonl` in ShareGPT format and metadata `metadata.json`

Applicable for:
* Distilling high-quality SFT training data from domain knowledge bases
* Automatically constructing training/evaluation sets for domain LLM fine-tuning
* Building scalable, LLM-swappable data production pipelines

---

## Project Structure

```text
sft-distillation/
├─ sft_distill/
│  ├─ __init__.py
│  ├─ config.py          # Configuration template & default parameters
│  ├─ models.py          # Pydantic data models
│  ├─ prompts.py         # Generation/evaluation prompt templates
│  ├─ llm_client.py      # Universal LLM API Client (async)
│  ├─ utils.py           # Common utilities (JSON extraction, etc.)
│  ├─ generator.py       # SFT data generator
│  ├─ evaluator.py       # Quality evaluator
│  ├─ filtering.py       # Filtering and deduplication logic
│  ├─ data_io.py         # JSONL I/O operations
│  ├─ pipeline.py        # DatasetDistillationPipeline main workflow
├─ scripts/
│  └─ run_pipeline.py    # Command-line entry point
├─ requirements.txt
├─ README.md
```

---

## Environment & Installation

### Requirements

* Python 3.10+

### Install Dependencies

```bash
pip install -r requirements.txt
```

You can also install in editable mode (optional):

```bash
pip install -e .
```

Or run directly with:

```bash
PYTHONPATH=. python scripts/run_pipeline.py --input ./knowledge_slices.jsonl
```

---

## Configuration

Default configuration is located in `sft_distill/config.py` via `CONFIG_TEMPLATE`:

```python
CONFIG_TEMPLATE = {
    "generator": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1 ",
        "api_key": "",
        "model": "qwen3-max",
        "max_concurrent": 5,
    },
    "evaluators": {
        "primary": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1 ",
            "api_key": "",
            "model": "Moonshot-Kimi-K2-Instruct",
            "max_concurrent": 3,
        },
        "secondary": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1 ",
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

You can override these configurations via `config.json` (shallow merge):

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

Specify at runtime:

```bash
python scripts/run_pipeline.py \
  --input ./knowledge_slices.jsonl \
  --output ./distilled_dataset \
  --config ./config.json
```

---

## Input Data Format

Input file is JSONL, one knowledge slice per line:

```json
{"id": "slice_1", "content": "Domain knowledge text of thousands to tens of thousands of characters...", "metadata": {"domain": "insurance"}}
```

Field descriptions:

* `id`: Knowledge slice ID (optional, auto-generated if not provided)
* `content`: Long text content (clauses, document snippets, etc.), required
* `metadata`: Optional metadata, e.g., source, domain labels

---

## Output Data Format

After execution, the `output_dir` (default `./distilled_dataset`) will contain:

* `train.jsonl`: Training set (ShareGPT format)
* `val.jsonl`: Validation set (ShareGPT format)
* `generated_raw.jsonl`: Raw generated SFT samples (unfiltered)
* `metadata.json`: Metadata (sample count, average score, etc.)

### ShareGPT Sample Example

```json
{
  "id": "slice_1_gen0_20250301010101",
  "conversations": [
    {
      "from": "human",
      "value": "Instruction text\n\nOptional input content"
    },
    {
      "from": "gpt",
      "value": "High-quality response text"
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

## Running Example

1. Prepare knowledge slices:

```bash
cat > knowledge_slices.jsonl << 'EOF'
{"id": "slice_1", "content": "......", "metadata": {}}
{"id": "slice_2", "content": "......", "metadata": {}}
EOF
```

2. Fill in API Key (in `config.json` or `config.py`)

3. Run pipeline:

```bash
python scripts/run_pipeline.py \
  --input ./knowledge_slices.jsonl \
  --output ./distilled_dataset \
  --config ./config.json
```

---

## Advanced Usage: Semantic Deduplication

`DataFilter` supports passing `embed_fn` for sentence embedding deduplication. For example:

```python
from typing import Iterable
import numpy as np

from sft_distill.filtering import DataFilter


def my_embed_fn(texts: Iterable[str]) -> np.ndarray:
    # Use your own embedding model
    return your_embedder.encode(list(texts))

config = {
    # ... other configurations
    "embed_fn": my_embed_fn,
}
```

When `embed_fn` is provided, deduplication will be based on **cosine similarity**; otherwise, simple **Jaccard text similarity** will be used.

---

## TODO

* Support for resuming from breakpoints (re-running only evaluation or deduplication stages)
* Support for more output formats (Alpaca, ChatML, etc.)
* Introduce more granular scoring dimensions and weighting strategies

---
