# SE-GA: Memory-Augmented Self-Evolution for GUI Agents&#x20;

[![arXiv](https://img.shields.io/badge/arXiv-2605.16883-b31b1b.svg)](https://arxiv.org/abs/2605.16883)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Paper: ICML 2026](https://img.shields.io/badge/Paper-ICML%202026-red.svg)](https://icml.cc/virtual/2026/poster/65853)
[![Dataset](https://img.shields.io/badge/🤗_HuggingFace-SE--GA-yellow)](https://huggingface.co/datasets/waterphd/SE-GA-dataset)

This is the implementation code for the paper "SE-GA: Memory-Augmented Self-Evolution for GUI Agents". This work has been accepted by ICML 2026.

***

## Environment Setup

### Requirements

- Python >= 3.11
- PyTorch >= 2.8.0 (with CUDA)
- NVIDIA GPUs&#x20;

### Install Dependencies

```bash
pip install -r requirements.txt

or

pip install -e .
```

Or install individually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.46.0 peft accelerate deepspeed
pip install vllm
pip install swanlab  # optional, for logging
```

### Environment Variables

For TTME memory (if using OpenAI-compatible embeddings):

```bash
export EMBEDDING_API_KEY=your-api-key
export EMBEDDING_BASE_URL=your-base-url
export EMBEDDING_MODEL=text-embedding-3-small
```

If you experience slow downloads from HuggingFace, set the mirror endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

You can also add this to your config YAML:

```yaml
env:
  HF_ENDPOINT: "https://hf-mirror.com"
```

***

## MASE-Training Pipeline

### Pipeline Overview

The training pipeline consists of three sequential stages:

| Stage             | Name          | Method                             | Goal                           | Model Input                 |
| ----------------- | ------------- | ---------------------------------- | ------------------------------ | --------------------------- |
| **Grounding SFT** | Grounding SFT | Supervised Fine-Tuning             | Learn basic GUI action format  | Screenshot + Task           |
| **Planning SFT**  | Planning SFT  | Supervised Fine-Tuning             | Learn reasoning with thinking  | Screenshot + Task + History |
| **RL**            | RL            | Group Relative Policy Optimization | Optimize action quality via RL | Screenshot + Task + History |

The model progressively learns:

1. **Grounding SFT**: Simple action generation (`{"action_type": "CLICK", "action_info": [x, y]}`)
2. **Planning SFT**: Structured thinking before action (`<thinking>...<analysis>...<reasoning>...<instruction>...</thinking><answer>...</answer>`)
3. **RL**: Improved action quality through reinforcement learning with custom reward functions

### Configuration

All training parameters are managed in a single YAML file: [`configs/default.yaml`](configs/default.yaml).

The configuration is organized into five sections:

```yaml
model:           # Shared model settings (dtype, attention, image resolution)
env:             # Logging project name
grounding_sft:   # Grounding SFT: model_id, hyperparameters, data paths
planning_sft:    # Planning SFT: model_id, hyperparameters, data paths
grpo:            # GRPO RL: model_id, hyperparameters, LoRA config, data paths
```

<br />

### Quick Start

**One-click full pipeline:**

```bash
bash run.sh
```

This runs all stages sequentially: Grounding SFT → Planning SFT → RL.

**Using Python directly:**

```bash
python -m gui_sega.run --config configs/default.yaml
```

### Running Individual Stages

```bash
# Run only SFT (Grounding + Planning sequentially)
bash run_sft.sh

# Run only Grounding SFT
bash run_sft.sh --stage grounding_sft

# Run only Planning SFT
bash run_sft.sh --stage planning_sft

# Run only GRPO RL
bash run_rl.sh

# Specify GPUs
bash run.sh --gpus 0,1 --num_gpus 2

# Use custom config
bash run.sh --config configs/my_experiment.yaml
```

<br />

***

## TTME-Inference:

The **Test-Time Memory Extension (TTME)** module provides a hierarchical memory system for GUI agents during inference, addressing failures in long-horizon tasks caused by constrained context windows.

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│                    TTMEMemory                        │
│                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │   Episodic   │ │   Semantic   │ │ Experiential │ │
│  │   Memory     │ │   Memory     │ │   Memory     │ │
│  │   (M_EPI)    │ │   (M_SEM)    │ │   (M_EXP)    │ │
│  ├──────────────┤ ├──────────────┤ ├──────────────┤ │
│  │ Short-term   │ │ Long-term    │ │ Historical   │ │
│  │ working      │ │ knowledge    │ │ task         │ │
│  │ memory       │ │ repository   │ │ strategies   │ │
│  ├──────────────┤ ├──────────────┤ ├──────────────┤ │
│  │ <o_k, a_k,   │ │ <k_sem, d>   │ │ <τ, g(τ),   │ │
│  │  o_{k+1}>    │ │              │ │  k_intent,   │ │
│  │              │ │              │ │  k_task>     │ │
│  ├──────────────┤ ├──────────────┤ ├──────────────┤ │
│  │ Sliding      │ │ Embedding    │ │ Hybrid       │ │
│  │ Window       │ │ Similarity   │ │ Retrieval    │ │
│  │ (horizon H)  │ │ (Top-K)      │ │ (λ·text +    │ │
│  │              │ │              │ │  (1-λ)·visual│ │
│  └──────────────┘ └──────────────┘ └──────────────┘ │
│                                                      │
│  Output: M_retrieved = C_epi + C_sem + C_exp        │
└─────────────────────────────────────────────────────┘
```

| Layer            | Function                                  | Data Structure                                | Retrieval                               |
| ---------------- | ----------------------------------------- | --------------------------------------------- | --------------------------------------- |
| **Episodic**     | Track recent actions within current task  | `<observation, action, next_observation>`     | Sliding window (horizon H)              |
| **Semantic**     | Store universal interaction rules         | `<description, embedding>`                    | Cosine similarity Top-K                 |
| **Experiential** | Recall strategies from similar past tasks | `<trajectory, summary, intent_emb, task_emb>` | Hybrid: λ·text\_sim + (1-λ)·visual\_sim |

Run the full example:

```bash
python -m gui_sega.ttme.memory_example
```

<br />

***

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{sega2026gui,
  title={SE-GA: Memory-Augmented Self-Evolution for GUI Agents},
  author={Shilong Jin, Lanjun Wang, and Zhuosheng Zhang},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgements

This project builds upon [TRL](https://github.com/huggingface/trl) (HuggingFace) and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Alibaba Cloud).
