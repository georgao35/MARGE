# MARGE: Improving Math Reasoning for LLMs with Guided Exploration

<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2505.12500)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/2505.12500)
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/gao-jingyue/Qwen2.5-Math-7B-MARGE)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
</div>

## Introduction

MARGE enhances mathematical reasoning by systematically exploring intermediate states derived from self-generated solutions, improving both exploration diversity and credit assignment during RL training, improving both accuracy and pass@k without external annotations.

## Run

### Installation

```
pip install -r requirements.txt
pip install -e .
```

### Training

The training commands to reproduce our experiments are listed in
```
bash run.sh
```

More details about the parameters can be found there.

### Evaluation

We adopted the Qwen math evaluation toolkit. Please refer to https://github.com/QwenLM/Qwen2.5-Math


### Checkpoints and Experiments

Below are our trained checkpoints of different backbone models.

| |🤗 huggingface | modelscope |
|---|---|---|
|Qwen2.5-Math-7B-Instruct-MARGE | [link](https://huggingface.co/gao-jingyue/Qwen2.5-Math-7B-MARGE) | [link](https://modelscope.cn/models/keietsu/Qwen2.5-Math-7B-MARGE)|
|Llama3.1-8B-Instruct-MARGE |  |[link](https://modelscope.cn/models/keietsu/Llama3.1-Inst-8B-MARGE) |
|MetaMath-mistral-MARGE | |[link](https://modelscope.cn/models/keietsu/MetaMath-Mistral-7B-MARGE) |

### Data Preparation

Our training set are sampled and deduplicated from opensource datasets, including: 

1. https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K
2. https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified\

You can use your own dataset or any other datasets you like, only need to align to the following format:
```
{"query": math query, "": guidance hit, "gt": correct answer}
```

## 

If you find our paper to be helpful, please cite with the following:
```
@article{gao2025marge,
  title={MARGE: Improving Math Reasoning for LLMs with Guided Exploration},
  author={Gao, Jingyue and Lin, Runji and Lu, Keming and Yu, Bowen and Lin, Junyang and Chen, Jianyu},
  journal={arXiv preprint arXiv:2505.12500},
  year={2025}
}
```
