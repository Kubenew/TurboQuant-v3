# TurboQuant-v3

**Ultra-efficient post-training INT4 quantization for LLMs**  
Group-wise INT4 + Activation-Aware Scaling (AWQ) + Protected FP16 channels + Low-Rank SVD correction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)

> Drop-in quantization that gives you **~4× memory reduction** and **2–3× inference speedup** with near-zero accuracy loss.

## Why TurboQuant-v3?

Modern LLMs are memory-bound. TurboQuant-v3 attacks this with a carefully balanced hybrid approach:

- **Group-wise INT4** – fine-grained quantization per channel/group
- **AWQ-style activation-aware scaling** – preserves critical weights based on real activations
- **Protected FP16 channels** – keeps outlier-sensitive weights in higher precision
- **Optional Low-Rank SVD correction** – recovers lost information with minimal overhead

Result: Better perplexity and downstream performance than plain INT4 or basic AWQ, while staying fully post-training (no fine-tuning required).

## Features

- Zero training / calibration-only workflow
- Hugging Face + PyTorch native (transformers-compatible)
- Modular design – swap components easily
- GPU-accelerated (CUDA kernels ready for extension)
- Reproducible benchmarks included

## Installation

```bash
# Option 1: Pip (recommended once published)
pip install turboquant-v3

# Option 2: From source (current)
git clone https://github.com/Kubenew/TurboQuant-v3.git
cd TurboQuant-v3
pip install -e .
