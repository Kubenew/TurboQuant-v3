# TurboQuant-v3

**Ultra-efficient post-training INT4 quantization for LLMs**  
Group-wise INT4 + AWQ-style activation scaling + Protected FP16 channels + Optional low-rank SVD correction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)

> Achieve **~4× memory reduction** and **2–3× inference speedup** with near-zero accuracy degradation — no fine-tuning required.

## Why TurboQuant-v3?

LLMs are heavily memory-bound. TurboQuant-v3 delivers a balanced hybrid quantization approach:

- **Group-wise INT4** quantization (per-row groups)
- **AWQ-style activation-aware scaling** using channel importance
- **Protected FP16 channels** for outlier-sensitive input dimensions
- **Optional low-rank SVD correction** to recover quantization error

The result is significantly better reconstruction quality than naive INT4 while staying fully post-training and calibration-only.

## Features

- Pure NumPy + PyTorch compatible core (easy to integrate with HF models)
- Compact packed INT4 representation (2 weights per byte)
- Modular compress / decompress API
- Configurable group size, protection ratio, and SVD rank
- Reproducible error metrics and size analysis
- Ready for extension to full model quantization

## Installation

```bash
# From source (recommended while in development)
git clone https://github.com/Kubenew/TurboQuant-v3.git
cd TurboQuant-v3
pip install -e ".[dev]"
