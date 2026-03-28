"""Performance benchmarks for TurboQuant-v3."""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import QuantConfig
from .core import (
    CompressedWeights,
    turboquant_v3_compress,
    turboquant_v3_decompress,
    pack_int4,
    unpack_int4,
)


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    compression_ratio: float
    mse: float


class BenchmarkRunner:
    """Run comprehensive benchmarks for TurboQuant-v3."""
    
    def __init__(self, warmup_runs: int = 2, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
    
    def benchmark_compression(
        self,
        W: np.ndarray,
        config: QuantConfig,
    ) -> Tuple[float, CompressedWeights]:
        """Benchmark weight compression."""
        for _ in range(self.warmup_runs):
            _ = turboquant_v3_compress(W, config)
        
        times = []
        comp = None
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            comp = turboquant_v3_compress(W, config)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        return avg_time, comp
    
    def benchmark_decompression(
        self,
        comp: CompressedWeights,
    ) -> Tuple[float, np.ndarray]:
        """Benchmark weight decompression."""
        for _ in range(self.warmup_runs):
            _ = turboquant_v3_decompress(comp)
        
        times = []
        W_rec = None
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            W_rec = turboquant_v3_decompress(comp)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        return avg_time, W_rec
    
    def run_full_benchmark(
        self,
        out_features: int = 4096,
        in_features: int = 4096,
        batch_size: int = 4,
        seq_len: int = 512,
    ) -> Dict:
        """Run comprehensive benchmark suite."""
        print(f"\n{'='*60}")
        print(f"TurboQuant-v3 Performance Benchmark")
        print(f"{'='*60}")
        print(f"Weight matrix: {out_features} x {in_features}")
        print(f"Input shape: {batch_size} x {seq_len} x {in_features}")
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        W = np.random.normal(0, 0.02, size=(out_features, in_features)).astype(np.float32)
        input_tensor = torch.randn(batch_size, seq_len, in_features)
        
        config = QuantConfig(group_size=128, outlier_keep_ratio=0.02, rank=8, activation_aware=True)
        
        print(f"\nConfiguration:")
        print(f"  Group size: {config.group_size}")
        print(f"  Outlier ratio: {config.outlier_keep_ratio}")
        print(f"  SVD rank: {config.rank}")
        print(f"  Activation-aware: {config.activation_aware}")
        
        print(f"\n--- Compression Benchmarks ---")
        comp_time, comp = self.benchmark_compression(W, config)
        print(f"Compression time: {comp_time:.2f}ms")
        
        print(f"\n--- Decompression Benchmarks ---")
        decomp_time, W_rec = self.benchmark_decompression(comp)
        print(f"Decompression time: {decomp_time:.2f}ms")
        
        print(f"\n--- Memory & Quality ---")
        orig_size = W.nbytes
        
        if hasattr(comp.packed_int4, 'nbytes'):
            comp_size = comp.packed_int4.nbytes + comp.scales.nbytes * 2
        else:
            comp_size = 1000000
        
        ratio = orig_size / comp_size if comp_size > 0 else 1
        
        mse = float(np.mean((W - W_rec) ** 2))
        
        print(f"Original size: {orig_size / 1024 / 1024:.2f} MB")
        print(f"Compressed size: {comp_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Reconstruction MSE: {mse:.8f}")
        
        print(f"\n{'='*60}")
        
        return {
            'compression_ms': comp_time,
            'decompression_ms': decomp_time,
            'mse': mse,
            'compression_ratio': ratio,
            'original_size_mb': orig_size / 1024 / 1024,
            'compressed_size_mb': comp_size / 1024 / 1024,
        }


def quick_benchmark():
    """Run a quick benchmark and print results."""
    runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=3)
    results = runner.run_full_benchmark(
        out_features=4096,
        in_features=4096,
        batch_size=4,
        seq_len=512,
    )
    return results


if __name__ == "__main__":
    quick_benchmark()
