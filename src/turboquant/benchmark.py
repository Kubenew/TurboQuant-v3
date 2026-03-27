"""Performance benchmarks for TurboQuant-v3."""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import statistics

from .config import QuantConfig
from .core import (
    CompressedWeights,
    turboquant_v3_compress as compress_v1,
    turboquant_v3_decompress as decompress_v1,
    pack_int4,
    unpack_int4,
)
from .core_optimized import (
    turboquant_v3_compress as compress_v2,
    turboquant_v3_decompress as decompress_v2,
    quantize_group_wise_vectorized,
    dequantize_group_wise_vectorized,
    ActivationCollector,
)


@dataclass
class BenchmarkResult:
    name: str
    time_ms: float
    memory_mb: float
    compression_ratio: float
    mse: float


class BenchmarkRunner:
    """Run comprehensive benchmarks for TurboQuant-v3."""
    
    def __init__(self, warmup_runs: int = 2, benchmark_runs: int = 10):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def benchmark_compression(
        self,
        W: np.ndarray,
        config: QuantConfig,
        use_optimized: bool = True,
    ) -> Tuple[float, CompressedWeights]:
        """Benchmark weight compression."""
        compress_func = compress_v2 if use_optimized else compress_v1
        
        for _ in range(self.warmup_runs):
            _ = compress_func(W, config)
        
        times = []
        comp = None
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            comp = compress_func(W, config)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return statistics.mean(times), comp
    
    def benchmark_decompression(
        self,
        comp: CompressedWeights,
        use_optimized: bool = True,
    ) -> Tuple[float, np.ndarray]:
        """Benchmark weight decompression."""
        decompress_func = decompress_v2 if use_optimized else decompress_v1
        
        for _ in range(self.warmup_runs):
            _ = decompress_func(comp)
        
        times = []
        W_rec = None
        for _ in range(self.benchmark_runs):
            start = time.perf_counter()
            W_rec = decompress_func(comp)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return statistics.mean(times), W_rec
    
    def benchmark_dequantization(
        self,
        comp: CompressedWeights,
        iterations: int = 100,
    ) -> float:
        """Benchmark dequantization specifically."""
        from .core_optimized import unpack_int4 as unpack_v2
        
        scales = comp.scales.astype(np.float32)
        zero_points = comp.zero_points.astype(np.float32) if comp.zero_points is not None else None
        
        W_quant = unpack_v2(comp.packed_int4, comp.shape).astype(np.float32) - 8
        
        for _ in range(self.warmup_runs):
            _ = dequantize_group_wise_vectorized(W_quant, scales, zero_points, comp.group_size)
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = dequantize_group_wise_vectorized(W_quant, scales, zero_points, comp.group_size)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return statistics.mean(times)
    
    def benchmark_pytorch_forward(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        iterations: int = 50,
    ) -> float:
        """Benchmark PyTorch model forward pass."""
        device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device('cpu')
        model.eval()
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_tensor)
            
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        return statistics.mean(times)
    
    def compare_original_vs_optimized(
        self,
        W: np.ndarray,
        config: QuantConfig,
    ) -> Dict[str, Dict[str, float]]:
        """Compare original vs optimized implementation."""
        results = {}
        
        orig_time, comp_orig = self.benchmark_compression(W, config, use_optimized=False)
        orig_decomp_time, W_rec_orig = self.benchmark_decompression(comp_orig, use_optimized=False)
        
        results['original_compress'] = {'time_ms': orig_time}
        results['original_decompress'] = {'time_ms': orig_decomp_time}
        
        opt_time, comp_opt = self.benchmark_compression(W, config, use_optimized=True)
        opt_decomp_time, W_rec_opt = self.benchmark_decompression(comp_opt, use_optimized=True)
        
        results['optimized_compress'] = {'time_ms': opt_time}
        results['optimized_decompress'] = {'time_ms': opt_decomp_time}
        
        results['speedup_compress'] = orig_time / opt_time if opt_time > 0 else 0
        results['speedup_decompress'] = orig_decomp_time / opt_decomp_time if opt_decomp_time > 0 else 0
        
        return results
    
    def run_full_benchmark(
        self,
        out_features: int = 4096,
        in_features: int = 4096,
        batch_size: int = 4,
        seq_len: int = 512,
    ) -> Dict[str, any]:
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
        orig_comp_time, _ = self.benchmark_compression(W, config, use_optimized=False)
        opt_comp_time, comp = self.benchmark_compression(W, config, use_optimized=True)
        
        print(f"Original compression:  {orig_comp_time:.2f} ms")
        print(f"Optimized compression: {opt_comp_time:.2f} ms")
        print(f"Speedup: {orig_comp_time/opt_comp_time:.2f}x")
        
        print(f"\n--- Decompression Benchmarks ---")
        orig_decomp_time, _ = self.benchmark_decompression(comp, use_optimized=False)
        opt_decomp_time, W_rec = self.benchmark_decompression(comp, use_optimized=True)
        
        print(f"Original decompression:  {orig_decomp_time:.2f} ms")
        print(f"Optimized decompression: {opt_decomp_time:.2f} ms")
        print(f"Speedup: {orig_decomp_time/opt_decomp_time:.2f}x")
        
        print(f"\n--- Memory & Quality ---")
        orig_size = W.nbytes
        comp_size = comp.get_size_bytes()
        ratio = orig_size / comp_size
        
        mse = np.mean((W - W_rec) ** 2)
        
        print(f"Original size: {orig_size / 1024 / 1024:.2f} MB")
        print(f"Compressed size: {comp_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Reconstruction MSE: {mse:.8f}")
        
        print(f"\n{'='*60}")
        
        return {
            'compression': {'original_ms': orig_comp_time, 'optimized_ms': opt_comp_time},
            'decompression': {'original_ms': orig_decomp_time, 'optimized_ms': opt_decomp_time},
            'quality': {'mse': float(mse), 'compression_ratio': ratio},
            'original_size_mb': orig_size / 1024 / 1024,
            'compressed_size_mb': comp_size / 1024 / 1024,
        }


def quick_benchmark():
    """Run a quick benchmark and print results."""
    runner = BenchmarkRunner(warmup_runs=1, benchmark_runs=5)
    results = runner.run_full_benchmark(
        out_features=4096,
        in_features=4096,
        batch_size=4,
        seq_len=512,
    )
    return results


if __name__ == "__main__":
    quick_benchmark()
