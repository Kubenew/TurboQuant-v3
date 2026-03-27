"""TurboQuant-v3: Ultra-efficient post-training quantization for LLMs."""

__version__ = "0.2.0"

from .config import QuantConfig, TurboQuantConfig, QuantizationVersion
from .core import (
    turboquant_v3_compress,
    turboquant_v3_decompress,
    compute_metrics,
    pack_int4,
    unpack_int4,
    CompressedWeights,
)
from .linear import QuantizedLinear, TurboQuantLinear
from .hf import TurboQuantizer, TurboQuantConfig, quantize_model, save_quantized_model, load_quantized_model
from .hf_modules import TurboQuantizedAttention, TurboQuantizedMLP, TurboQuantizedLayerNorm

from .core_optimized import (
    turboquant_v3_compress as turboquant_v3_compress_v2,
    turboquant_v3_decompress as turboquant_v3_decompress_v2,
    quantize_group_wise_vectorized,
    dequantize_group_wise_vectorized,
    ActivationCollector,
)

try:
    from .cuda_ops import (
        is_cuda_available,
        int4_dequantize,
        int4_pack,
        int4_gemm,
        awq_scale,
        QuantizedLinearCUDA,
    )
    CUDA_OPS_AVAILABLE = True
except ImportError:
    CUDA_OPS_AVAILABLE = False
    is_cuda_available = lambda: False

try:
    from .torch_compile import (
        OptimizedQuantizedLinear,
        TorchCompileQuantizedLinear,
        create_quantized_model,
        CalibrationHook,
    )
    TORCH_COMPILE_AVAILABLE = True
except ImportError:
    TORCH_COMPILE_AVAILABLE = False

try:
    from .benchmark import BenchmarkRunner, BenchmarkResult, quick_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

__all__ = [
    # Config
    "QuantConfig",
    "TurboQuantConfig",
    "QuantizationVersion",
    # Core functions
    "turboquant_v3_compress",
    "turboquant_v3_decompress",
    "compute_metrics",
    "pack_int4",
    "unpack_int4",
    "CompressedWeights",
    # Optimized core
    "turboquant_v3_compress_v2",
    "turboquant_v3_decompress_v2",
    "quantize_group_wise_vectorized",
    "dequantize_group_wise_vectorized",
    "ActivationCollector",
    # Linear layers
    "QuantizedLinear",
    "TurboQuantLinear",
    "OptimizedQuantizedLinear",
    "TorchCompileQuantizedLinear",
    # HuggingFace
    "TurboQuantizer",
    "quantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "TurboQuantizedAttention",
    "TurboQuantizedMLP",
    "TurboQuantizedLayerNorm",
    # CUDA
    "is_cuda_available",
    "CUDA_OPS_AVAILABLE",
    # Torch compile
    "create_quantized_model",
    "CalibrationHook",
    "TORCH_COMPILE_AVAILABLE",
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "quick_benchmark",
    "BENCHMARK_AVAILABLE",
]
