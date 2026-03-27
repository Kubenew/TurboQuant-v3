"""TurboQuant-v3: Ultra-efficient post-training quantization for LLMs."""

__version__ = "0.1.0"

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

__all__ = [
    "QuantConfig",
    "TurboQuantConfig",
    "QuantizationVersion",
    "turboquant_v3_compress",
    "turboquant_v3_decompress",
    "compute_metrics",
    "pack_int4",
    "unpack_int4",
    "CompressedWeights",
    "QuantizedLinear",
    "TurboQuantLinear",
    "TurboQuantizer",
    "TurboQuantConfig",
    "quantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "TurboQuantizedAttention",
    "TurboQuantizedMLP",
    "TurboQuantizedLayerNorm",
    "is_cuda_available",
    "CUDA_OPS_AVAILABLE",
]
