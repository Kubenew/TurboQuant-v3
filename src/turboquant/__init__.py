"""TurboQuant-v3: Ultra-efficient post-training quantization for LLMs."""

__version__ = "0.2.1"

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
from .hf import TurboQuantizer, quantize_model, save_quantized_model, load_quantized_model
from .hf_modules import TurboQuantizedAttention, TurboQuantizedMLP, TurboQuantizedLayerNorm

try:
    from .torch_compile import (
        OptimizedQuantizedLinear,
        TorchCompileQuantizedLinear,
        create_quantized_model,
    )
    TORCH_COMPILE_AVAILABLE = True
except (ImportError, SyntaxError):
    TORCH_COMPILE_AVAILABLE = False

try:
    from .cuda_ops import (
        is_cuda_available,
        QuantizedLinearCUDA,
    )
    CUDA_OPS_AVAILABLE = True
except (ImportError, SyntaxError):
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
    "quantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "TurboQuantizedAttention",
    "TurboQuantizedMLP",
    "TurboQuantizedLayerNorm",
    "is_cuda_available",
    "CUDA_OPS_AVAILABLE",
    "TORCH_COMPILE_AVAILABLE",
    "OptimizedQuantizedLinear",
    "TorchCompileQuantizedLinear",
    "create_quantized_model",
]
