"""
TurboQuant-v3: Efficient INT4 quantization for LLMs.
"""

__version__ = "0.3.0"

from .quantize import (
    QuantConfig,
    turboquant_v3_compress,
    turboquant_v3_decompress,
)

__all__ = [
    "__version__",
    "QuantConfig",
    "turboquant_v3_compress",
    "turboquant_v3_decompress",
]
