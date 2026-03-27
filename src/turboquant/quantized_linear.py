"""
QuantizedLinear: Drop-in replacement for torch.nn.Linear using TurboQuant-v3 compression.

This module allows seamless integration with Hugging Face models and PyTorch workflows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .quantize import (
    turboquant_v3_compress,
    turboquant_v3_decompress,
    QuantConfig,
)


class QuantizedLinear(nn.Module):
    """
    Quantized replacement for nn.Linear using TurboQuant-v3 (INT4 + AWQ + Protected Channels + SVD).
    
    The weight is stored in a highly compressed format and dequantized on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[QuantConfig] = None,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()

        # Will be set during quantization
        self.register_buffer("compressed_weight", None)  # stores the dict from turboquant_v3_compress
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None

        # Placeholder for original weight shape
        self.weight_shape = (out_features, in_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[QuantConfig] = None,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> "QuantizedLinear":
        """
        Convert a regular nn.Linear layer into a QuantizedLinear layer.
        
        Args:
            linear: Original nn.Linear module to quantize
            config: TurboQuant-v3 configuration
            calibration_data: Optional activations for better AWQ scaling (future enhancement)
        """
        if config is None:
            config = QuantConfig()

        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            config=config,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # Extract weight as numpy for compression (TurboQuant-v3 currently uses NumPy)
        weight_np = linear.weight.data.cpu().numpy().astype(np.float32)

        # Compress using TurboQuant-v3
        compressed = turboquant_v3_compress(weight_np, config)

        # Store compressed representation
        qlinear.compressed_weight = compressed

        # Copy bias if present
        if linear.bias is not None:
            qlinear.bias.data.copy_(linear.bias.data)

        return qlinear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        if self.compressed_weight is None:
            raise RuntimeError("QuantizedLinear has not been quantized yet.")

        # Decompress back to full weight (float32)
        weight_np = turboquant_v3_decompress(self.compressed_weight)

        # Convert to torch tensor on the correct device/dtype
        weight = torch.from_numpy(weight_np).to(
            device=input.device,
            dtype=input.dtype
        )

        # Standard linear forward
        output = F.linear(input, weight, self.bias)

        return output

    def extra_repr(self) -> str:
        """Nice representation in model summary."""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits=4, group_size={self.config.group_size}, "
                f"protected_ratio={self.config.outlier_keep_ratio}, rank={self.config.rank}")
