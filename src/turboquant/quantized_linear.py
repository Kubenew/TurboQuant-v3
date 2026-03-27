"""
QuantizedLinear: Drop-in quantized version of nn.Linear using TurboQuant-v3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .quantize import QuantConfig, turboquant_v3_compress, turboquant_v3_decompress


class QuantizedLinear(nn.Module):
    """nn.Linear replacement with TurboQuant-v3 compression."""

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

        self.register_buffer("compressed_weight", None)
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[QuantConfig] = None,
    ) -> "QuantizedLinear":
        """Convert existing nn.Linear to QuantizedLinear."""
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

        weight_np = linear.weight.data.cpu().numpy().astype(np.float32)
        compressed = turboquant_v3_compress(weight_np, config)
        qlinear.compressed_weight = compressed

        if linear.bias is not None:
            qlinear.bias.data.copy_(linear.bias.data)

        return qlinear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.compressed_weight is None:
            raise RuntimeError("Layer not quantized yet.")

        weight_np = turboquant_v3_decompress(self.compressed_weight)
        weight = torch.from_numpy(weight_np).to(device=input.device, dtype=input.dtype)

        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group={self.config.group_size}, rank={self.config.rank}"
