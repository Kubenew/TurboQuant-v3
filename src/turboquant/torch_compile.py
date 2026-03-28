"""Optimized PyTorch modules with torch.compile() support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from .config import QuantConfig
from .core import (
    CompressedWeights,
    turboquant_v3_compress,
    turboquant_v3_decompress,
    pack_int4,
    unpack_int4,
)


class OptimizedQuantizedLinear(nn.Module):
    """Optimized QuantizedLinear with torch.compile() support."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()
        self.group_size = self.config.group_size
        
        self.register_buffer('_packed_int4', torch.zeros(0, dtype=torch.uint8))
        self.register_buffer('_scales', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('_zero_points', torch.zeros(0, dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        
        self._is_quantized = False
        self._original_shape: Optional[Tuple[int, int]] = None
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[QuantConfig] = None,
    ) -> "OptimizedQuantizedLinear":
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(linear)}")
        
        config = config or QuantConfig()
        
        quantized = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )
        
        W = linear.weight.data.detach().float().cpu().numpy()
        comp = turboquant_v3_compress(W, config)
        
        quantized._load_from_compressed(comp)
        
        if linear.bias is not None:
            quantized.bias.data.copy_(linear.bias.data.detach().half())
        
        return quantized
    
    def _load_from_compressed(self, comp: CompressedWeights) -> None:
        self._original_shape = comp.shape
        
        packed = comp.packed_int4
        if hasattr(packed, 'copy'):
            self._packed_int4 = torch.from_numpy(packed.copy())
        else:
            self._packed_int4 = torch.zeros(1, dtype=torch.uint8)
        
        scales = comp.scales
        if hasattr(scales, 'copy'):
            self._scales = torch.from_numpy(scales.copy())
        else:
            self._scales = torch.zeros(1, dtype=torch.float16)
        
        if comp.zero_points is not None:
            zp = comp.zero_points
            if hasattr(zp, 'copy'):
                self._zero_points = torch.from_numpy(zp.copy())
            else:
                self._zero_points = torch.zeros(1, dtype=torch.float16)
        else:
            self._zero_points = torch.zeros(len(self._scales), dtype=torch.float16)
        
        self.group_size = comp.group_size
        self._is_quantized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_linear() first.")
        
        comp = {
            "shape": self._original_shape,
            "group_size": self.group_size,
            "packed_rows": self._packed_int4.cpu().numpy() if self._packed_int4.numel() > 0 else np.array([]),
            "scales": self._scales.cpu().numpy(),
            "rank": 0,
            "protected_cols": np.array([]),
            "protected_fp16": np.array([]),
            "U_corr": None,
            "V_corr": None,
        }
        
        W_np = turboquant_v3_decompress(comp)
        weight = torch.from_numpy(W_np).to(device=x.device, dtype=x.dtype)
        
        output = F.linear(x, weight, self.bias)
        
        return output
    
    def get_weight_stats(self) -> Dict[str, Any]:
        if not self._is_quantized:
            return {"quantized": False}
        
        orig_size = self.out_features * self.in_features * 4
        comp_size = self._packed_int4.numel() + self._scales.numel() * 2
        
        return {
            "quantized": True,
            "original_size_bytes": orig_size,
            "quantized_size_bytes": comp_size,
            "compression_ratio": orig_size / comp_size if comp_size > 0 else 0,
            "group_size": self.group_size,
        }
    
    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group={self.group_size}, quantized={self._is_quantized}"


class TorchCompileQuantizedLinear(OptimizedQuantizedLinear):
    """QuantizedLinear optimized for torch.compile()."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[QuantConfig] = None,
        compile_mode: str = "default",
    ):
        super().__init__(in_features, out_features, bias, config)
        self.compile_mode = compile_mode
        self._compiled_forward = None
    
    def compile(self, mode: Optional[str] = None) -> None:
        mode = mode or self.compile_mode
        if mode == "default":
            self._compiled_forward = torch.compile(self.forward)
        elif mode == "reduce-overhead":
            self._compiled_forward = torch.compile(self.forward, mode="reduce-overhead")
        elif mode == "max-autotune":
            self._compiled_forward = torch.compile(self.forward, mode="max-autotune")
    
    def forward_compiled(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is None:
            self.compile()
        return self._compiled_forward(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is not None:
            return self.forward_compiled(x)
        return super().forward(x)


def create_quantized_model(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
    compile_layers: bool = True,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Convert a PyTorch model to use quantized linear layers."""
    config = config or QuantConfig()
    
    quantized_model = model
    
    for name, module in list(quantized_model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            quantized_layer = OptimizedQuantizedLinear.from_linear(module, config=config)
            
            if compile_layers:
                try:
                    quantized_layer.compile()
                except Exception:
                    pass
            
            if parent_name:
                parent = quantized_model.get_submodule(parent_name)
            else:
                parent = quantized_model
            
            setattr(parent, child_name, quantized_layer)
    
    if device:
        quantized_model = quantized_model.to(device)
    
    return quantized_model


__all__ = [
    "OptimizedQuantizedLinear",
    "TorchCompileQuantizedLinear",
    "create_quantized_model",
]
