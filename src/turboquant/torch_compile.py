"""Optimized PyTorch modules with torch.compile() support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from .config import QuantConfig
from .core_optimized import (
    CompressedWeights,
    turboquant_v3_compress,
    turboquant_v3_decompress,
    pack_int4,
    unpack_int4,
    dequantize_group_wise_vectorized,
    ActivationCollector,
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
        self.register_buffer('_protected_channels', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('_protected_indices', torch.zeros(0, dtype=torch.long))
        self.register_buffer('_svd_u', torch.zeros(0, dtype=torch.float16))
        self.register_buffer('_svd_v', torch.zeros(0, dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        
        self._is_quantized = False
        self._original_shape: Optional[Tuple[int, int]] = None
        self._dequantized_cache: Optional[torch.Tensor] = None
    
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
        
        self._packed_int4 = torch.from_numpy(comp.packed_int4.copy())
        self._scales = torch.from_numpy(comp.scales.copy())
        
        if comp.zero_points is not None:
            self._zero_points = torch.from_numpy(comp.zero_points.copy())
        else:
            self._zero_points = torch.zeros(len(self._scales), dtype=torch.float16)
        
        if comp.protected_channels is not None:
            self._protected_channels = torch.from_numpy(comp.protected_channels.T.copy())
            self._protected_indices = torch.from_numpy(comp.protected_indices.copy())
        else:
            self._protected_channels = torch.zeros(0, dtype=torch.float16)
            self._protected_indices = torch.zeros(0, dtype=torch.long)
        
        if comp.svd_u is not None and comp.svd_v is not None:
            self._svd_u = torch.from_numpy(comp.svd_u.copy())
            self._svd_v = torch.from_numpy(comp.svd_v.copy())
        else:
            self._svd_u = torch.zeros(0, dtype=torch.float16)
            self._svd_v = torch.zeros(0, dtype=torch.float16)
        
        self.group_size = comp.group_size
        self._is_quantized = True
    
    def _dequantize_to_float32(self) -> torch.Tensor:
        if self._packed_int4.numel() == 0:
            raise RuntimeError("Layer not quantized")
        
        shape = self._original_shape or (self.out_features, self.in_features)
        device = self._packed_int4.device
        dtype = self._scales.dtype
        
        packed_np = self._packed_int4.cpu().numpy()
        scales_np = self._scales.cpu().float().numpy()
        zero_points_np = self._zero_points.cpu().float().numpy() if self._zero_points.numel() > 0 else None
        
        W_quant = unpack_int4(packed_np, shape).astype(np.float32) - 8
        
        W_deq = dequantize_group_wise_vectorized(
            W_quant, scales_np, zero_points_np, self.group_size
        )
        
        if self._svd_u.numel() > 0 and self._svd_v.numel() > 0:
            svd_u = self._svd_u.cpu().float().numpy()
            svd_v = self._svd_v.cpu().float().numpy()
            W_deq = W_deq + svd_u @ svd_v.T
        
        if self._protected_channels.numel() > 0:
            protected = self._protected_channels.cpu().float().numpy()
            indices = self._protected_indices.cpu().numpy()
            for i, idx in enumerate(indices):
                if idx < W_deq.shape[0]:
                    W_deq[idx] = protected[:, i]
        
        W_tensor = torch.from_numpy(W_deq.astype(np.float16)).to(device=device, dtype=dtype)
        
        return W_tensor
    
    def dequantize_weights(self, cache: bool = True) -> torch.Tensor:
        if cache and self._dequantized_cache is not None:
            return self._dequantized_cache
        
        W_deq = self._dequantize_to_float32()
        
        if cache:
            self._dequantized_cache = W_deq
        
        return W_deq
    
    def invalidate_cache(self) -> None:
        self._dequantized_cache = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_linear() first.")
        
        W_deq = self.dequantize_weights(cache=True)
        
        output = F.linear(x, W_deq, self.bias)
        
        return output
    
    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer not quantized. Call from_linear() first.")
        
        W_deq = self._dequantize_to_float32()
        
        output = F.linear(x, W_deq, self.bias)
        
        return output
    
    def forward_compile_compatible(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
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
                quantized_layer.compile()
            
            if parent_name:
                parent = quantized_model.get_submodule(parent_name)
            else:
                parent = quantized_model
            
            setattr(parent, child_name, quantized_layer)
    
    if device:
        quantized_model = quantized_model.to(device)
    
    return quantized_model


class CalibrationHook:
    """Context manager for collecting activation statistics during model inference."""
    
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        self.model = model
        self.layer_names = layer_names
        self.collector: Optional[ActivationCollector] = None
        self.hooks: List = []
    
    def __enter__(self):
        self.collector = ActivationCollector()
        self._register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_hooks()
    
    def _register_hooks(self):
        import torch.nn as nn
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    x = input[0]
                elif isinstance(input, torch.Tensor):
                    x = input
                else:
                    return
                
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    x_detached = x.detach().float().cpu().numpy()
                    self.collector._activations.append(x_detached)
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if self.layer_names is None or any(ln in name for ln in self.layer_names):
                    self.hooks.append(module.register_forward_hook(hook_fn(name)))
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_stats(self) -> np.ndarray:
        if self.collector is None:
            return None
        return self.collector.get_activation_stats(self.collector._activations)


__all__ = [
    "OptimizedQuantizedLinear",
    "TorchCompileQuantizedLinear",
    "create_quantized_model",
    "CalibrationHook",
    "ActivationCollector",
]
