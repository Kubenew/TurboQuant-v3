"""PyTorch QuantizedLinear layer for TurboQuant-v3."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .config import QuantConfig
from .core import (
    CompressedWeights,
    turboquant_v3_compress,
    turboquant_v3_decompress,
    pack_int4,
    unpack_int4,
)


class QuantizedLinear(nn.Module):
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
        
        self.register_buffer("packed_int4", torch.zeros(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("zero_points", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("protected_channels", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("protected_indices", torch.zeros(0, dtype=torch.long))
        self.register_buffer("svd_u", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("svd_v", torch.zeros(0, dtype=torch.float16))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        
        self.group_size = self.config.group_size
        self.outlier_keep_ratio = self.config.outlier_keep_ratio
        self.activation_aware = self.config.activation_aware
        self._is_quantized = False
        self._original_shape: Optional[Tuple[int, int]] = None
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[QuantConfig] = None,
        activations: Optional[torch.Tensor] = None,
    ) -> "QuantizedLinear":
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
        
        if activations is not None:
            activations_np = activations.detach().float().cpu().numpy()
        else:
            activations_np = None
        
        comp = turboquant_v3_compress(W, config, activations_np)
        
        quantized._load_from_compressed(comp)
        
        if linear.bias is not None:
            quantized.bias = nn.Parameter(
                linear.bias.data.detach().half().clone()
            )
        
        return quantized
    
    def _load_from_compressed(self, comp: CompressedWeights):
        self._original_shape = comp.shape
        
        packed = torch.from_numpy(comp.packed_int4.copy())
        self.packed_int4 = nn.Parameter(packed, requires_grad=False)
        
        scales = torch.from_numpy(comp.scales.copy())
        self.scales = nn.Parameter(scales, requires_grad=False)
        
        if comp.zero_points is not None:
            zero_points = torch.from_numpy(comp.zero_points.copy())
            self.zero_points = nn.Parameter(zero_points, requires_grad=False)
        else:
            self.zero_points = nn.Parameter(
                torch.zeros(len(scales), dtype=torch.float16),
                requires_grad=False,
            )
        
        if comp.protected_channels is not None:
            protected = torch.from_numpy(comp.protected_channels.T.copy())
            self.protected_channels = nn.Parameter(protected, requires_grad=False)
            protected_indices = torch.from_numpy(comp.protected_indices.copy())
            self.protected_indices = nn.Parameter(protected_indices, requires_grad=False)
        else:
            self.protected_channels = nn.Parameter(
                torch.zeros(0, dtype=torch.float16),
                requires_grad=False,
            )
            self.protected_indices = nn.Parameter(
                torch.zeros(0, dtype=torch.long),
                requires_grad=False,
            )
        
        if comp.svd_u is not None and comp.svd_v is not None:
            svd_u = torch.from_numpy(comp.svd_u.copy())
            svd_v = torch.from_numpy(comp.svd_v.copy())
            self.svd_u = nn.Parameter(svd_u, requires_grad=False)
            self.svd_v = nn.Parameter(svd_v, requires_grad=False)
        else:
            self.svd_u = nn.Parameter(torch.zeros(0, dtype=torch.float16), requires_grad=False)
            self.svd_v = nn.Parameter(torch.zeros(0, dtype=torch.float16), requires_grad=False)
        
        self.group_size = comp.group_size
        self.outlier_keep_ratio = comp.outlier_keep_ratio
        self.activation_aware = comp.activation_aware
        self._is_quantized = True
    
    def _dequantize_weights(self) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer is not quantized. Call from_linear() first.")
        
        shape = self._original_shape or (self.out_features, self.in_features)
        
        if self.packed_int4.ndim > 1:
            flat_packed = self.packed_int4.flatten()
            n_elements = np.prod(shape)
            unpacked_flat = torch.zeros(
                flat_packed.size * 2, dtype=torch.uint8, device=self.packed_int4.device
            )
            unpacked_flat[:n_elements // 2] = flat_packed[:n_elements // 2] & 0x0F
            unpacked_flat[n_elements // 2:n_elements - (n_elements % 2)] = (
                flat_packed[:n_elements // 2] >> 4
            ) & 0x0F
            if n_elements % 2 == 1:
                unpacked_flat[-1] = flat_packed[n_elements // 2] & 0x0F
            
            W_quant = (unpacked_flat[:n_elements].float() - 8.0).reshape(shape)
        else:
            n_elements = np.prod(shape)
            unpacked = torch.zeros(n_elements, dtype=torch.uint8, device=self.packed_int4.device)
            unpacked[:n_elements // 2] = self.packed_int4[:n_elements // 2] & 0x0F
            unpacked[n_elements // 2:n_elements - (n_elements % 2)] = (
                self.packed_int4[:n_elements // 2] >> 4
            ) & 0x0F
            if n_elements % 2 == 1:
                unpacked[-1] = self.packed_int4[n_elements // 2] & 0x0F
            W_quant = (unpacked.float() - 8.0).reshape(shape)
        
        n_groups = (self.in_features + self.group_size - 1) // self.group_size
        W_dequant = torch.zeros_like(W_quant)
        
        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            W_dequant[:, start:end] = W_quant[:, start:end] * self.scales[g]
        
        if self.svd_u.numel() > 0 and self.svd_v.numel() > 0:
            W_dequant = W_dequant + self.svd_u @ self.svd_v.T
        
        if self.protected_channels.numel() > 0 and self.protected_indices.numel() > 0:
            for idx, protected in zip(self.protected_indices, self.protected_channels):
                W_dequant[idx] = protected
        
        return W_dequant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_dequant = self._dequantize_weights()
        
        output = x.float() @ W_dequant.T.half()
        
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)
        
        return output.half()
    
    def forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer is not quantized. Call from_linear() first.")
        
        x_f = x.float()
        shape = self._original_shape or (self.out_features, self.in_features)
        n_elements = np.prod(shape)
        
        if self.packed_int4.ndim > 1:
            flat_packed = self.packed_int4.flatten()
            unpacked_flat = torch.zeros(
                flat_packed.size * 2, dtype=torch.uint8, device=self.packed_int4.device
            )
            unpacked_flat[:n_elements // 2] = flat_packed[:n_elements // 2] & 0x0F
            unpacked_flat[n_elements // 2:n_elements - (n_elements % 2)] = (
                flat_packed[:n_elements // 2] >> 4
            ) & 0x0F
            if n_elements % 2 == 1:
                unpacked_flat[-1] = flat_packed[n_elements // 2] & 0x0F
            W_quant = (unpacked_flat[:n_elements].float() - 8.0)
        else:
            unpacked = torch.zeros(n_elements, dtype=torch.uint8, device=self.packed_int4.device)
            unpacked[:n_elements // 2] = self.packed_int4[:n_elements // 2] & 0x0F
            unpacked[n_elements // 2:n_elements - (n_elements % 2)] = (
                self.packed_int4[:n_elements // 2] >> 4
            ) & 0x0F
            if n_elements % 2 == 1:
                unpacked[-1] = self.packed_int4[n_elements // 2] & 0x0F
            W_quant = (unpacked.float() - 8.0)
        
        W_quant = W_quant.reshape(shape)
        
        n_groups = (self.in_features + self.group_size - 1) // self.group_size
        W_fused = torch.zeros(self.out_features, self.in_features, dtype=torch.float32, device=x.device)
        
        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            scale = self.scales[g].item()
            W_fused[:, start:end] = W_quant[:, start:end].to(x.device) * scale
        
        if self.svd_u.numel() > 0 and self.svd_v.numel() > 0:
            W_fused = W_fused + self.svd_u.to(x.device) @ self.svd_v.T.to(x.device)
        
        if self.protected_channels.numel() > 0 and self.protected_indices.numel() > 0:
            for idx, protected in zip(self.protected_indices, self.protected_channels):
                W_fused[idx] = protected.to(x.device)
        
        output = x_f @ W_fused.T
        
        if self.bias is not None:
            output = output + self.bias.float()
        
        return output.half()
    
    def get_weight_stats(self) -> Dict[str, Any]:
        if not self._is_quantized:
            return {"quantized": False}
        
        original_size = self.out_features * self.in_features * 4
        packed_size = self.packed_int4.numel()
        scales_size = self.scales.numel() * 2
        zero_points_size = self.zero_points.numel() * 2
        svd_size = self.svd_u.numel() * 2 + self.svd_v.numel() * 2
        protected_size = self.protected_channels.numel() * 2
        
        total_quantized = packed_size + scales_size + zero_points_size + svd_size + protected_size
        
        return {
            "quantized": True,
            "original_size_bytes": original_size,
            "quantized_size_bytes": total_quantized,
            "compression_ratio": original_size / total_quantized,
            "group_size": self.group_size,
            "outlier_keep_ratio": self.outlier_keep_ratio,
            "has_svd_correction": self.svd_u.numel() > 0,
            "has_protected_channels": self.protected_channels.numel() > 0,
        }
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quantized={self._is_quantized}"
        )


class TurboQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 4,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        self.register_buffer("qweight", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("scales", torch.zeros(0, dtype=torch.float16))
        self.register_buffer("qzeros", torch.zeros(0, dtype=torch.int32))
        self.register_buffer("g_idx", torch.zeros(0, dtype=torch.long))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        
        self._is_quantized = False
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ) -> "TurboQuantLinear":
        config = QuantConfig(group_size=group_size)
        
        quantized = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size,
        )
        
        W = linear.weight.data.detach().float().cpu().numpy()
        comp = turboquant_v3_compress(W, config)
        
        qweight = torch.from_numpy(comp.packed_int4.copy())
        if qweight.dtype == torch.uint8:
            qweight = qweight.to(torch.int32)
        
        scales = torch.from_numpy(comp.scales.copy())
        
        if comp.zero_points is not None:
            qzeros = torch.from_numpy(comp.zero_points.copy()).to(torch.int32)
        else:
            qzeros = torch.zeros(len(scales), dtype=torch.int32)
        
        n_out = W.shape[0]
        n_in = W.shape[1]
        n_groups = (n_in + group_size - 1) // group_size
        g_idx = torch.arange(n_groups, dtype=torch.long).repeat_interleave(group_size)[:n_in]
        
        quantized.qweight = nn.Parameter(qweight, requires_grad=False)
        quantized.scales = nn.Parameter(scales, requires_grad=False)
        quantized.qzeros = nn.Parameter(qzeros, requires_grad=False)
        quantized.g_idx = nn.Parameter(g_idx, requires_grad=False)
        quantized._is_quantized = True
        
        if linear.bias is not None:
            quantized.bias = nn.Parameter(
                linear.bias.data.detach().half().clone()
            )
        
        return quantized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_quantized:
            raise RuntimeError("Layer is not quantized")
        
        raise NotImplementedError(
            "TurboQuantLinear requires CUDA kernel for forward pass. "
            "Use the CUDA extension or dequantize weights first."
        )
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bits={self.bits}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )
