"""Python wrapper for TurboQuant-v3 CUDA kernels."""

import torch
from typing import Optional, Tuple, Union
import warnings

_cuda_module = None
_cuda_available = None


def _load_cuda_module():
    global _cuda_module, _cuda_available
    
    if _cuda_available is not None:
        return _cuda_available
    
    if not torch.cuda.is_available():
        _cuda_available = False
        return False
    
    try:
        import turboquant_cuda
        _cuda_module = turboquant_cuda
        _cuda_available = True
    except (ImportError, OSError):
        _cuda_available = False
        _cuda_module = None
    
    return _cuda_available


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def int4_dequantize(
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    group_size: int = 64,
) -> torch.Tensor:
    if not _load_cuda_module():
        raise RuntimeError(
            "CUDA module not available. Either CUDA is not installed or "
            "the turboquant_cuda extension was not compiled. "
            "Run: python setup_cuda.py install"
        )
    
    n_rows, n_cols = packed_w.shape[0], packed_w.shape[1] * 2
    
    return _cuda_module.int4_dequantize(
        packed_w, scales, zero_points, group_size, n_rows, n_cols
    )


def int4_pack(
    w_quant: torch.Tensor,
    n_rows: int,
    n_cols: int,
) -> torch.Tensor:
    if not _load_cuda_module():
        raise RuntimeError(
            "CUDA module not available. Either CUDA is not installed or "
            "the turboquant_cuda extension was not compiled. "
            "Run: python setup_cuda.py install"
        )
    
    return _cuda_module.int4_pack(w_quant, n_rows, n_cols)


def int4_gemm(
    input: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 64,
    transposed: bool = False,
) -> torch.Tensor:
    if not _load_cuda_module():
        raise RuntimeError(
            "CUDA module not available. Either CUDA is not installed or "
            "the turboquant_cuda extension was not compiled. "
            "Run: python setup_cuda.py install"
        )
    
    return _cuda_module.int4_gemm(
        input, packed_w, scales, zero_points, bias, group_size, transposed
    )


def awq_scale(
    weights: torch.Tensor,
    activations: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    if not _load_cuda_module():
        raise RuntimeError(
            "CUDA module not available. Either CUDA is not installed or "
            "the turboquant_cuda extension was not compiled. "
            "Run: python setup_cuda.py install"
        )
    
    return _cuda_module.awq_scale(weights, activations, group_size)


def int4_dequantize_cpu(
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    group_size: int = 64,
) -> torch.Tensor:
    n_rows, n_cols = packed_w.shape[0], packed_w.shape[1] * 2
    output = torch.empty((n_rows, n_cols), dtype=torch.float32, device="cpu")
    
    for row in range(n_rows):
        for col in range(n_cols):
            packed_col = col // 2
            elem_idx = col % 2
            packed_idx = row * ((n_cols + 1) // 2) + packed_col
            
            packed_val = packed_w[row, packed_col].item()
            if elem_idx == 0:
                val = packed_val & 0x0F
            else:
                val = (packed_val >> 4) & 0x0F
            
            if val >= 8:
                val = val - 16
            
            group_idx = col // group_size
            scale = scales[group_idx].item()
            zp = zero_points[group_idx].item() if zero_points is not None else 0.0
            
            output[row, col] = (val - zp) * scale
    
    return output


def int4_gemm_cpu(
    input: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 64,
) -> torch.Tensor:
    W_dequant = int4_dequantize_cpu(packed_w, scales, zero_points, group_size)
    output = input.float() @ W_dequant.T
    
    if bias is not None:
        output = output + bias
    
    return output


class TurboQuantGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        packed_w: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor],
        group_size: int,
    ):
        if input.is_cuda and _load_cuda_module():
            output = int4_gemm(input, packed_w, scales, zero_points, None, group_size)
        else:
            output = int4_gemm_cpu(input, packed_w, scales, zero_points, None, group_size)
        
        ctx.save_for_backward(packed_w, scales, zero_points)
        ctx.group_size = group_size
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Gradient computation not supported for quantized GEMM")


class QuantizedLinearCUDA(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        self.register_buffer("packed_w", torch.zeros(0, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("zero_points", torch.zeros(0, dtype=torch.float32))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
    
    def load_from_quantized_linear(self, quantized_layer):
        from ..linear import QuantizedLinear
        
        if not isinstance(quantized_layer, QuantizedLinear):
            raise TypeError("Expected QuantizedLinear instance")
        
        self.packed_w = quantized_layer.packed_int4.clone()
        self.scales = quantized_layer.scales.float().clone()
        
        if quantized_layer.zero_points is not None and quantized_layer.zero_points.numel() > 0:
            self.zero_points = quantized_layer.zero_points.float().clone()
        else:
            self.zero_points = torch.zeros_like(self.scales)
        
        if quantized_layer.bias is not None:
            self.bias = quantized_layer.bias.float().clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _load_cuda_module() or not x.is_cuda:
            output = int4_gemm_cpu(
                x.float(), self.packed_w, self.scales,
                self.zero_points, self.bias, self.group_size
            )
        else:
            if self.packed_w.device != x.device:
                self.packed_w = self.packed_w.to(x.device)
                self.scales = self.scales.to(x.device)
                self.zero_points = self.zero_points.to(x.device)
                if self.bias is not None:
                    self.bias = self.bias.to(x.device)
            
            output = int4_gemm(
                x.float(), self.packed_w, self.scales,
                self.zero_points, self.bias, self.group_size
            )
        
        return output.half() if output.dtype == torch.float32 else output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}, "
            f"cuda_available={is_cuda_available()}"
        )


__all__ = [
    "is_cuda_available",
    "int4_dequantize",
    "int4_pack",
    "int4_gemm",
    "awq_scale",
    "int4_dequantize_cpu",
    "int4_gemm_cpu",
    "QuantizedLinearCUDA",
    "TurboQuantGemmFunction",
]
