"""
TurboQuant-v3: Modular INT4 Quantization with AWQ-style scaling,
protected FP16 channels, and optional low-rank SVD correction.

Supports Hugging Face transformers models (linear layers only for now).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

__all__ = ["quantize_model", "QuantConfig"]

class QuantConfig:
    """Configuration for TurboQuant-v3 quantization."""
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        use_awq: bool = True,
        protect_channels: bool = True,
        protect_ratio: float = 0.01,      # fraction of channels to protect in FP16
        apply_svd_correction: bool = True,
        svd_rank: int = 32,
        calibration_samples: int = 128,
        device: str = "cuda",
    ):
        self.bits = bits
        self.group_size = group_size
        self.use_awq = use_awq
        self.protect_channels = protect_channels
        self.protect_ratio = protect_ratio
        self.apply_svd_correction = apply_svd_correction
        self.svd_rank = svd_rank
        self.calibration_samples = calibration_samples
        self.device = device


def _get_scale_and_zero_point(weight: torch.Tensor, group_size: int, bits: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group scale and zero-point for symmetric/asymmetric quantization."""
    orig_shape = weight.shape
    weight = weight.view(-1, group_size)
    max_val = weight.abs().max(dim=1, keepdim=True)[0]
    scale = max_val / (2 ** (bits - 1) - 1)
    scale = torch.clamp(scale, min=1e-5)
    zero_point = torch.zeros_like(scale)
    return scale.view(-1, 1).expand_as(weight).view(orig_shape), zero_point


def _awq_scale(weight: torch.Tensor, act_scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Activation-aware weight scaling (AWQ-style)."""
    if act_scales is None:
        return torch.ones_like(weight)
    act_scales = act_scales.view(-1, 1)
    scale = act_scales.pow(0.5)
    return scale


def _protect_outlier_channels(weight: torch.Tensor, protect_ratio: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Protect top outlier channels by keeping them in FP16."""
    if protect_ratio <= 0:
        return weight, torch.zeros_like(weight), torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)

    channel_norms = weight.abs().max(dim=1)[0] if weight.dim() == 2 else weight.abs().max(dim=-1)[0]
    num_protect = max(1, int(protect_ratio * weight.shape[0]))
    _, protect_idx = torch.topk(channel_norms, num_protect)
    
    mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
    mask[protect_idx] = True
    
    protected_weight = weight.clone()
    protected_weight[~mask] = 0.0   # zero out non-protected for main quant path
    
    return weight, protected_weight, mask


def _low_rank_svd_correction(weight: torch.Tensor, error: torch.Tensor, rank: int) -> torch.Tensor:
    """Apply low-rank SVD correction to recover quantization error."""
    if rank <= 0 or error.norm() < 1e-6:
        return torch.zeros_like(weight)
    
    U, S, Vh = torch.linalg.svd(error.float(), full_matrices=False)
    correction = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    return correction.to(weight.dtype)


@torch.no_grad()
def quantize_linear_layer(
    layer: nn.Linear,
    config: QuantConfig,
    act_scales: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Quantize a single nn.Linear layer using TurboQuant-v3 method."""
    weight = layer.weight.data.to(config.device).float()
    orig_dtype = layer.weight.dtype

    # Step 1: AWQ-style scaling
    awq_scale = _awq_scale(weight, act_scales, config.group_size) if config.use_awq else torch.ones_like(weight)

    scaled_weight = weight * awq_scale

    # Step 2: Protect outlier channels
    if config.protect_channels:
        weight_to_quant, protected_weight, protect_mask = _protect_outlier_channels(
            scaled_weight, config.protect_ratio
        )
    else:
        weight_to_quant = scaled_weight
        protected_weight = torch.zeros_like(scaled_weight)
        protect_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)

    # Step 3: Group-wise INT4 quantization
    scale, zero_point = _get_scale_and_zero_point(weight_to_quant, config.group_size, config.bits)

    # Quantize to INT4 (stored as int8 for simplicity, can be packed later)
    quantized_weight = torch.round(weight_to_quant / scale).clamp(
        - (2 ** (config.bits - 1)), 2 ** (config.bits - 1) - 1
    ).to(torch.int8)

    # Step 4: Optional SVD correction on quantization error
    if config.apply_svd_correction:
        dequant_approx = quantized_weight.to(torch.float32) * scale
        error = weight_to_quant - dequant_approx
        correction = _low_rank_svd_correction(error, config.svd_rank)
    else:
        correction = torch.zeros_like(weight)

    # Reconstruct final weight (protected + dequantized + correction)
    final_weight = protected_weight + (quantized_weight.to(torch.float32) * scale) + correction

    # Store quantized parameters
    layer.weight.data = final_weight.to(orig_dtype)
    layer.register_buffer("quantized_weight", quantized_weight.to("cpu"))
    layer.register_buffer("scale", scale.to(orig_dtype).to("cpu"))
    layer.register_buffer("protect_mask", protect_mask.to("cpu"))

    # TODO: Add custom forward with dequantization for real deployment
    return layer


def quantize_model(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
    calibration_data: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Quantize an entire model (or selected modules) with TurboQuant-v3.
    
    Args:
        model: Hugging Face or PyTorch model
        config: Quantization configuration
        calibration_data: Optional tensor for computing activation scales (batch of inputs)
    
    Returns:
        Quantized model (in-place)
    """
    if config is None:
        config = QuantConfig()

    model.to(config.device)
    model.eval()

    # TODO: Compute activation scales from calibration_data (simplified placeholder)
    act_scales_dict: Dict[str, torch.Tensor] = {}

    # Quantize all Linear layers (extend to other modules if needed)
    for name, module in tqdm(list(model.named_modules()), desc="Quantizing layers"):
        if isinstance(module, nn.Linear) and "lm_head" not in name:  # usually skip lm_head
            act_scale = act_scales_dict.get(name, None)
            quantize_linear_layer(module, config, act_scale)

    print(f"✅ TurboQuant-v3 quantization complete: {config.bits}-bit with group_size={config.group_size}")
    return model
