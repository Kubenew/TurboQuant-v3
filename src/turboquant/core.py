"""Core quantization algorithms for TurboQuant-v3."""

import numpy as np
from typing import Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass

from .config import QuantConfig


@dataclass
class CompressedWeights:
    packed_int4: np.ndarray
    scales: np.ndarray
    zero_points: Optional[np.ndarray]
    protected_channels: Optional[np.ndarray]
    protected_indices: Optional[np.ndarray]
    svd_u: Optional[np.ndarray]
    svd_v: Optional[np.ndarray]
    group_size: int
    outlier_keep_ratio: float
    activation_aware: bool
    shape: Tuple[int, ...]


def compute_channel_importance(W: np.ndarray, activations: Optional[np.ndarray] = None) -> np.ndarray:
    if activations is not None and W.shape[0] == activations.shape[-1]:
        importance = np.std(activations, axis=0)
    else:
        importance = np.std(W, axis=1)
    return importance


def identify_outliers(W: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    channel_mag = np.max(np.abs(W), axis=1)
    n_outlier = max(1, int(len(channel_mag) * ratio))
    outlier_indices = np.argsort(channel_mag)[-n_outlier:]
    mask = np.zeros(W.shape[0], dtype=bool)
    mask[outlier_indices] = True
    return mask, channel_mag, outlier_indices


def quantize_group_wise(
    W: np.ndarray,
    group_size: int,
    scales: np.ndarray,
    zero_points: Optional[np.ndarray] = None,
    symmetric: bool = False,
) -> np.ndarray:
    n_groups = (W.shape[1] + group_size - 1) // group_size
    W quantized = np.zeros_like(W).astype(np.int8)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, W.shape[1])
        W_group = W[:, start:end]
        
        if symmetric:
            qmin, qmax = -127, 127
            W_quantized[:, start:end] = np.round(W_group / scales[g:g+1]).astype(np.int8)
            W_quantized[:, start:end] = np.clip(W_quantized[:, start:end], qmin, qmax)
        else:
            qmin, qmax = 0, 255
            W_quantized[:, start:end] = np.round(W_group / scales[g:g+1] + zero_points[g:g+1]).astype(np.uint8)
            W_quantized[:, start:end] = np.clip(W_quantized[:, start:end], qmin, qmax)
    
    return W_quantized


def dequantize_group_wise(
    W_quantized: np.ndarray,
    scales: np.ndarray,
    zero_points: Optional[np.ndarray] = None,
    group_size: int = 64,
    symmetric: bool = False,
) -> np.ndarray:
    n_groups = (W_quantized.shape[1] + group_size - 1) // group_size
    W_rec = np.zeros_like(W_quantized, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, W_quantized.shape[1])
        W_group = W_quantized[:, start:end].astype(np.float32)
        
        if symmetric:
            W_rec[:, start:end] = W_group * scales[g]
        else:
            W_rec[:, start:end] = (W_group - zero_points[g]) * scales[g]
    
    return W_rec


def compute_awq_scales(
    W: np.ndarray,
    activations: Optional[np.ndarray] = None,
    group_size: int = 64,
) -> np.ndarray:
    channel_importance = compute_channel_importance(W, activations)
    n_groups = (W.shape[1] + group_size - 1) // group_size
    scales = np.ones(n_groups, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, W.shape[1])
        group_importance = channel_importance[start:end].mean()
        if group_importance > 0:
            scales[g] = 1.0 / (group_importance ** 0.5)
    
    return scales


def svd_low_rank_correction(
    W_residual: np.ndarray,
    rank: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    if rank <= 0:
        return None, None
    
    n, m = W_residual.shape
    k = min(rank, n, m)
    
    U, s, Vt = np.linalg.svd(W_residual, full_matrices=False)
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    
    U_reduced = U_k * np.sqrt(s_k)
    V_reduced = (np.sqrt(s_k)[:, np.newaxis] * Vt_k).T
    
    return U_reduced.astype(np.float16), V_reduced.astype(np.float16)


def pack_int4(int4_array: np.ndarray) -> np.ndarray:
    int4_array = np.asarray(int4_array, dtype=np.uint8)
    if int4_array.ndim == 1:
        n = len(int4_array)
        packed = np.zeros((n + 1) // 2, dtype=np.uint8)
        packed.view(np.uint8)[:n // 2] = (int4_array[:n // 2] & 0x0F) | ((int4_array[1:n // 2 * 2:2] & 0x0F) << 4)
        if n % 2 == 1:
            packed[-1] = int4_array[-1] & 0x0F
    else:
        original_shape = int4_array.shape
        flat = int4_array.flatten()
        n = len(flat)
        packed = np.zeros((n + 1) // 2, dtype=np.uint8)
        packed[:n // 2] = (flat[:n // 2] & 0x0F) | ((flat[1:n // 2 * 2:2] & 0x0F) << 4)
        if n % 2 == 1:
            packed[-1] = flat[-1] & 0x0F
        packed = packed.reshape(-1, *original_shape[1:])
    return packed


def unpack_int4(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    flat_packed = packed.flatten()
    n = np.prod(shape)
    unpacked = np.zeros(n, dtype=np.uint8)
    unpacked[:n // 2] = flat_packed[:n // 2] & 0x0F
    unpacked[n // 2:n - (n % 2)] = (flat_packed[:n // 2] >> 4) & 0x0F
    if n % 2 == 1:
        unpacked[-1] = flat_packed[n // 2] & 0x0F
    
    unpacked = unpacked[:n].reshape(shape)
    if len(packed.shape) > 1:
        unpacked = unpacked.reshape(shape)
    return unpacked


def turboquant_v3_compress(
    W: np.ndarray,
    config: QuantConfig,
    activations: Optional[np.ndarray] = None,
) -> CompressedWeights:
    W_fp32 = np.asarray(W, dtype=np.float32)
    original_shape = W_fp32.shape
    
    outlier_mask, channel_mag, outlier_indices = identify_outliers(W_fp32, config.outlier_keep_ratio)
    
    W_outlier = W_fp32[outlier_mask] if outlier_mask.any() else np.array([]).reshape(0, -1)
    W_main = W_fp32[~outlier_mask]
    
    if config.activation_aware:
        scales = compute_awq_scales(W_main, activations, config.group_size)
    else:
        scales = np.std(W_main, axis=1, keepdims=True) / 127.0
        n_groups = (W_main.shape[1] + config.group_size - 1) // config.group_size
        scales = np.repeat(scales.mean(axis=0, keepdims=True), n_groups, axis=0)
    
    if config.zero_point:
        zero_points = np.zeros(len(scales), dtype=np.float32)
    else:
        zero_points = None
    
    W_quant = quantize_group_wise(
        W_main.T if W_main.ndim > 1 else W_main.reshape(1, -1),
        config.group_size,
        scales,
        zero_points,
        symmetric=not config.zero_point,
    )
    
    if W_main.ndim > 1:
        W_quant = W_quant.T
    
    packed_int4 = pack_int4(W_quant + 8 if config.zero_point else W_quant)
    
    if config.rank > 0:
        W_residual = W_main - quantize_group_wise(
            W_main if W_main.ndim == 1 else W_main.T,
            config.group_size,
            scales,
            zero_points,
            symmetric=not config.zero_point,
        ).T if W_main.ndim > 1 else quantize_group_wise(
            W_main.reshape(1, -1),
            config.group_size,
            scales,
            zero_points,
            symmetric=not config.zero_point,
        )
        if config.rank > 0:
            svd_u, svd_v = svd_low_rank_correction(W_residual, config.rank)
        else:
            svd_u, svd_v = None, None
    else:
        svd_u, svd_v = None, None
    
    protected_channels = W_fp32[outlier_mask].T if outlier_mask.any() else None
    
    return CompressedWeights(
        packed_int4=packed_int4,
        scales=scales.astype(np.float16),
        zero_points=zero_points.astype(np.float16) if zero_points is not None else None,
        protected_channels=protected_channels,
        protected_indices=outlier_indices,
        svd_u=svd_u,
        svd_v=svd_v,
        group_size=config.group_size,
        outlier_keep_ratio=config.outlier_keep_ratio,
        activation_aware=config.activation_aware,
        shape=original_shape,
    )


def turboquant_v3_decompress(comp: CompressedWeights) -> np.ndarray:
    if comp.packed_int4.ndim > 2:
        W_shape = comp.shape
        flat_packed = comp.packed_int4.flatten()
        W_quant_flat = unpack_int4(flat_packed, (flat_packed.size * 2,))[:np.prod(W_shape)]
        W_quant = W_quant_flat.reshape(W_shape)
    else:
        W_quant = unpack_int4(comp.packed_int4, comp.shape)
    
    W_quant = W_quant.astype(np.float32) - 8 if comp.zero_points is not None else W_quant.astype(np.float32)
    
    W_rec = dequantize_group_wise(
        W_quant,
        comp.scales,
        comp.zero_points,
        comp.group_size,
        symmetric=comp.zero_points is None,
    )
    
    if comp.svd_u is not None and comp.svd_v is not None:
        W_rec = W_rec + comp.svd_u @ comp.svd_v.T
    
    if comp.protected_channels is not None and comp.protected_indices is not None:
        W_rec[comp.protected_indices] = comp.protected_channels.T[:len(comp.protected_indices)]
    
    return W_rec.astype(np.float32)


def compute_metrics(W: np.ndarray, W_rec: np.ndarray) -> Dict[str, float]:
    W = np.asarray(W, dtype=np.float32)
    W_rec = np.asarray(W_rec, dtype=np.float32)
    
    mse = np.mean((W - W_rec) ** 2)
    max_err = np.max(np.abs(W - W_rec))
    rel_err = np.linalg.norm(W - W_rec) / (np.linalg.norm(W) + 1e-8)
    
    W_flat = W.flatten()
    W_rec_flat = W_rec.flatten()
    
    max_val = max(np.abs(W_flat).max(), np.abs(W_rec_flat).max())
    if max_val > 0:
        psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-10))
    else:
        psnr = float('inf')
    
    return {
        "mse": float(mse),
        "max_error": float(max_err),
        "relative_error": float(rel_err),
        "psnr_db": float(psnr),
    }
