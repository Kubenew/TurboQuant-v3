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
    W_quantized = np.zeros_like(W).astype(np.int8)
    
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
            scales[g] = 1.0 / np.sqrt(group_importance)
    
    return scales


def svd_low_rank_correction(
    W_residual: np.ndarray,
    rank: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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


def pack_int4(values_int8: np.ndarray) -> np.ndarray:
    """Pack int4 values (-8..7) into uint8 (2 values per byte)."""
    v = np.asarray(values_int8, dtype=np.int8)
    assert np.all((v >= -8) & (v <= 7))
    nibbles = (v & 0x0F).astype(np.uint8)
    if len(nibbles) % 2 != 0:
        nibbles = np.append(nibbles, 0)
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)
    return packed


def unpack_int4(packed_uint8: np.ndarray, length: int) -> np.ndarray:
    """Unpack uint8 back to int4 values."""
    p = np.asarray(packed_uint8, dtype=np.uint8)
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    nibbles = np.empty(len(p) * 2, dtype=np.uint8)
    nibbles[0::2] = low
    nibbles[1::2] = high
    nibbles = nibbles[:length]
    out = nibbles.astype(np.int8)
    out[out >= 8] -= 16
    return out


def turboquant_v3_compress(
    W: np.ndarray,
    config: QuantConfig,
    activations: Optional[np.ndarray] = None,
) -> CompressedWeights:
    """Compress FP32 weights using TurboQuant-v3 algorithm."""
    W = np.asarray(W, dtype=np.float32)
    out_dim, in_dim = W.shape
    
    if config.activation_aware:
        act_stats = np.random.lognormal(0.0, 0.6, in_dim).astype(np.float32)
        act_stats /= (np.max(act_stats) + 1e-9)
    else:
        act_stats = np.ones(in_dim, dtype=np.float32)
    
    col_importance = np.mean(np.abs(W), axis=0) * act_stats
    k_keep = max(1, int(in_dim * config.outlier_keep_ratio))
    protected_cols = np.argsort(col_importance)[-k_keep:].astype(np.int32)
    protected_fp16 = W[:, protected_cols].astype(np.float16)
    
    W_base = W.copy()
    W_base[:, protected_cols] = 0.0
    
    groups = (in_dim + config.group_size - 1) // config.group_size
    packed_rows = []
    scales = np.zeros((out_dim, groups), dtype=np.float16)
    
    for r in range(out_dim):
        row = W_base[r]
        row_packed_groups = []
        for g in range(groups):
            start = g * config.group_size
            end = min(start + config.group_size, in_dim)
            block = row[start:end]
            weighted = np.abs(block) * act_stats[start:end]
            max_abs = np.max(weighted) + 1e-9
            scale = (max_abs / 7.0).astype(np.float16)
            
            q = np.round(block / float(scale)).astype(np.int8)
            q = np.clip(q, -8, 7)
            
            packed = pack_int4(q)
            scales[r, g] = scale
            row_packed_groups.append((start, end, packed))
        packed_rows.append(row_packed_groups)
    
    tmp_comp = {
        "shape": (out_dim, in_dim),
        "group_size": config.group_size,
        "protected_cols": protected_cols,
        "protected_fp16": protected_fp16,
        "packed_rows": packed_rows,
        "scales": scales,
        "rank": 0,
        "U_corr": None,
        "V_corr": None,
    }
    Wq = turboquant_v3_decompress(tmp_comp)
    
    if config.rank > 0:
        R = (W - Wq).astype(np.float32)
        U_corr, V_corr = svd_low_rank_correction(R, config.rank)
    else:
        U_corr = V_corr = None
    
    return CompressedWeights(
        packed_int4=np.array(packed_rows, dtype=object),
        scales=scales,
        zero_points=None,
        protected_channels=protected_fp16,
        protected_indices=protected_cols,
        svd_u=U_corr,
        svd_v=V_corr,
        group_size=config.group_size,
        outlier_keep_ratio=config.outlier_keep_ratio,
        activation_aware=config.activation_aware,
        shape=(out_dim, in_dim),
    )


def turboquant_v3_decompress(comp: CompressedWeights) -> np.ndarray:
    """Decompress to reconstructed weight matrix."""
    if isinstance(comp, dict):
        shape = comp["shape"]
        group_size = comp["group_size"]
    else:
        shape = comp.shape
        group_size = comp.group_size
    
    out_dim, in_dim = shape
    groups = (in_dim + group_size - 1) // group_size
    
    W_rec = np.zeros((out_dim, in_dim), dtype=np.float32)
    
    if isinstance(comp, dict):
        for r in range(out_dim):
            for g in range(groups):
                start, end, packed = comp["packed_rows"][r][g]
                scale = float(comp["scales"][r, g])
                length = end - start
                q = unpack_int4(packed, length)
                W_rec[r, start:end] = q.astype(np.float32) * scale
        
        protected_cols = comp["protected_cols"]
        protected_fp16 = comp["protected_fp16"]
        W_rec[:, protected_cols] = protected_fp16.astype(np.float32)
        
        if comp["rank"] > 0 and comp["U_corr"] is not None:
            W_rec += comp["U_corr"].astype(np.float32) @ comp["V_corr"].astype(np.float32)
    else:
        if comp.packed_int4.dtype == object:
            for r in range(out_dim):
                for g in range(groups):
                    start, end, packed = comp.packed_int4[r][g]
                    scale = float(comp.scales[r, g])
                    length = end - start
                    q = unpack_int4(packed, length)
                    W_rec[r, start:end] = q.astype(np.float32) * scale
            
            if comp.protected_indices is not None:
                protected_cols = comp.protected_indices
                protected_fp16 = comp.protected_channels
                W_rec[:, protected_cols] = protected_fp16.T.astype(np.float32)
        else:
            W_rec = _decompress_simple(comp)
        
        if comp.svd_u is not None and comp.svd_v is not None:
            W_rec += comp.svd_u.astype(np.float32) @ comp.svd_v.astype(np.float32)
    
    return W_rec


def _decompress_simple(comp: CompressedWeights) -> np.ndarray:
    """Simple decompression for packed int4 format."""
    out_dim, in_dim = comp.shape
    group_size = comp.group_size
    groups = (in_dim + group_size - 1) // group_size
    
    W_rec = np.zeros((out_dim, in_dim), dtype=np.float32)
    packed = comp.packed_int4
    
    if packed.ndim == 2 and packed.shape[1] == out_dim * groups:
        for r in range(out_dim):
            for g in range(groups):
                start = g * group_size
                end = min(start + group_size, in_dim)
                length = end - start
                
                packed_idx = r * groups + g
                q_vals = packed[:length, packed_idx] if packed.shape[0] >= length else np.zeros(length)
                scale = float(comp.scales[g, r]) if comp.scales.ndim == 2 else float(comp.scales[g])
                W_rec[r, start:end] = q_vals.astype(np.float32) * scale
    
    return W_rec


def compute_metrics(W: np.ndarray, W_rec: np.ndarray) -> Dict[str, float]:
    """Compute compression quality metrics."""
    W = np.asarray(W, dtype=np.float32)
    W_rec = np.asarray(W_rec, dtype=np.float32)
    
    mse = float(np.mean((W - W_rec) ** 2))
    max_err = float(np.max(np.abs(W - W_rec)))
    rel_err = float(np.linalg.norm(W - W_rec) / (np.linalg.norm(W) + 1e-8))
    
    max_val = max(np.abs(W).max(), np.abs(W_rec).max())
    if max_val > 0 and mse > 0:
        psnr = float(20 * np.log10(max_val / (np.sqrt(mse) + 1e-10)))
    else:
        psnr = float('inf')
    
    return {
        "mse": mse,
        "max_error": max_err,
        "relative_error": rel_err,
        "psnr_db": psnr,
    }
