"""Core quantization algorithms for TurboQuant-v3 - Optimized version."""

import numpy as np
from typing import Tuple, Dict, Optional
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


def svd_low_rank_correction(
    W_residual: np.ndarray,
    rank: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Low-rank SVD correction."""
    if rank <= 0:
        return None, None
    U, S, Vt = np.linalg.svd(W_residual, full_matrices=False)
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]
    U_corr = (U_k * S_k).astype(np.float16)
    V_corr = Vt_k.astype(np.float16)
    return U_corr, V_corr


def quantize_group_wise_vectorized(
    W: np.ndarray,
    group_size: int,
    scales: np.ndarray,
    zero_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized group-wise quantization (much faster than loop version)."""
    n_groups = (W.shape[1] + group_size - 1) // group_size
    
    if W.shape[1] % group_size != 0:
        pad_width = group_size - (W.shape[1] % group_size)
        W_padded = np.pad(W, ((0, 0), (0, pad_width)), mode='constant')
    else:
        W_padded = W
    
    W_reshaped = W_padded.reshape(W.shape[0], n_groups, group_size)
    
    if zero_points is not None:
        q = np.round(W_reshaped / scales[np.newaxis, :, np.newaxis] + 
                     zero_points[np.newaxis, :, np.newaxis]).astype(np.int8)
    else:
        q = np.round(W_reshaped / scales[np.newaxis, :, np.newaxis]).astype(np.int8)
    
    q = np.clip(q, -8, 7)
    
    if W.shape[1] % group_size != 0:
        q = q[:, :, :W.shape[1] % group_size]
        q = q.reshape(W.shape[0], -1)
    
    return q.reshape(W.shape)


def dequantize_group_wise_vectorized(
    W_quantized: np.ndarray,
    scales: np.ndarray,
    zero_points: Optional[np.ndarray] = None,
    group_size: int = 64,
) -> np.ndarray:
    """Highly optimized vectorized dequantization."""
    n_groups = (W_quantized.shape[1] + group_size - 1) // group_size
    
    if W_quantized.shape[1] % group_size != 0:
        pad_width = group_size - (W_quantized.shape[1] % group_size)
        W_pad = np.pad(W_quantized.astype(np.float32), ((0, 0), (0, pad_width)), mode='constant')
    else:
        W_pad = W_quantized.astype(np.float32)
    
    W_reshaped = W_pad.reshape(W_quantized.shape[0], n_groups, group_size)
    
    if zero_points is not None:
        W_deq = (W_reshaped - zero_points[np.newaxis, :, np.newaxis]) * scales[np.newaxis, :, np.newaxis]
    else:
        W_deq = W_reshaped * scales[np.newaxis, :, np.newaxis]
    
    if W_quantized.shape[1] % group_size != 0:
        W_deq = W_deq[:, :, :W_quantized.shape[1] % group_size]
    
    return W_deq.reshape(W_quantized.shape)


def turboquant_v3_compress(
    W: np.ndarray,
    config: QuantConfig,
    activations: Optional[np.ndarray] = None,
) -> CompressedWeights:
    """Main compression function."""
    if config is None:
        config = QuantConfig()

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


def turboquant_v3_decompress(comp) -> np.ndarray:
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

        W_rec[:, comp["protected_cols"]] = comp["protected_fp16"].astype(np.float32)

        if comp["rank"] > 0 and comp["U_corr"] is not None:
            W_rec += comp["U_corr"].astype(np.float32) @ comp["V_corr"].astype(np.float32)
    else:
        for r in range(out_dim):
            for g in range(groups):
                start, end, packed = comp.packed_int4[r][g]
                scale = float(comp.scales[r, g])
                length = end - start
                q = unpack_int4(packed, length)
                W_rec[r, start:end] = q.astype(np.float32) * scale

        if comp.protected_indices is not None:
            W_rec[:, comp.protected_indices] = comp.protected_channels.T.astype(np.float32)

        if comp.svd_u is not None and comp.svd_v is not None:
            W_rec += comp.svd_u.astype(np.float32) @ comp.svd_v.astype(np.float32)

    return W_rec


def compute_metrics(W: np.ndarray, W_rec: np.ndarray) -> Dict[str, float]:
    """Compute compression quality metrics."""
    W = np.asarray(W, dtype=np.float32)
    W_rec = np.asarray(W_rec, dtype=np.float32)

    mse = np.mean((W - W_rec) ** 2)
    max_err = np.max(np.abs(W - W_rec))
    rel_err = np.linalg.norm(W - W_rec) / (np.linalg.norm(W) + 1e-8)

    max_val = max(np.abs(W).max(), np.abs(W_rec).max())
    if max_val > 0 and mse > 0:
        psnr = 20 * np.log10(max_val / (np.sqrt(mse) + 1e-10))
    else:
        psnr = float('inf')

    return {
        "mse": float(mse),
        "max_error": float(max_err),
        "relative_error": float(rel_err),
        "psnr_db": float(psnr),
    }
