"""
TurboQuant-v3: Group-wise INT4 quantization with AWQ-style scaling,
protected FP16 channels, and optional low-rank SVD correction.

This module is a clean, reusable port of the original notebook logic.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

__all__ = ["turboquant_v3_compress", "turboquant_v3_decompress", "QuantConfig"]


class QuantConfig:
    """Configuration for TurboQuant-v3 compression."""
    def __init__(
        self,
        group_size: int = 64,
        outlier_keep_ratio: float = 0.02,
        rank: int = 8,
        activation_aware: bool = True,
    ):
        self.group_size = group_size
        self.outlier_keep_ratio = outlier_keep_ratio
        self.rank = rank
        self.activation_aware = activation_aware


# ====================== INT4 Pack / Unpack ======================
def pack_int4(values_int8: np.ndarray) -> np.ndarray:
    """Pack int4 values (-8..7) into uint8 (2 values per byte)."""
    v = np.asarray(values_int8, dtype=np.int8)
    assert np.all((v >= -8) & (v <= 7)), "Values must be in int4 range (-8..7)"
    nibbles = (v & 0x0F).astype(np.uint8)
    if len(nibbles) % 2 != 0:
        nibbles = np.append(nibbles, 0)
    packed = (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)
    return packed


def unpack_int4(packed_uint8: np.ndarray, length: int) -> np.ndarray:
    """Unpack uint8 array back to int4 values (-8..7)."""
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


# ====================== Low-Rank SVD Correction ======================
def lowrank_correction(R: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute low-rank approximation U_corr, V_corr using SVD."""
    if rank <= 0:
        return None, None
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]
    U_corr = (U_k * S_k).astype(np.float16)
    V_corr = Vt_k.astype(np.float16)
    return U_corr, V_corr


# ====================== Main Compress / Decompress ======================
def turboquant_v3_compress(
    W: np.ndarray,
    config: Optional[QuantConfig] = None,
) -> Dict[str, Any]:
    """
    Compress weight matrix W using TurboQuant-v3 algorithm.
    
    Returns a compact compressed representation (dict).
    """
    if config is None:
        config = QuantConfig()

    W = np.asarray(W, dtype=np.float32)
    out_dim, in_dim = W.shape

    # Simulated / placeholder activation statistics (replace with real calibration in LLM)
    if config.activation_aware:
        act_stats = np.random.lognormal(mean=0.0, sigma=0.6, size=in_dim).astype(np.float32)
        act_stats /= (np.max(act_stats) + 1e-9)
    else:
        act_stats = np.ones(in_dim, dtype=np.float32)

    # Select protected columns (most important input channels)
    col_importance = np.mean(np.abs(W), axis=0) * act_stats
    k_keep = max(1, int(in_dim * config.outlier_keep_ratio))
    protected_cols = np.argsort(col_importance)[-k_keep:].astype(np.int32)
    protected_fp16 = W[:, protected_cols].astype(np.float16)

    # Zero out protected columns in base matrix
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

    # Temporary decompress for residual
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

    # Low-rank correction
    if config.rank > 0:
        R = (W - Wq).astype(np.float32)
        U_corr, V_corr = lowrank_correction(R, config.rank)
    else:
        U_corr = V_corr = None

    return {
        "shape": (out_dim, in_dim),
        "group_size": config.group_size,
        "protected_cols": protected_cols,
        "protected_fp16": protected_fp16,
        "packed_rows": packed_rows,
        "scales": scales,
        "rank": config.rank,
        "U_corr": U_corr,
        "V_corr": V_corr,
    }


def turboquant_v3_decompress(comp: Dict[str, Any]) -> np.ndarray:
    """Decompress compressed representation back to weight matrix."""
    out_dim, in_dim = comp["shape"]
    group_size = comp["group_size"]
    groups = (in_dim + group_size - 1) // group_size

    W_rec = np.zeros((out_dim, in_dim), dtype=np.float32)

    for r in range(out_dim):
        for g in range(groups):
            start, end, packed = comp["packed_rows"][r][g]
            scale = float(comp["scales"][r, g])
            length = end - start
            q = unpack_int4(packed, length)
            W_rec[r, start:end] = q.astype(np.float32) * scale

    # Add back protected FP16 channels
    protected_cols = comp["protected_cols"]
    W_rec[:, protected_cols] = comp["protected_fp16"].astype(np.float32)

    # Add low-rank correction if present
    if comp["rank"] > 0 and comp["U_corr"] is not None:
        U_corr = comp["U_corr"].astype(np.float32)
        V_corr = comp["V_corr"].astype(np.float32)
        W_rec += U_corr @ V_corr

    return W_rec
