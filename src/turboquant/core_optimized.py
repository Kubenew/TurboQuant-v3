"""Core quantization algorithms for TurboQuant-v3 with optimized performance."""

import numpy as np
from typing import Tuple, Dict, Optional, Union, List, Callable
from dataclasses import dataclass, field, asdict
import json
import os

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
    version: str = "3.0"

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save compressed weights to disk."""
        path = os.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path.with_suffix('.packed_int4.npy'), self.packed_int4)
        np.save(path.with_suffix('.scales.npy'), self.scales)
        
        if self.zero_points is not None:
            np.save(path.with_suffix('.zero_points.npy'), self.zero_points)
        if self.protected_channels is not None:
            np.save(path.with_suffix('.protected_channels.npy'), self.protected_channels)
        if self.protected_indices is not None:
            np.save(path.with_suffix('.protected_indices.npy'), self.protected_indices)
        if self.svd_u is not None:
            np.save(path.with_suffix('.svd_u.npy'), self.svd_u)
        if self.svd_v is not None:
            np.save(path.with_suffix('.svd_v.npy'), self.svd_v)
        
        metadata = {
            'shape': self.shape,
            'group_size': self.group_size,
            'outlier_keep_ratio': self.outlier_keep_ratio,
            'activation_aware': self.activation_aware,
            'version': self.version,
            'has_zero_points': self.zero_points is not None,
            'has_protected_channels': self.protected_channels is not None,
            'has_protected_indices': self.protected_indices is not None,
            'has_svd_u': self.svd_u is not None,
            'has_svd_v': self.svd_v is not None,
        }
        with open(path.with_suffix('.meta.json'), 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> 'CompressedWeights':
        """Load compressed weights from disk."""
        path = os.Path(path)
        
        with open(path.with_suffix('.meta.json'), 'r') as f:
            meta = json.load(f)
        
        packed_int4 = np.load(path.with_suffix('.packed_int4.npy'))
        scales = np.load(path.with_suffix('.scales.npy'))
        zero_points = np.load(path.with_suffix('.zero_points.npy')) if meta['has_zero_points'] else None
        protected_channels = np.load(path.with_suffix('.protected_channels.npy')) if meta['has_protected_channels'] else None
        protected_indices = np.load(path.with_suffix('.protected_indices.npy')) if meta['has_protected_indices'] else None
        svd_u = np.load(path.with_suffix('.svd_u.npy')) if meta['has_svd_u'] else None
        svd_v = np.load(path.with_suffix('.svd_v.npy')) if meta['has_svd_v'] else None
        
        return cls(
            packed_int4=packed_int4,
            scales=scales,
            zero_points=zero_points,
            protected_channels=protected_channels,
            protected_indices=protected_indices,
            svd_u=svd_u,
            svd_v=svd_v,
            group_size=meta['group_size'],
            outlier_keep_ratio=meta['outlier_keep_ratio'],
            activation_aware=meta['activation_aware'],
            shape=tuple(meta['shape']),
            version=meta.get('version', '3.0'),
        )

    def get_size_bytes(self) -> int:
        """Get total size of compressed weights in bytes."""
        size = self.packed_int4.nbytes + self.scales.nbytes
        if self.zero_points is not None:
            size += self.zero_points.nbytes
        if self.protected_channels is not None:
            size += self.protected_channels.nbytes
        if self.protected_indices is not None:
            size += self.protected_indices.nbytes
        if self.svd_u is not None:
            size += self.svd_u.nbytes
        if self.svd_v is not None:
            size += self.svd_v.nbytes
        return size


class ActivationCollector:
    """Collect real activation statistics for activation-aware quantization."""
    
    def __init__(self, group_size: int = 64):
        self.group_size = group_size
        self._hook_handle: Optional = None
        self._activations: List[np.ndarray] = []
        self._input_stats: Optional[np.ndarray] = None
    
    def register_hook(self, model, layer_names: Optional[List[str]] = None) -> None:
        """Register forward hooks to collect activations."""
        import torch.nn as nn
        
        def hook_fn(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            
            if isinstance(x, torch.Tensor):
                x_np = x.detach().float().cpu().numpy()
                self._activations.append(x_np)
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if layer_names is None or any(ln in name for ln in layer_names):
                    hooks.append(module.register_forward_hook(hook_fn))
        
        self._hooks = hooks
    
    def unregister_hook(self) -> None:
        """Unregister all hooks."""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
            self._activations = []
    
    def get_activation_stats(self, activations: List[np.ndarray]) -> np.ndarray:
        """Compute per-channel activation statistics from collected activations."""
        if not activations:
            return np.ones(self.group_size, dtype=np.float32)
        
        all_stats = []
        for act in activations:
            if act.ndim == 3:
                act = act.reshape(-1, act.shape[-1])
            elif act.ndim == 2:
                pass
            else:
                continue
            
            channel_std = np.std(act, axis=0)
            all_stats.append(channel_std)
        
        if all_stats:
            mean_stats = np.mean(all_stats, axis=0)
            return mean_stats
        return np.ones(self.group_size, dtype=np.float32)
    
    def compute_scales_from_activations(
        self, 
        weights: np.ndarray, 
        activations: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """Compute AWQ-style scales using activation statistics."""
        if activations is not None:
            act_stats = self.get_activation_stats(activations)
        elif self._activations:
            act_stats = self.get_activation_stats(self._activations)
        else:
            act_stats = np.ones(weights.shape[1], dtype=np.float32)
        
        act_stats = np.clip(act_stats, 1e-8, None)
        
        n_groups = (weights.shape[1] + self.group_size - 1) // self.group_size
        scales = np.ones(n_groups, dtype=np.float32)
        
        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, weights.shape[1])
            group_act_importance = act_stats[start:end].mean()
            if group_act_importance > 0:
                scales[g] = 1.0 / np.sqrt(group_act_importance)
        
        return scales


def compute_channel_importance(
    W: np.ndarray, 
    activations: Optional[np.ndarray] = None
) -> np.ndarray:
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
        qmin, qmax = 0, 255
        q = np.round(W_reshaped / scales[np.newaxis, :, np.newaxis] + 
                     zero_points[np.newaxis, :, np.newaxis]).astype(np.int8)
    else:
        qmin, qmax = -127, 127
        q = np.round(W_reshaped / scales[np.newaxis, :, np.newaxis]).astype(np.int8)
    
    q = np.clip(q, qmin, qmax)
    
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


def pack_int4(int4_array: np.ndarray) -> np.ndarray:
    """Pack int4 values into uint8 bytes."""
    int4_array = np.asarray(int4_array, dtype=np.int8)
    int4_array = np.clip(int4_array, -8, 7)
    
    if int4_array.ndim == 1:
        n = len(int4_array)
        nibbles = (int4_array & 0x0F).astype(np.uint8)
        
        if n % 2 == 0:
            packed = np.zeros(n // 2, dtype=np.uint8)
            packed[:] = nibbles[0::2] | (nibbles[1::2] << 4)
        else:
            padded = np.append(nibbles, 0)
            packed = np.zeros((n + 1) // 2, dtype=np.uint8)
            packed[:n // 2] = nibbles[0:n // 2] | (nibbles[1:n // 2 + 1] << 4)
            packed[n // 2] = nibbles[0] & 0x0F
        
        return packed
    else:
        original_shape = int4_array.shape
        flat = int4_array.flatten()
        n = len(flat)
        
        nibbles = (flat & 0x0F).astype(np.uint8)
        
        if n % 2 == 0:
            packed = nibbles[0::2] | (nibbles[1::2] << 4)
        else:
            padded = np.append(nibbles, 0)
            packed = np.zeros((n + 1) // 2, dtype=np.uint8)
            packed[:n // 2] = nibbles[0:n // 2] | (nibbles[1:n // 2 + 1] << 4)
            packed[n // 2] = nibbles[0] & 0x0F
        
        return packed.reshape(-1, *original_shape[1:])


def unpack_int4(packed: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Unpack uint8 bytes back to int4 values."""
    packed = np.asarray(packed, dtype=np.uint8)
    
    if packed.ndim == 1:
        flat_packed = packed
    else:
        flat_packed = packed.flatten()
    
    n = np.prod(shape)
    nibbles = np.empty(len(flat_packed) * 2, dtype=np.uint8)
    nibbles[0::2] = flat_packed & 0x0F
    nibbles[1::2] = (flat_packed >> 4) & 0x0F
    
    unpacked = nibbles[:n].astype(np.int8)
    unpacked[unpacked >= 8] -= 16
    
    return unpacked.reshape(shape)


def turboquant_v3_compress(
    W: np.ndarray,
    config: QuantConfig,
    activations: Optional[np.ndarray] = None,
    act_stats: Optional[np.ndarray] = None,
) -> CompressedWeights:
    """Compress FP32 weights using TurboQuant-v3 algorithm."""
    W_fp32 = np.asarray(W, dtype=np.float32)
    original_shape = W_fp32.shape
    
    outlier_mask, channel_mag, outlier_indices = identify_outliers(W_fp32, config.outlier_keep_ratio)
    W_main = W_fp32[~outlier_mask].reshape(-1, W_fp32.shape[1])
    
    if act_stats is not None:
        scales = np.ones((W_main.shape[0], (W_main.shape[1] + config.group_size - 1) // config.group_size), 
                         dtype=np.float32)
        for i in range(W_main.shape[0]):
            if config.activation_aware and act_stats is not None:
                scales[i] = compute_awq_scales(W_main[i:i+1], act_stats, config.group_size)
            else:
                channel_std = np.std(W_main[i])
                scales[i] = (channel_std / 127.0 * np.ones((W_main.shape[1] + config.group_size - 1) // config.group_size)))
    elif config.activation_aware:
        scales = np.ones((W_main.shape[0], (W_main.shape[1] + config.group_size - 1) // config.group_size), 
                         dtype=np.float32)
        for i in range(W_main.shape[0]):
            scales[i] = compute_awq_scales(W_main[i:i+1], activations, config.group_size)
    else:
        channel_std = np.std(W_main, axis=1, keepdims=True)
        n_groups = (W_main.shape[1] + config.group_size - 1) // config.group_size
        scales = np.repeat(channel_std / 127.0, n_groups, axis=1)
    
    zero_points = np.zeros(scales.shape, dtype=np.float32) if config.zero_point else None
    
    W_quant = quantize_group_wise_vectorized(W_main, config.group_size, scales, zero_points)
    
    packed_int4 = pack_int4(W_quant + 8)
    
    if config.rank > 0:
        W_deq = dequantize_group_wise_vectorized(W_quant, scales, zero_points, config.group_size)
        W_residual = W_main - W_deq
        svd_u, svd_v = svd_low_rank_correction(W_residual, config.rank)
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
    """Decompress weights back to FP32."""
    if comp.packed_int4.ndim > 2:
        W_shape = comp.shape
        flat_packed = comp.packed_int4.flatten()
        W_quant_flat = unpack_int4(flat_packed, (flat_packed.size * 2,))[:np.prod(W_shape)]
        W_quant = W_quant_flat.reshape(W_shape)
    else:
        W_quant = unpack_int4(comp.packed_int4, comp.shape)
    
    W_quant = W_quant.astype(np.float32) - 8
    
    scales = comp.scales.astype(np.float32)
    zero_points = comp.zero_points.astype(np.float32) if comp.zero_points is not None else None
    
    if scales.ndim == 1:
        W_rec = dequantize_group_wise_vectorized(
            W_quant, scales, zero_points, comp.group_size
        )
    else:
        W_rec = np.zeros_like(W_quant, dtype=np.float32)
        for i in range(W_quant.shape[0]):
            W_rec[i] = dequantize_group_wise_vectorized(
                W_quant[i:i+1], scales[i], zero_points[i] if zero_points is not None else None, comp.group_size
            )
    
    if comp.svd_u is not None and comp.svd_v is not None:
        W_rec = W_rec + comp.svd_u @ comp.svd_v.T
    
    if comp.protected_channels is not None and comp.protected_indices is not None:
        for i, idx in enumerate(comp.protected_indices):
            if idx < W_rec.shape[0]:
                W_rec[idx] = comp.protected_channels[:, i]
    
    return W_rec.astype(np.float32)


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
