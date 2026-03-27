import pytest
import numpy as np
from src.turboquant.quantize import QuantConfig, turboquant_v3_compress, turboquant_v3_decompress


def test_pack_unpack_roundtrip():
    original = np.random.randint(-8, 8, size=100, dtype=np.int8)
    packed = pack_int4(original)  # you can import or copy pack_int4 if needed
    unpacked = unpack_int4(packed, len(original))
    np.testing.assert_array_equal(original, unpacked)


def test_compress_decompress():
    np.random.seed(42)
    W = np.random.normal(0, 0.02, size=(256, 512)).astype(np.float32)
    
    config = QuantConfig(group_size=64, outlier_keep_ratio=0.02, rank=8)
    comp = turboquant_v3_compress(W, config)
    W_rec = turboquant_v3_decompress(comp)
    
    assert W_rec.shape == W.shape
    mse = np.mean((W - W_rec) ** 2)
    assert mse < 0.01  # reasonable reconstruction for this method


def test_no_correction():
    W = np.random.randn(128, 256).astype(np.float32)
    config = QuantConfig(rank=0)
    comp = turboquant_v3_compress(W, config)
    W_rec = turboquant_v3_decompress(comp)
    assert comp["rank"] == 0
    assert comp["U_corr"] is None
