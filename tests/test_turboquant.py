import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from turboquant import (
    QuantConfig,
    TurboQuantConfig,
    turboquant_v3_compress,
    turboquant_v3_decompress,
    compute_metrics,
    pack_int4,
    unpack_int4,
    QuantizedLinear,
)


class TestConfig:
    def test_quant_config_defaults(self):
        config = QuantConfig()
        assert config.group_size == 64
        assert config.outlier_keep_ratio == 0.02
        assert config.rank == 8
        assert config.activation_aware is True
        assert config.zero_point is True

    def test_quant_config_validation(self):
        with pytest.raises(ValueError):
            QuantConfig(group_size=0)
        with pytest.raises(ValueError):
            QuantConfig(outlier_keep_ratio=-0.1)
        with pytest.raises(ValueError):
            QuantConfig(outlier_keep_ratio=1.5)

    def test_turbo_quant_config_from_dict(self):
        config_dict = {
            "bits": 4,
            "group_size": 128,
            "version": "gemm",
            "zero_point": True,
            "activation_aware": True,
            "outlier_keep_ratio": 0.02,
            "rank": 8,
        }
        config = TurboQuantConfig.from_dict(config_dict)
        assert config.bits == 4
        assert config.group_size == 128
        assert config.version.value == "gemm"


class TestCoreFunctions:
    def test_pack_unpack_int4(self):
        n = 100
        original = np.random.randint(0, 16, size=n, dtype=np.uint8)
        
        packed = pack_int4(original)
        assert packed.shape[0] == (n + 1) // 2
        
        unpacked = unpack_int4(packed, (n,))
        np.testing.assert_array_equal(original[:n], unpacked[:n])

    def test_turboquant_compress_decompress(self):
        np.random.seed(42)
        W = np.random.normal(0, 0.02, size=(512, 1024)).astype(np.float32)
        
        config = QuantConfig(
            group_size=64,
            outlier_keep_ratio=0.02,
            rank=8,
            activation_aware=True
        )
        
        comp = turboquant_v3_compress(W, config)
        W_rec = turboquant_v3_decompress(comp)
        
        assert W_rec.shape == W.shape
        
        metrics = compute_metrics(W, W_rec)
        assert "mse" in metrics
        assert "max_error" in metrics
        assert "relative_error" in metrics
        assert "psnr_db" in metrics
        
        assert metrics["mse"] < 0.01
        assert metrics["relative_error"] < 0.1

    def test_turboquant_compress_decompress_2d(self):
        np.random.seed(123)
        W = np.random.normal(0, 0.01, size=(256, 512)).astype(np.float32)
        
        config = QuantConfig(group_size=32, rank=4)
        
        comp = turboquant_v3_compress(W, config)
        W_rec = turboquant_v3_decompress(comp)
        
        np.testing.assert_allclose(W, W_rec, rtol=0.1)

    def test_turboquant_without_svd(self):
        np.random.seed(456)
        W = np.random.normal(0, 0.02, size=(128, 256)).astype(np.float32)
        
        config = QuantConfig(group_size=64, rank=0)
        
        comp = turboquant_v3_compress(W, config)
        W_rec = turboquant_v3_decompress(comp)
        
        metrics = compute_metrics(W, W_rec)
        assert metrics["mse"] < 0.05


class TestQuantizedLinear:
    def test_from_linear(self):
        linear = nn.Linear(512, 256, bias=False)
        
        config = QuantConfig(group_size=64, rank=8)
        quantized = QuantizedLinear.from_linear(linear, config=config)
        
        assert isinstance(quantized, QuantizedLinear)
        assert quantized.in_features == 512
        assert quantized.out_features == 256
        assert quantized._is_quantized is True

    def test_from_linear_with_bias(self):
        linear = nn.Linear(256, 128, bias=True)
        
        config = QuantConfig()
        quantized = QuantizedLinear.from_linear(linear, config=config)
        
        assert quantized.bias is not None

    def test_forward(self):
        linear = nn.Linear(512, 256, bias=False)
        np.random.seed(42)
        torch.manual_seed(42)
        
        config = QuantConfig(group_size=64, rank=8)
        quantized = QuantizedLinear.from_linear(linear, config=config)
        
        x = torch.randn(4, 512)
        
        output_quantized = quantized(x)
        
        assert output_quantized.shape == (4, 256)
        assert output_quantized.dtype == torch.float16

    def test_weight_stats(self):
        linear = nn.Linear(512, 256)
        
        config = QuantConfig()
        quantized = QuantizedLinear.from_linear(linear, config=config)
        
        stats = quantized.get_weight_stats()
        
        assert stats["quantized"] is True
        assert "original_size_bytes" in stats
        assert "quantized_size_bytes" in stats
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 1.0


class TestIntegration:
    def test_multiple_layers(self):
        np.random.seed(42)
        torch.manual_seed(42)
        
        layers = [
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
        ]
        
        config = QuantConfig(group_size=32, rank=4)
        
        quantized_layers = []
        for linear in layers:
            q = QuantizedLinear.from_linear(linear, config=config)
            quantized_layers.append(q)
        
        x = torch.randn(2, 512)
        
        for i, q in enumerate(quantized_layers):
            x = q(x)
        
        assert x.shape == (2, 64)

    def test_end_to_end_with_different_configs(self):
        np.random.seed(42)
        
        W = np.random.normal(0, 0.02, size=(128, 256)).astype(np.float32)
        
        configs = [
            QuantConfig(group_size=32, rank=8),
            QuantConfig(group_size=64, rank=4),
            QuantConfig(group_size=128, rank=0),
            QuantConfig(group_size=64, rank=16, activation_aware=False),
        ]
        
        for config in configs:
            comp = turboquant_v3_compress(W, config)
            W_rec = turboquant_v3_decompress(comp)
            metrics = compute_metrics(W, W_rec)
            
            assert metrics["mse"] < 0.1


class TestEdgeCases:
    def test_small_matrix(self):
        W = np.random.normal(0, 0.01, size=(8, 16)).astype(np.float32)
        
        config = QuantConfig(group_size=4)
        comp = turboquant_v3_compress(W, config)
        W_rec = turboquant_v3_decompress(comp)
        
        assert W_rec.shape == W.shape

    def test_large_group_size(self):
        W = np.random.normal(0, 0.01, size=(64, 128)).astype(np.float32)
        
        config = QuantConfig(group_size=128)
        comp = turboquant_v3_compress(W, config)
        W_rec = turboquant_v3_decompress(comp)
        
        assert W_rec.shape == W.shape

    def test_high_outlier_ratio(self):
        W = np.random.normal(0, 0.02, size=(128, 256)).astype(np.float32)
        W[:10, :] = W[:10, :] * 10
        
        config = QuantConfig(outlier_keep_ratio=0.1)
        comp = turboquant_v3_compress(W, config)
        W_rec = turboquant_v3_decompress(comp)
        
        metrics = compute_metrics(W, W_rec)
        assert metrics["relative_error"] < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
