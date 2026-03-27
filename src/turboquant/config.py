"""Configuration classes for TurboQuant-v3."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QuantizationVersion(str, Enum):
    GEMM = "gemm"
    EXLLAMA = "exllama"
    IPEX = "ipex"


@dataclass
class QuantConfig:
    group_size: int = 64
    outlier_keep_ratio: float = 0.02
    rank: int = 8
    activation_aware: bool = True
    zero_point: bool = True
    dtype: str = "float16"

    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if not 0 <= self.outlier_keep_ratio <= 1:
            raise ValueError("outlier_keep_ratio must be between 0 and 1")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")


@dataclass
class TurboQuantConfig:
    bits: int = 4
    group_size: int = 128
    version: QuantizationVersion = QuantizationVersion.GEMM
    zero_point: bool = True
    fuse_max_seq_len: Optional[int] = None
    do_fuse: bool = False
    activation_aware: bool = True
    outlier_keep_ratio: float = 0.02
    rank: int = 8

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError("Only 4-bit and 8-bit quantization supported")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")

    def to_dict(self):
        return {
            "quant_method": "turboquant",
            "bits": self.bits,
            "group_size": self.group_size,
            "version": self.version.value if isinstance(self.version, QuantizationVersion) else self.version,
            "zero_point": self.zero_point,
            "fuse_max_seq_len": self.fuse_max_seq_len,
            "do_fuse": self.do_fuse,
            "activation_aware": self.activation_aware,
            "outlier_keep_ratio": self.outlier_keep_ratio,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, config_dict):
        version = config_dict.get("version", "gemm")
        if isinstance(version, str):
            version = QuantizationVersion(version)
        return cls(
            bits=config_dict.get("bits", 4),
            group_size=config_dict.get("group_size", 128),
            version=version,
            zero_point=config_dict.get("zero_point", True),
            fuse_max_seq_len=config_dict.get("fuse_max_seq_len"),
            do_fuse=config_dict.get("do_fuse", False),
            activation_aware=config_dict.get("activation_aware", True),
            outlier_keep_ratio=config_dict.get("outlier_keep_ratio", 0.02),
            rank=config_dict.get("rank", 8),
        )
