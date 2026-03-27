"""HuggingFace integration for TurboQuant-v3."""

import os
import json
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.utils import (
    is_torch_available,
    logging,
)

from .config import TurboQuantConfig, QuantConfig, QuantizationVersion
from .linear import QuantizedLinear, TurboQuantLinear
from .core import turboquant_v3_compress, CompressedWeights

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.get_logger(__name__)

try:
    from transformers.quantizers.quantizer_base import HfQuantizer
    HF_QUANTIZER_AVAILABLE = True
except ImportError:
    HF_QUANTIZER_AVAILABLE = False


if is_torch_available():
    from transformers.modeling_utils import PreTrainedModel


class TurboQuantizer:
    """TurboQuant-v3 quantizer for HuggingFace models."""
    
    required_dependencies = ["torch", "numpy"]
    sufficient_dependencies = True
    
    def __init__(
        self,
        quantization_config: Optional[TurboQuantConfig] = None,
        **kwargs,
    ):
        self.quantization_config = quantization_config or TurboQuantConfig()
        self.processor_config = None
        self.gguf_config = None
        
        self.group_size = self.quantization_config.group_size
        self.bits = self.quantization_config.bits
        self.version = self.quantization_config.version
        self.zero_point = self.quantization_config.zero_point
        self.activation_aware = self.quantization_config.activation_aware
        self.outlier_keep_ratio = self.quantization_config.outlier_keep_ratio
        self.rank = self.quantization_config.rank
    
    @property
    def is_serializable(self) -> bool:
        return True
    
    @property
    def is_trainable(self) -> bool:
        return True
    
    def update_missing_post_init_defaults(self, **kwargs):
        pass
    
    def validate_environment(self, kwargs_dict: Dict[str, Any]) -> None:
        pass
    
    def validate_model_card(self, model) -> None:
        pass
    
    def create_quantized_method(
        self,
        model: "PreTrainedModel",
        hook_module: nn.Module,
        input_nodes: Any,
        output_nodes: Any,
        quantized_weights_folder: Optional[Union[str, Path]] = None,
    ) -> nn.Module:
        from .hf_modules import TurboQuantizedAttention, TurboQuantizedMLP
        
        if isinstance(hook_module, nn.Linear):
            config = QuantConfig(
                group_size=self.group_size,
                outlier_keep_ratio=self.outlier_keep_ratio,
                rank=self.rank,
                activation_aware=self.activation_aware,
                zero_point=self.zero_point,
            )
            return QuantizedLinear.from_linear(hook_module, config=config)
        
        return hook_module
    
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        quantization_config: Optional[TurboQuantConfig] = None,
        **kwargs,
    ) -> None:
        if quantization_config is not None:
            self.quantization_config = quantization_config
            self.bits = quantization_config.bits
            self.group_size = quantization_config.group_size
            self.version = quantization_config.version
            self.zero_point = quantization_config.zero_point
            self.activation_aware = quantization_config.activation_aware
            self.outlier_keep_ratio = quantization_config.outlier_keep_ratio
            self.rank = quantization_config.rank
    
    def _process_model_after_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ) -> None:
        pass
    
    def inject_quantization_attribues(
        self,
        config: PretrainedConfig,
        quantization_config: Optional[TurboQuantConfig] = None,
    ) -> PretrainedConfig:
        if quantization_config is not None:
            config.quantization_config = quantization_config.to_dict()
        elif hasattr(config, "quantization_config") and config.quantization_config is not None:
            config.quantization_config = TurboQuantConfig.from_dict(
                config.quantization_config
            ).to_dict()
        else:
            config.quantization_config = self.quantization_config.to_dict()
        
        return config
    
    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "TurboQuantizer":
        quantization_config_dict = getattr(config, "quantization_config", None)
        
        if quantization_config_dict is not None:
            quantization_config = TurboQuantConfig.from_dict(quantization_config_dict)
        else:
            quantization_config = TurboQuantConfig()
        
        return cls(quantization_config=quantization_config)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantization_config": self.quantization_config.to_dict(),
        }
    
    def get_quantization_config(self) -> Optional[TurboQuantConfig]:
        return self.quantization_config


if HF_QUANTIZER_AVAILABLE:
    class HfTurboQuantizer(HfQuantizer):
        def __init__(
            self,
            quantization_config: Optional[TurboQuantConfig] = None,
            **kwargs,
        ):
            super().__init__(quantization_config=quantization_config, **kwargs)
            self.quantization_config = quantization_config or TurboQuantConfig()
            self.turboquantizer = TurboQuantizer(quantization_config=self.quantization_config)
        
        def validate_environment(self, kwargs_dict: Dict[str, Any]) -> None:
            super().validate_environment(kwargs_dict)
            self.turboquantizer.validate_environment(kwargs_dict)
        
        def validate_model_card(self, model) -> None:
            super().validate_model_card(model)
            self.turboquantizer.validate_model_card(model)
        
        def create_quantized_method(
            self,
            model: "PreTrainedModel",
            hook_module: nn.Module,
            input_nodes: Any,
            output_nodes: Any,
            quantized_weights_folder: Optional[Union[str, Path]] = None,
        ) -> nn.Module:
            return self.turboquantizer.create_quantized_method(
                model, hook_module, input_nodes, output_nodes, quantized_weights_folder
            )
        
        def _process_model_before_weight_loading(
            self,
            model: "PreTrainedModel",
            **kwargs,
        ) -> None:
            self.turboquantizer._process_model_before_weight_loading(model, **kwargs)
        
        def _process_model_after_weight_loading(
            self,
            model: "PreTrainedModel",
            **kwargs,
        ) -> None:
            self.turboquantizer._process_model_after_weight_loading(model, **kwargs)
        
        def is_serializable(self) -> bool:
            return self.turboquantizer.is_serializable
        
        def is_trainable(self) -> bool:
            return self.turboquantizer.is_trainable


def quantize_model(
    model: "PreTrainedModel",
    quantization_config: Optional[TurboQuantConfig] = None,
    calibration_data: Optional[torch.Tensor] = None,
    inplace: bool = True,
) -> "PreTrainedModel":
    if not inplace:
        model = model.cpu().to(torch.float32).to(str(model.__class__.__name__))
    
    config = quantization_config or TurboQuantConfig()
    quantizer = TurboQuantizer(quantization_config=config)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            activations = None
            if calibration_data is not None and len(calibration_data.shape) >= 2:
                activations = calibration_data
            
            quant_config = QuantConfig(
                group_size=config.group_size,
                outlier_keep_ratio=config.outlier_keep_ratio,
                rank=config.rank,
                activation_aware=config.activation_aware,
                zero_point=config.zero_point,
            )
            
            quantized_layer = QuantizedLinear.from_linear(module, config=quant_config, activations=activations)
            
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            setattr(parent, child_name, quantized_layer)
    
    return model


def save_quantized_model(
    model: "PreTrainedModel",
    save_directory: Union[str, Path],
    quantization_config: Optional[TurboQuantConfig] = None,
    safe_serialization: bool = True,
) -> None:
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(save_directory, safe_serialization=safe_serialization)
    
    if quantization_config is not None:
        config_path = save_directory / "quantization_config.json"
        with open(config_path, "w") as f:
            json.dump(quantization_config.to_dict(), f, indent=2)


def load_quantized_model(
    model_name_or_path: Union[str, Path],
    device_map: Optional[Union[str, Dict[str, Any]]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> "PreTrainedModel":
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    config_path = Path(model_name_or_path) / "quantization_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        quantization_config = TurboQuantConfig.from_dict(config_dict)
    else:
        quantization_config = None
    
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=kwargs.get("trust_remote_code", False),
    )
    
    if quantization_config:
        config.quantization_config = quantization_config.to_dict()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=kwargs.get("trust_remote_code", False),
        **kwargs,
    )
    
    return model
