"""HuggingFace quantized modules for TurboQuant-v3 integration."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

try:
    from transformers.cache_utils import Cache
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    Cache = None

from .linear import QuantizedLinear
from .config import QuantConfig


class TurboQuantizedAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        kv_cache_quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        self.q_proj: Optional[nn.Module] = None
        self.k_proj: Optional[nn.Module] = None
        self.v_proj: Optional[nn.Module] = None
        self.o_proj: Optional[nn.Module] = None
        
        self.dropout = dropout
        self.bias = bias
        self.kv_cache_quant_config = kv_cache_quant_config
    
    def init_weights(self, config: Optional[QuantConfig] = None):
        config = config or QuantConfig()
        
        self.q_proj = QuantizedLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=self.bias,
            config=config,
        )
        
        self.k_proj = QuantizedLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.bias,
            config=config,
        )
        
        self.v_proj = QuantizedLinear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            config=config,
        )
        
        self.o_proj = QuantizedLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            config=config,
        )
    
    @classmethod
    def from_native_module(
        cls,
        module: nn.Module,
        config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "TurboQuantizedAttention":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for this functionality")
        
        from transformers.models.llama.modeling_llama import LlamaAttention
        
        if hasattr(module, "num_heads"):
            num_heads = module.num_heads
        else:
            num_heads = getattr(module.config, "num_attention_heads", 32)
        
        num_kv_heads = getattr(module, "num_key_value_heads", num_heads)
        hidden_size = getattr(module.config, "hidden_size", 4096)
        bias = getattr(module, "q_proj", None) is not None and hasattr(module.q_proj, "bias")
        
        quantized_attn = cls(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=bias,
            **kwargs,
        )
        quantized_attn.init_weights(config)
        
        if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
            quantized_attn.q_proj = QuantizedLinear.from_linear(module.q_proj, config=config)
        if hasattr(module, "k_proj") and isinstance(module.k_proj, nn.Linear):
            quantized_attn.k_proj = QuantizedLinear.from_linear(module.k_proj, config=config)
        if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
            quantized_attn.v_proj = QuantizedLinear.from_linear(module.v_proj, config=config)
        if hasattr(module, "o_proj") and isinstance(module.o_proj, nn.Linear):
            quantized_attn.o_proj = QuantizedLinear.from_linear(module.o_proj, config=config)
        
        return quantized_attn
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        if self.q_proj is None:
            raise RuntimeError("Call init_weights() before using the module")
        
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)
        
        q_states = q_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = k_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_states = v_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.num_kv_groups > 1:
            k_states = k_states.repeat_interleave(self.num_kv_groups, dim=1)
            v_states = v_states.repeat_interleave(self.num_kv_groups, dim=1)
        
        if past_key_value is not None:
            k_states = torch.cat([past_key_value[0], k_states], dim=2)
            v_states = torch.cat([past_key_value[1], v_states], dim=2)
        
        past_key_value = (k_states, v_states) if use_cache else None
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, k_states.size(2)):
                attention_mask = attention_mask[:, None, :, :] + 0.0
        
        attn_weights = torch.matmul(q_states, k_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None if not output_attentions else attn_weights, past_key_value


class TurboQuantizedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        
        self.gate_proj: Optional[nn.Module] = None
        self.up_proj: Optional[nn.Module] = None
        self.down_proj: Optional[nn.Module] = None
    
    def init_weights(self, config: Optional[QuantConfig] = None):
        config = config or QuantConfig()
        
        self.gate_proj = QuantizedLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            config=config,
        )
        
        self.up_proj = QuantizedLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=bias,
            config=config,
        )
        
        self.down_proj = QuantizedLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            config=config,
        )
    
    @classmethod
    def from_native_module(
        cls,
        module: nn.Module,
        config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "TurboQuantizedMLP":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for this functionality")
        
        hidden_size = getattr(module, "hidden_size", 4096)
        intermediate_size = getattr(module, "intermediate_size", 11008)
        hidden_act = getattr(module, "act_fn", "silu")
        bias = hasattr(module, "gate_proj") and hasattr(module.gate_proj, "bias")
        
        quantized_mlp = cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            bias=bias,
            **kwargs,
        )
        quantized_mlp.init_weights(config)
        
        if hasattr(module, "gate_proj") and isinstance(module.gate_proj, nn.Linear):
            quantized_mlp.gate_proj = QuantizedLinear.from_linear(module.gate_proj, config=config)
        if hasattr(module, "up_proj") and isinstance(module.up_proj, nn.Linear):
            quantized_mlp.up_proj = QuantizedLinear.from_linear(module.up_proj, config=config)
        if hasattr(module, "down_proj") and isinstance(module.down_proj, nn.Linear):
            quantized_mlp.down_proj = QuantizedLinear.from_linear(module.down_proj, config=config)
        
        return quantized_mlp
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is None:
            raise RuntimeError("Call init_weights() before using the module")
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if self.hidden_act == "silu":
            hidden = torch.nn.functional.silu(gate) * up
        elif self.hidden_act == "gelu":
            hidden = torch.nn.functional.gelu(gate) * up
        else:
            hidden = gate * up
        
        down = self.down_proj(hidden)
        
        return down


class TurboQuantizedLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, Tuple[int]]):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.weight = None
        self.bias = None
    
    def init_weights(self):
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
    
    @classmethod
    def from_native_module(
        cls,
        module: nn.Module,
        **kwargs,
    ) -> "TurboQuantizedLayerNorm":
        normalized_shape = getattr(module, "normalized_shape", 4096)
        
        quantized_ln = cls(normalized_shape=normalized_shape)
        quantized_ln.init_weights()
        
        if hasattr(module, "weight") and module.weight is not None:
            quantized_ln.weight.data = module.weight.data.clone()
        if hasattr(module, "bias") and module.bias is not None:
            quantized_ln.bias.data = module.bias.data.clone()
        
        return quantized_ln
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias
        )
