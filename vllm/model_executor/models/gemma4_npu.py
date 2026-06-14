"""Gemma4 model implementation for Davinci NPU."""
import re
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from vllm.model_executor.layers.npu.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from vllm.model_executor.layers.npu.util import (get_default_stream, get_pointer,
                                                  to_npu_dtype, DataType)
from vllm.model_executor.layers.npu.py_npu_ops import (
    split_qkv_layer, gated_gelu_layer,
    qkv_norm_with_weight_layer, qkv_norm_no_weight_layer,
    page_attn_gqa_dim512_layer, page_attn_dim256_gqa_layer,
    rope_standard_layer,
    add_layer, rmsnorm_layer, gelu_pytorch_tanh_layer, mul_scalar_layer,
    mul_layer)

import acl


def _get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def print_tensor(name, x):
    return
    acl.rt.synchronize_stream(get_default_stream())
    shape = x.shape
    x = x.cpu()[..., :8]
    print(name, x, shape)


class Gemma4RMSNorm(nn.Module):
    """Simple RMSNorm without residual handling for Gemma4.
    Gemma4 manages residuals explicitly, unlike Qwen2."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))

    def forward(self, x):
        hidden_size = x.shape[-1]
        assert hidden_size % 16 == 0
        first_dim = x.numel() // hidden_size

        output = torch.empty_like(x)
        rmsnorm_layer(get_pointer(output), get_pointer(self.weight),
                      get_pointer(x),
                      first_dim, hidden_size, self.variance_epsilon,
                      to_npu_dtype(x.dtype), get_default_stream())
        return output


class Gemma4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ReplicatedLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only gelu_pytorch_tanh is supported for Gemma4.")

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        last_dim = gate_up.shape[-1] // 2
        output_shape = list(gate_up.shape)
        output_shape[-1] = last_dim
        # gated_gelu kernel expects [all_gates, all_ups] layout per call.
        # Call per-token (same as C++ engine) to avoid layout mismatch.
        # Each token has gate=[0:last_dim], up=[last_dim:2*last_dim].
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        flat_in = gate_up.reshape(-1, last_dim * 2)
        flat_out = output.reshape(-1, last_dim)
        for t in range(flat_in.shape[0]):
            gated_gelu_layer(get_pointer(flat_out[t]), get_pointer(flat_in[t]),
                             last_dim, to_npu_dtype(gate_up.dtype),
                             get_default_stream())
        x, _ = self.down_proj(output)
        return x


class Gemma4Attention(nn.Module):

    def __init__(self,
                 config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int = 4096 * 32,
                 use_k_eq_v: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 attn_logits_soft_cap: Optional[float] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.use_k_eq_v = use_k_eq_v
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Gemma4 uses scale=1.0 because Q/K norms handle scaling.
        self.scaling = 1.0

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = ReplicatedLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Q/K norms with learnable weights, V norm without
        self.q_norm_weight = nn.Parameter(torch.empty(self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.empty(self.head_dim))

        # Determine layer type and sliding window
        self.is_sliding = False
        self.is_full_attention = True
        self.sliding_window = None
        self.partial_rotary_factor = 1.0
        self.rope_theta = 10000.0

        if hasattr(config, 'layer_types'):
            m = re.search(r'layers\.(\d+)', prefix)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx < len(config.layer_types):
                    layer_type = config.layer_types[layer_idx]
                    self.is_sliding = layer_type == "sliding_attention"
                    self.is_full_attention = layer_type == "full_attention"

        if self.is_sliding:
            self.sliding_window = config.sliding_window
            if hasattr(config, 'rope_local_base_freq'):
                self.rope_theta = config.rope_local_base_freq

        if self.is_full_attention:
            if hasattr(config, 'rope_parameters') and isinstance(config.rope_parameters, dict):
                layer_type_key = "full_attention"
                if layer_type_key in config.rope_parameters:
                    rp = config.rope_parameters[layer_type_key]
                    self.partial_rotary_factor = rp.get('partial_rotary_factor', 1.0)
                    self.rope_theta = rp.get('rope_theta', self.rope_theta)

        # KV sharing: layers in the last num_kv_shared_layers share KV cache
        self.is_kv_shared_layer = False
        num_kv_shared_layers = getattr(config, "num_kv_shared_layers", 0)
        if num_kv_shared_layers > 0:
            m = re.search(r'layers\.(\d+)', prefix)
            if m:
                layer_idx = int(m.group(1))
                first_kv_shared = config.num_hidden_layers - num_kv_shared_layers
                if layer_idx >= first_kv_shared:
                    self.is_kv_shared_layer = True

        # Build freqs_cis table: [max_position_embeddings, head_dim]
        # partial_rotary_factor < 1.0: non-rotated dims get cos=1, sin=0
        self.max_position_embeddings = max_position_embeddings
        self.register_buffer(
            "freqs_cis",
            self._build_freqs_cis(max_position_embeddings, self.head_dim,
                                  self.rope_theta, self.partial_rotary_factor),
            persistent=False,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        # Pre-allocated page table buffer: max entries = max_pos / block_size rounded up
        kernel_n_tile = 64  # matches both dim256 and dim512 kernel n_tile
        max_page_entries = (max_position_embeddings + kernel_n_tile - 1) // kernel_n_tile
        self._page_table_buf = torch.empty(max_page_entries, dtype=torch.long, device="npu")

    def _build_freqs_cis(self, max_pos, head_dim, theta, partial_rotary_factor=1.0):
        # Build freqs_cis table using numpy, then create tensor on NPU directly.
        # For partial RoPE (partial_rotary_factor < 1.0), non-rotated dimensions
        # are set to cos=1, sin=0 (identity) so a single npu_rope_standard_layer
        # call produces partial rotation (matching C++ engine approach).
        import numpy as np
        rotary_dim = int(head_dim * partial_rotary_factor)
        rotary_dim = (rotary_dim // 2) * 2  # must be even
        num_freqs = rotary_dim // 2

        freqs = 1.0 / (theta ** (np.arange(0, 2 * num_freqs, 2, dtype=np.float64) / head_dim))
        positions = np.arange(max_pos, dtype=np.float64)
        freqs = np.outer(positions, freqs)
        freqs_cos = np.cos(freqs).astype(np.float32)
        freqs_sin = np.sin(freqs).astype(np.float32)

        # Build table: cos/sin interleaved. Rotated dims get actual values,
        # non-rotated dims get cos=1, sin=0 (identity rotation)
        result_np = np.zeros((max_pos, head_dim), dtype=np.float32)
        result_np[:, 0::2] = 1.0  # cos=1 for all positions (rotated dims will be overwritten)
        # sin=0 already set by np.zeros for all positions
        # Place cos at even indices, sin at odd indices for rotated pairs
        for f in range(num_freqs):
            result_np[:, 2*f] = freqs_cos[:, f]
            result_np[:, 2*f+1] = freqs_sin[:, f]
        # Return CPU tensor; model.npu() will move it later.
        # Moving to NPU here corrupts the buffer because register_buffer
        # stores it, then model.npu() tries to move it again.
        return torch.from_numpy(result_np.copy())

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        qkv_last_dim = qkv.shape[-1]
        token_num = qkv.reshape(-1, qkv_last_dim).shape[0]
        q = torch.empty(qkv.shape[:-1] + (self.q_size,), dtype=qkv.dtype, device="npu")
        k = torch.empty(qkv.shape[:-1] + (self.kv_size,), dtype=qkv.dtype, device="npu")
        v = torch.empty(qkv.shape[:-1] + (self.kv_size,), dtype=qkv.dtype, device="npu")

        split_qkv_layer(get_pointer(q), get_pointer(k),
                        get_pointer(v), get_pointer(qkv),
                        token_num, self.q_size,
                        self.kv_size, self.kv_size,
                        to_npu_dtype(hidden_states.dtype), get_default_stream())

        # Reshape for per-head processing
        q_reshaped = q.reshape(token_num, self.num_heads, self.head_dim)
        k_reshaped = k.reshape(token_num, self.num_kv_heads, self.head_dim)
        v_reshaped = v.reshape(token_num, self.num_kv_heads, self.head_dim)

        # Apply Q/K/V per-head norms
        # Gemma4 uses shared weight [head_dim] across all heads.
        # The kernel's shared_weight=True flag handles this internally.
        q_normed = torch.empty_like(q_reshaped)
        k_normed = torch.empty_like(k_reshaped)

        qkv_norm_with_weight_layer(
            get_pointer(q_normed), get_pointer(q_reshaped),
            get_pointer(self.q_norm_weight), token_num, self.num_heads,
            self.head_dim, self.config.rms_norm_eps,
            True,  # shared_weight
            to_npu_dtype(hidden_states.dtype), get_default_stream())

        if not self.is_kv_shared_layer:
            qkv_norm_with_weight_layer(
                get_pointer(k_normed), get_pointer(k_reshaped),
                get_pointer(self.k_norm_weight), token_num, self.num_kv_heads,
                self.head_dim, self.config.rms_norm_eps,
                True,  # shared_weight
                to_npu_dtype(hidden_states.dtype), get_default_stream())

            v_normed = torch.empty_like(v_reshaped)
            qkv_norm_no_weight_layer(
                get_pointer(v_normed), get_pointer(v_reshaped),
                token_num, self.num_kv_heads, self.head_dim,
                self.config.rms_norm_eps,
                to_npu_dtype(hidden_states.dtype), get_default_stream())

        # Apply RoPE per batch item — each sequence may have a different
        # starting position (e.g. decode with multiple sequences at different
        # stages, or prefill where each sequence resets to position 0).
        q_flat = q_normed.reshape(token_num, self.q_size)
        q_roped_flat = torch.empty(token_num, self.q_size, dtype=q_flat.dtype, device="npu")

        if not self.is_kv_shared_layer:
            k_flat = k_normed.reshape(token_num, self.kv_size)
            k_roped_flat = torch.empty(token_num, self.kv_size, dtype=k_flat.dtype, device="npu")

        batch_size = len(attn_metadata.seq_lens)
        token_offset = 0
        for batch_i in range(batch_size):
            seq_len = attn_metadata.seq_lens[batch_i]
            seq_start_pos = attn_metadata.start_positions[batch_i]

            # Q RoPE for this batch item
            rope_standard_layer(
                get_pointer(q_roped_flat[token_offset:token_offset + seq_len, ...]),
                get_pointer(self.freqs_cis),
                get_pointer(q_flat[token_offset:token_offset + seq_len, ...]),
                seq_start_pos, seq_len, self.num_heads, self.q_size,
                to_npu_dtype(hidden_states.dtype), get_default_stream())

            # K RoPE for this batch item
            if not self.is_kv_shared_layer:
                rope_standard_layer(
                    get_pointer(k_roped_flat[token_offset:token_offset + seq_len, ...]),
                    get_pointer(self.freqs_cis),
                    get_pointer(k_flat[token_offset:token_offset + seq_len, ...]),
                    seq_start_pos, seq_len, self.num_kv_heads, self.kv_size,
                    to_npu_dtype(hidden_states.dtype), get_default_stream())

            token_offset += seq_len

        if not self.is_kv_shared_layer:
            q = q_roped_flat
            k_out = k_roped_flat
            v_out = v_normed.reshape(token_num, self.kv_size)
            self._should_write_kv = True
        else:
            q = q_roped_flat
            # KV-shared layers: K/V come from the shared cache (filled by L13/L14).
            # K/V are computed but not written to cache and don't need RoPE.
            k_out = k_reshaped.reshape(token_num, self.kv_size)
            v_out = v_reshaped.reshape(token_num, self.kv_size)
            self._should_write_kv = False

        # Page attention with KV cache update.
        # dim256 kernel: n_tile=64, each page_table entry = 1 vllm block (64 tokens)
        # dim512 kernel: n_tile=64, each page_table entry = 1 vllm block (64 tokens)
        # Both kernels access K/V at: key + page_table[ni] * n_tile * kv_dim
        # Since n_tile = block_size = 64, page_table[ni] = physical block index directly.
        block_size = self.attn.block_size
        kernel_n_tile = 64  # matches both dim256 and dim512 kernel n_tile

        attn_output = torch.empty(q.shape, dtype=q.dtype, device="npu")

        flat_seq_offset = 0
        batch_size = len(attn_metadata.seq_lens)
        for batch_i in range(batch_size):
            curr_seq_len = attn_metadata.seq_lens[batch_i]
            curr_offset = attn_metadata.offsets[batch_i]
            remain_seq_len = curr_seq_len
            _curr_block_table_npu, curr_block_table_host = attn_metadata.block_tables[batch_i]
            offset_in_block = curr_offset % block_size
            block_table_i = curr_offset // block_size
            curr_pos = curr_offset + curr_seq_len


            curr_seq_offset = 0
            while remain_seq_len > 0:
                copy_seq_len = min(block_size - offset_in_block, remain_seq_len)
                copy_bytes = copy_seq_len * kv_cache.dtype.itemsize * self.num_kv_heads * self.head_dim

                k_cache_base = kv_cache[0, curr_block_table_host[block_table_i], offset_in_block, 0]
                v_cache_base = kv_cache[1, curr_block_table_host[block_table_i], offset_in_block, 0]

                if self._should_write_kv:
                    ret = acl.rt.memcpy_async(k_cache_base.data_ptr(), copy_bytes,
                                              k_out[flat_seq_offset + curr_seq_offset].data_ptr(),
                                              copy_bytes, 3, get_default_stream())
                    assert ret == 0, "failed to copy k cache"
                    ret = acl.rt.memcpy_async(v_cache_base.data_ptr(), copy_bytes,
                                              v_out[flat_seq_offset + curr_seq_offset].data_ptr(),
                                              copy_bytes, 3, get_default_stream())
                    assert ret == 0, "failed to copy v cache"

                remain_seq_len -= copy_seq_len
                block_table_i += 1
                offset_in_block = (offset_in_block + copy_seq_len) % block_size
                curr_seq_offset += copy_seq_len

            # Direct mapping: n_tile=64 = block_size, 1 page_table entry per vllm block
            num_kernel_entries = (curr_pos + kernel_n_tile - 1) // kernel_n_tile
            page_table_list = list(curr_block_table_host[:num_kernel_entries])
            while len(page_table_list) < num_kernel_entries:
                page_table_list.append(curr_block_table_host[-1])
            # Use pre-allocated buffer + async H2D copy to avoid torch.tensor JIT trigger
            page_table_cpu = torch.tensor(page_table_list, dtype=torch.long, device="cpu")
            ret = acl.rt.memcpy_async(self._page_table_buf.data_ptr(),
                                      num_kernel_entries * 8,
                                      page_table_cpu.data_ptr(),
                                      num_kernel_entries * 8,
                                      1, get_default_stream())
            assert ret == 0, "failed to copy page table"
            page_table_npu = self._page_table_buf[:num_kernel_entries]


            qk_scale = 1.0

            if self.is_sliding:
                page_attn_dim256_gqa_layer(
                    get_pointer(attn_output[flat_seq_offset, ...]),
                    get_pointer(page_table_npu),
                    get_pointer(q[flat_seq_offset:flat_seq_offset + curr_seq_len, ...]),
                    get_pointer(kv_cache[0, ...]),
                    get_pointer(kv_cache[1, ...]),
                    curr_seq_len, curr_pos, curr_offset,
                    self.num_heads, self.num_kv_heads,
                    self.sliding_window, qk_scale,
                    to_npu_dtype(q.dtype), get_default_stream())
            else:
                group_size = self.num_heads // self.num_kv_heads
                page_attn_gqa_dim512_layer(
                    get_pointer(attn_output[flat_seq_offset, ...]),
                    get_pointer(page_table_npu),
                    get_pointer(q[flat_seq_offset:flat_seq_offset + curr_seq_len, ...]),
                    get_pointer(kv_cache[0, ...]),
                    get_pointer(kv_cache[1, ...]),
                    curr_seq_len, curr_pos, curr_offset,
                    group_size, self.num_kv_heads,
                    qk_scale,
                    to_npu_dtype(q.dtype), get_default_stream())

            flat_seq_offset += curr_seq_len

        output, _ = self.o_proj(attn_output)
        return output


class Gemma4DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        max_model_len: int = 4096,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        layer_idx = 0
        m = re.search(r'layers\.(\d+)', prefix)
        if m:
            layer_idx = int(m.group(1))
        self.layer_idx = layer_idx

        layer_type = config.layer_types[layer_idx]
        self.is_full_attention = layer_type == "full_attention"
        if self.is_full_attention:
            head_dim = getattr(config, "global_head_dim", config.head_dim)
        else:
            head_dim = config.head_dim

        use_k_eq_v = self.is_full_attention and getattr(config, "attention_k_eq_v", False)

        if use_k_eq_v:
            num_kv_heads = getattr(config, "num_global_key_value_heads", config.num_key_value_heads)
        else:
            num_kv_heads = config.num_key_value_heads

        # Limit freqs_cis table to actual max model length instead of full config max
        max_pos = min(config.max_position_embeddings, max_model_len)

        self.self_attn = Gemma4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_pos,
            use_k_eq_v=use_k_eq_v,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=getattr(config, "attn_logit_softcapping", None),
            prefix=f"{prefix}.self_attn",
        )

        # Double-wide MLP for KV-shared layers
        first_kv_shared_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        is_kv_shared = layer_idx >= first_kv_shared_idx > 0
        use_double_wide = getattr(config, "use_double_wide_mlp", False) and is_kv_shared
        layer_intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)

        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=layer_intermediate_size,
            hidden_act=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Layer norms (pure RMSNorm, no residual fusion)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # PLE (Per-Layer Embedding) components
        hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        if hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = ReplicatedLinear(
                self.hidden_size,
                hidden_size_per_layer_input,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_input_gate",
            )
            self.per_layer_projection = ReplicatedLinear(
                hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_projection",
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Layer scalar (loaded from checkpoint) — applies to ALL text layers
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        per_layer_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gemma4 residual pattern:
        # 1. input_norm(x) -> attn -> post_attn_norm -> ADD residual
        # 2. pre_ff_norm -> mlp -> post_ff_norm -> ADD residual
        # 3. PLE: gate(hidden) * per_layer_input -> projection -> norm -> ADD
        # 4. layer_scalar multiplication
        if residual is None:
            residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        # Add first residual
        new_residual = torch.empty_like(hidden_states)
        add_layer(get_pointer(new_residual),
                  get_pointer(hidden_states), get_pointer(residual),
                  hidden_states.numel(), to_npu_dtype(hidden_states.dtype),
                  get_default_stream())
        residual = new_residual
        hidden_states = residual

        # MLP block
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        # Add second residual
        new_residual = torch.empty_like(hidden_states)
        add_layer(get_pointer(new_residual),
                  get_pointer(hidden_states), get_pointer(residual),
                  hidden_states.numel(), to_npu_dtype(hidden_states.dtype),
                  get_default_stream())
        residual = new_residual
        hidden_states = residual

        # PLE (Per-Layer Embedding)
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate, _ = self.per_layer_input_gate(hidden_states)
            # Apply gelu_pytorch_tanh activation
            gated = torch.empty_like(gate)
            gelu_pytorch_tanh_layer(get_pointer(gated), get_pointer(gate),
                                     gate.numel(),
                                     to_npu_dtype(gate.dtype),
                                     get_default_stream())
            # Element-wise multiply: gated * per_layer_input
            ple_mul = torch.empty_like(gated)
            mul_layer(get_pointer(ple_mul), get_pointer(gated),
                      get_pointer(per_layer_input),
                      gated.numel(), to_npu_dtype(gated.dtype),
                      get_default_stream())

            ple_proj, _ = self.per_layer_projection(ple_mul)
            ple_out = self.post_per_layer_input_norm(ple_proj)
            # Add PLE contribution
            new_hidden = torch.empty_like(hidden_states)
            add_layer(get_pointer(new_hidden),
                      get_pointer(hidden_states), get_pointer(ple_out),
                      hidden_states.numel(), to_npu_dtype(hidden_states.dtype),
                      get_default_stream())
            hidden_states = new_hidden

        # Layer scalar multiplication
        new_hidden = torch.empty_like(hidden_states)
        mul_scalar_layer(get_pointer(new_hidden), get_pointer(hidden_states),
                         hidden_states.numel(),
                         float(self.layer_scalar),
                         to_npu_dtype(hidden_states.dtype),
                         get_default_stream())
        hidden_states = new_hidden

        return hidden_states, None


class Gemma4Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        raw_config = vllm_config.model_config.hf_config
        config = _get_text_config(raw_config)
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # PLE components at model level
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input > 0:
            total_ple_dim = self.hidden_size_per_layer_input * config.num_hidden_layers
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                getattr(config, "vocab_size_per_layer_input", config.vocab_size),
                total_ple_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens_per_layer",
            )
            self.register_buffer(
                "embed_scale_per_layer",
                torch.tensor(self.hidden_size_per_layer_input ** 0.5),
                persistent=False,
            )
            self.per_layer_model_projection = ReplicatedLinear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input, eps=config.rms_norm_eps)
            self.register_buffer(
                "per_layer_input_scale",
                torch.rsqrt(torch.tensor(2.0)),
                persistent=False,
            )
            self.register_buffer(
                "per_layer_projection_scale",
                torch.tensor(config.hidden_size ** -0.5),
                persistent=False,
            )
        else:
            self.embed_tokens_per_layer = None
            self.embed_scale_per_layer = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None

        max_model_len = vllm_config.model_config.max_model_len

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix,
                                              max_model_len=max_model_len),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Embedding scale = sqrt(hidden_size)
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size ** 0.5),
            persistent=False,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        out = torch.empty_like(embeds)
        mul_scalar_layer(get_pointer(out), get_pointer(embeds),
                         embeds.numel(), float(self.normalizer),
                         to_npu_dtype(embeds.dtype), get_default_stream())
        return out

    def get_per_layer_inputs(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.embed_tokens_per_layer is None:
            return None

        per_layer_inputs_mask = torch.logical_and(
            input_ids >= 0,
            input_ids < getattr(self.config, "vocab_size_per_layer_input", self.config.vocab_size),
        )
        per_layer_inputs_tokens = torch.where(
            per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))

        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)
        per_layer_embeds = per_layer_embeds * self.embed_scale_per_layer

        per_layer_embeds = per_layer_embeds.reshape(
            -1,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return per_layer_embeds

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.per_layer_model_projection is None:
            return None

        # Flatten to 2D for linear layer: [batch, seq, hidden] -> [batch*seq, hidden]
        num_tokens = inputs_embeds.shape[0] * inputs_embeds.shape[1] if inputs_embeds.ndim == 3 else inputs_embeds.shape[0]
        flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        per_layer_projection, _ = self.per_layer_model_projection(flat_embeds)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale

        per_layer_projection = per_layer_projection.reshape(
            num_tokens,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        # Combine: (projection + per_layer_inputs) * 1/sqrt(2)
        combined = per_layer_projection + per_layer_inputs
        combined = combined * self.per_layer_input_scale
        return combined

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_inputs)
            else:
                hidden_states = self.get_input_embeddings(input_ids)
                per_layer_embeds = self.get_per_layer_inputs(input_ids)
                per_layer_inputs = self.project_per_layer_inputs(
                    hidden_states, per_layer_embeds)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            per_layer_inputs = intermediate_tensors.get("per_layer_inputs")

        kv_shared_sliding_src = 13  # L13 is the KV source for shared sliding layers
        kv_shared_full_src = 14     # L14 is the KV source for shared full_attention layers

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # Extract per-layer input for this specific layer
            layer_per_input = None
            if per_layer_inputs is not None:
                layer_per_input = per_layer_inputs[:, i, :].contiguous()  # [tokens, layer, dim] -> [tokens, dim]

            # KV cache sharing: shared layers read from L13/L14's cache
            if layer.self_attn.is_kv_shared_layer:
                if layer.self_attn.is_sliding:
                    cache_src = kv_shared_sliding_src
                else:
                    cache_src = kv_shared_full_src
                local_src = cache_src - self.start_layer
                if 0 <= local_src < len(kv_caches):
                    layer_kv_cache = kv_caches[local_src]
                else:
                    layer_kv_cache = kv_caches[i - self.start_layer]
            else:
                layer_kv_cache = kv_caches[i - self.start_layer]

            hidden_states, residual = layer(
                positions,
                hidden_states,
                layer_kv_cache,
                attn_metadata,
                residual,
                per_layer_input=layer_per_input,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
                "per_layer_inputs": per_layer_inputs,
            })

        # Final norm
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            # Apply norm to hidden_states (Gemma4 uses hidden_states directly)
            hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        use_k_eq_v = getattr(self.config, "attention_k_eq_v", False)
        k_eq_v_layer_indices = set()
        if use_k_eq_v:
            for idx, lt in enumerate(self.config.layer_types):
                if lt == "full_attention":
                    k_eq_v_layer_indices.add(idx)

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # Also include buffers for layer_scalar loading
        params_dict.update(dict(self.named_buffers()))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "freqs_cis" in name:
                continue
            orig_name = name
            name = name.replace("language_model.", "")

            # k_eq_v: duplicate k_proj as v_proj
            if "self_attn.k_proj" in name and k_eq_v_layer_indices:
                m = re.search(r'layers\.(\d+)\.', name)
                if m and int(m.group(1)) in k_eq_v_layer_indices:
                    v_name = name.replace("k_proj", "v_proj")

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                # k_eq_v: also load k_proj as v_proj
                if (weight_name == "k_proj" and
                        "self_attn.k_proj" in name and k_eq_v_layer_indices):
                    m = re.search(r'layers\.(\d+)\.', name)
                    if m and int(m.group(1)) in k_eq_v_layer_indices:
                        v_mapped = mapped_name.replace("k_proj", "v_proj")
                        if v_mapped in params_dict:
                            weight_loader(params_dict[v_mapped], loaded_weight,
                                          "v" if shard_id == "k" else shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remap checkpoint q_norm.weight -> param q_norm_weight
                # and k_norm.weight -> k_norm_weight
                for suffix, replacement in [(".q_norm.weight", ".q_norm_weight"),
                                            (".k_norm.weight", ".k_norm_weight")]:
                    if name.endswith(suffix):
                        name = name.replace(suffix, replacement)
                        break
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                # Handle buffers (layer_scalar, etc.)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


class Gemma4ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        raw_config = vllm_config.model_config.hf_config
        config = _get_text_config(raw_config)
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Gemma4Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
            self.lm_head.quant_method = self.model.embed_tokens.linear_method
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(
                                              prefix, "lm_head"))

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Skip multimodal and tied weights
        _skip_prefixes = {"audio_tower.", "vision_tower.", "embed_audio.",
                          "embed_vision.", "mm_", "multi_modal", "processor."}
        if self.config.tie_word_embeddings:
            _skip_prefixes.add("lm_head.")

        def _weight_iterator():
            use_k_eq_v = getattr(self.config, "attention_k_eq_v", False)
            k_eq_v_layer_indices = set()
            if use_k_eq_v:
                for idx, lt in enumerate(self.config.layer_types):
                    if lt == "full_attention":
                        k_eq_v_layer_indices.add(idx)

            for name, weight in weights:
                name = name.replace("language_model.", "")
                # Skip multimodal weights
                if any(name.startswith(p) or p in name for p in _skip_prefixes):
                    continue
                # k_eq_v: duplicate k_proj as v_proj
                if "self_attn.k_proj" in name and k_eq_v_layer_indices:
                    m = re.search(r'layers\.(\d+)\.', name)
                    if m and int(m.group(1)) in k_eq_v_layer_indices:
                        yield name, weight
                        yield name.replace("k_proj", "v_proj"), weight.clone()
                        continue
                yield name, weight

        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=list(_skip_prefixes) + [
                                       "language_model.audio_tower.",
                                       "language_model.vision_tower."])
        loader.load_weights(_weight_iterator())
