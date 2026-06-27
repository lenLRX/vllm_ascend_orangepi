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
import numpy as np

# Lazy imports for GGUF Q4_0 support
_gguf_q4_0_available = False
try:
    from vllm.model_executor.layers.npu.py_npu_ops import (
        matmul_gguf_q4_0_layer, convert_gguf_q4_0_qweight)
    _gguf_q4_0_available = True
except Exception:
    pass

# Q6_K constants (super-block size for K-quant formats)
_QK_K = 256
_Q6_K_BLOCK_SIZE = 210  # sizeof(block_q6_K) = ql[128] + qh[64] + scales[16] + d(2)

# GGML quantization type enum values
_GGML_TYPE_F16 = 1
_GGML_TYPE_Q4_0 = 2
_GGML_TYPE_Q6_K = 14

# Map GGUF single-projection names to NPU stacked-layer names
_STACKED_REMAP = {"q_proj": "qkv_proj", "k_proj": "qkv_proj",
                  "v_proj": "qkv_proj",
                  "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}
from vllm.model_executor.layers.npu.py_npu_ops import (
    split_qkv_layer, gated_gelu_layer,
    qkv_norm_with_weight_layer, qkv_norm_no_weight_layer,
    page_attn_gqa_dim512_layer, page_attn_dim256_gqa_layer,
    rope_standard_layer,
    add_layer, rmsnorm_layer, gelu_pytorch_tanh_layer, mul_scalar_layer,
    mul_layer, ple_slice_layer)

import acl


def _f32_to_fp16_bits(val):
    """Convert float32 scalar to fp16 bit pattern (uint16)."""
    return np.array([val], dtype=np.float16).view(np.uint16)[0]


def _dump_npu_tensor(tensor, filepath):
    """Save an NPU tensor as .npy file for comparison."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cpu = tensor.cpu().float().numpy()
    np.save(filepath, cpu)


def _quantize_f32_to_q4_0(W_f32):
    """Quantize float32 weight [N, K] to Q4_0 GGUF interleaved format.

    Returns a flat 1D uint8 array of N * K // 32 * 18 bytes.
    """
    N, K = W_f32.shape
    assert K % 32 == 0
    k_blocks = K // 32
    total_blocks = N * k_blocks
    gguf = np.zeros(total_blocks * 18, dtype=np.uint8)
    for n_i in range(N):
        for kb in range(k_blocks):
            block = n_i * k_blocks + kb
            k_start = kb * 32
            w_block = W_f32[n_i, k_start:k_start + 32]
            max_abs = np.max(np.abs(w_block))
            if max_abs < 1e-8:
                d_fp16 = 0
                nibbles = np.zeros(32, dtype=np.uint8)
            else:
                d = max_abs / 7.0
                d_fp16 = _f32_to_fp16_bits(d)
                q_vals = np.clip(np.round(w_block / d), -8, 7).astype(np.int32) + 8
                nibbles = q_vals.astype(np.uint8)
            offset = block * 18
            gguf[offset] = d_fp16 & 0xFF
            gguf[offset + 1] = (d_fp16 >> 8) & 0xFF
            for j in range(16):
                gguf[offset + 2 + j] = nibbles[j] | (nibbles[j + 16] << 4)
    return gguf


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
        self.layer_idx = 0

        if hasattr(config, 'layer_types'):
            m = re.search(r'layers\.(\d+)', prefix)
            if m:
                self.layer_idx = int(m.group(1))
                if self.layer_idx < len(config.layer_types):
                    layer_type = config.layer_types[self.layer_idx]
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
            first_kv_shared = config.num_hidden_layers - num_kv_shared_layers
            if self.layer_idx >= first_kv_shared:
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
        # Retain per-sequence page tables until all kernels have been launched,
        # otherwise an async kernel may read freed/reused memory.
        page_table_tensors = []
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
            page_table_npu = torch.tensor(page_table_list, dtype=torch.long, device="npu")
            page_table_tensors.append(page_table_npu)

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

        if page_table_tensors:
            torch.npu.synchronize()

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

        # Layer scalar (loaded from checkpoint) — applies to ALL text layers.
        # Data-dependent (a learned per-layer weight) so it must stay a buffer,
        # but its value is fixed after load and the forward reads it only via the
        # cached Python float _layer_scalar_f (set in Gemma4Model.load_weights),
        # never from the NPU buffer.  So create the buffer with torch.empty, NOT
        # torch.ones — under the NPU device context torch.ones JIT-compiles a
        # te_OnesLike TBE kernel — and default the cached float to 1.0 (a no-op
        # scale if no checkpoint loads it).  load_weights also skips filling the
        # buffer (avoids a te_Fill); its device value is intentionally unused.
        self.register_buffer("layer_scalar", torch.empty(1))
        self._layer_scalar_f = 1.0

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

        # Layer scalar multiplication.  _layer_scalar_f is a Python float set at
        # load time (see Gemma4Model.load_weights; defaults to 1.0 = no-op scale
        # if no checkpoint), so the forward never reads the NPU buffer / does an
        # .item().  The guard is a defensive fallback that must NOT read the
        # buffer (it is uninitialized torch.empty — see __init__).
        if self._layer_scalar_f is None:
            self._layer_scalar_f = 1.0
        new_hidden = torch.empty_like(hidden_states)
        mul_scalar_layer(get_pointer(new_hidden), get_pointer(hidden_states),
                         hidden_states.numel(),
                         self._layer_scalar_f,
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
            self.embed_scale_per_layer = float(self.hidden_size_per_layer_input ** 0.5)
            self.per_layer_model_projection = ReplicatedLinear(
                config.hidden_size,
                total_ple_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input, eps=config.rms_norm_eps)
            self.per_layer_input_scale = float(2.0 ** -0.5)
            self.per_layer_projection_scale = float(config.hidden_size ** -0.5)
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

        # Embedding scale = sqrt(hidden_size). Pure config constant, not data
        # dependent — keep as a plain Python float so the CCE scalar arg needs
        # no float(tensor) D2H sync per call.
        self.normalizer = float(config.hidden_size ** 0.5)

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

        # Compute mask on CPU to avoid torch.logical_and/torch.where TBE JIT triggers
        vocab_size_per_layer = getattr(self.config, "vocab_size_per_layer_input",
                                       self.config.vocab_size)
        ids_cpu = input_ids.cpu()
        mask_cpu = (ids_cpu >= 0) & (ids_cpu < vocab_size_per_layer)
        tokens_cpu = torch.where(mask_cpu, ids_cpu, torch.zeros_like(ids_cpu))
        per_layer_inputs_tokens = torch.empty_like(input_ids)
        per_layer_inputs_tokens.copy_(tokens_cpu)

        per_layer_embeds = self.embed_tokens_per_layer(per_layer_inputs_tokens)
        # Replace scalar mul with CCE kernel to avoid TBE JIT trigger
        _ple_out = torch.empty_like(per_layer_embeds)
        mul_scalar_layer(get_pointer(_ple_out), get_pointer(per_layer_embeds),
                         per_layer_embeds.numel(), float(self.embed_scale_per_layer),
                         to_npu_dtype(per_layer_embeds.dtype), get_default_stream())
        per_layer_embeds = _ple_out

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
        # Scale via CCE kernel to avoid the aten.mul TBE JIT trigger.
        # MUST be in-place (out ptr == in ptr).  Writing into a separate
        # torch.empty_like()/padded output here garbles decode (the kernel's
        # values are correct to ~1 ULP, but a fresh output allocation at this
        # site corrupts downstream memory); in-place into the matmul output —
        # which the model already owns and consumes next — is clean and safe
        # for an elementwise scalar mul.  Do not "refactor" to a temp buffer.
        mul_scalar_layer(get_pointer(per_layer_projection), get_pointer(per_layer_projection),
                         per_layer_projection.numel(), float(self.per_layer_projection_scale),
                         to_npu_dtype(per_layer_projection.dtype), get_default_stream())

        per_layer_projection = per_layer_projection.reshape(
            num_tokens,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        # Combine: (projection + per_layer_inputs) * 1/sqrt(2)
        # Both tensors are in standard format after norm/embedding, so CCE kernels work.
        combined = torch.empty_like(per_layer_projection)
        add_layer(get_pointer(combined),
                  get_pointer(per_layer_projection), get_pointer(per_layer_inputs),
                  per_layer_projection.numel(),
                  to_npu_dtype(per_layer_projection.dtype), get_default_stream())
        result = torch.empty_like(combined)
        mul_scalar_layer(get_pointer(result), get_pointer(combined),
                         combined.numel(), float(self.per_layer_input_scale),
                         to_npu_dtype(combined.dtype), get_default_stream())
        return result

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
                # Dump initial embeddings (layer -1) if enabled
                if getattr(self, '_dump_tensors', False):
                    _dump_npu_tensor(hidden_states, "/tmp/npu_dump/layer_-1.npy")
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

        # PLE per-layer slice constants (shape metadata only, no D2H sync).
        # The custom ple_slice_layer kernel replaces the strided
        # per_layer_inputs[:, i, :].contiguous(), which JIT-compiled a distinct
        # te_StridedSliceD per layer (35 cold compiles).
        if per_layer_inputs is not None:
            ple_num_tokens = per_layer_inputs.shape[0]
            ple_num_layers = self.config.num_hidden_layers
            ple_block_bytes = (self.hidden_size_per_layer_input
                               * per_layer_inputs.element_size())

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            # Extract per-layer input for this specific layer
            layer_per_input = None
            if per_layer_inputs is not None:
                # Custom CCE kernel gathers per_layer_inputs[:, i, :] into a
                # contiguous [tokens, dim] buffer.  layer index i is a runtime
                # arg -> a single compiled kernel serves every layer (no
                # per-offset TBE StridedSliceD JIT compile).
                layer_per_input = per_layer_inputs.new_empty(
                    (ple_num_tokens, self.hidden_size_per_layer_input))
                ple_slice_layer(
                    get_pointer(layer_per_input),
                    get_pointer(per_layer_inputs),
                    i, ple_num_tokens, ple_num_layers, ple_block_bytes,
                    get_default_stream())

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
            # Dump layer output hidden states if enabled
            if getattr(self, '_dump_tensors', False):
                acl.rt.synchronize_stream(get_default_stream())
                _dump_npu_tensor(hidden_states,
                                 f"/tmp/npu_dump/layer_{i:02d}.npy")

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

    def _load_gguf_quant_weight(self, name: str, data: torch.Tensor):
        """Store/process a GGUF quantized weight (Q4_0, Q6_K, F16)."""
        base = name.rsplit(".", 1)[0]  # e.g., "layers.0.self_attn.q_proj"
        parent = self
        for part in base.split("."):
            if part.isdigit():
                parent = parent[int(part)]
            else:
                real_part = _STACKED_REMAP.get(part, part)
                parent = getattr(parent, real_part)

        raw_u8 = data.numpy().view('uint8')
        # The gguf library reshapes quantized tensors to a byte-shape
        # (e.g., [N, bytes_per_row]); the C++ converter expects a flat 1D array.
        if raw_u8.ndim != 1:
            raw_u8 = raw_u8.ravel()
        raw_len = raw_u8.size  # total number of uint8 elements
        qtype = getattr(parent, '_qweight_type', 0)

        # Determine the shard ID for stacked layers (qkv_proj, gate_up_proj).
        # The GGUF tensors use individual projection names (q_proj, k_proj, …)
        # that were remapped to the stacked module (_STACKED_REMAP above).
        # Extract the original name to know which shard this is.
        shard_id = None
        orig_last = name.rsplit(".", 2)[-2]  # e.g., "q_proj" from "q_proj.qweight"
        if orig_last in ("q_proj", "k_proj", "v_proj"):
            shard_id = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}[orig_last]
        elif orig_last in ("gate_proj", "up_proj"):
            shard_id = {"gate_proj": 0, "up_proj": 1}[orig_last]

        # Determine N, K from the layer's expected weight shape.
        # For stacked layers (qkv_proj, gate_up_proj), the shard's N and K
        # differ from the combined parent weight shape.
        N = K = 0
        is_stacked = shard_id is not None
        if is_stacked:
            # N = raw_data_bytes * 32 / (18 * K);  K = hidden_size from parent
            if hasattr(parent, 'weight') and parent.weight is not None:
                K = parent.weight.shape[-1]  # hidden_size, shared by all shards
            elif hasattr(parent, 'output_size') and hasattr(parent, 'input_size'):
                K = parent.input_size
            if K > 0:
                N = raw_len * 32 // (18 * K)
        else:
            if hasattr(parent, 'weight') and parent.weight is not None:
                N, K = parent.weight.shape
            elif hasattr(parent, 'output_size') and hasattr(parent, 'input_size'):
                N, K = parent.output_size, parent.input_size
            elif hasattr(parent, 'num_embeddings') and hasattr(parent, 'embedding_dim'):
                N, K = parent.num_embeddings, parent.embedding_dim
            else:
                import logging
                logging.getLogger(__name__).debug(
                    "GGUF: skipping %s — no weight shape on parent %s",
                    name, type(parent).__name__)
                return

        # Q6_K dequantization (token_embd, per_layer_token_embd)
        if qtype == _GGML_TYPE_Q6_K:
            self._dequant_q6_k_and_load(parent, data, raw_u8, N, K)
            return

        # F16: direct load (weights stored as raw fp16)
        expected_f16 = N * K * 2
        if raw_len == expected_f16:
            fp16_data = data.numpy().view(np.float16).reshape(N, K)
            parent.weight.data = torch.from_numpy(
                fp16_data).to(parent.weight.device, parent.weight.dtype)
            return

        # Q4_0: store raw data for later NZ conversion.
        # For stacked layers (qkv_proj, gate_up_proj), store per-shard so the
        # matmul kernel can be called independently for each shard.
        expected_q4_0 = N * K // 32 * 18
        if raw_len == expected_q4_0:
            if shard_id is not None:
                if not hasattr(parent, '_qweight_raw_shards'):
                    parent._qweight_raw_shards = {}
                    parent._qweight_shard_n = {}
                    parent._qweight_shard_k = {}
                parent._qweight_raw_shards[shard_id] = raw_u8
                parent._qweight_shard_n[shard_id] = N
                parent._qweight_shard_k[shard_id] = K
            else:
                parent._qweight_raw = raw_u8
                parent._qweight_n, parent._qweight_k = N, K
            return

        # Unknown format — skip
        import logging
        logging.getLogger(__name__).warning(
            "GGUF: unknown quant format for %s: len=%d, N=%d, K=%d, "
            "expected_f16=%d, expected_q4_0=%d, qtype=%d",
            name, raw_len, N, K, expected_f16, expected_q4_0, qtype)

    def _dequant_q6_k_and_load(self, parent, data, raw_u8, N, K):
        """Dequantize Q6_K tensor and load as fp16 weight."""
        import gguf as _gguf

        # The gguf library has already reshaped the raw tensor data to
        # the byte shape: (outer_dim, inner_dim // 256 * 210).
        # gguf.dequantize() expects exactly this byte-shape tensor.
        # After dequant: float32 with shape (outer_dim, inner_dim)
        # which is directly the vLLM weight shape since the gguf library
        # reversed the dimensions for us.
        dequant_f32 = _gguf.dequantize(data.numpy(), _gguf.GGMLQuantizationType.Q6_K)

        # Convert to fp16 and load
        fp16_data = dequant_f32.astype(np.float16)
        if hasattr(parent, 'weight') and parent.weight is not None:
            parent.weight.data = torch.from_numpy(
                fp16_data).to(parent.weight.device, parent.weight.dtype)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "GGUF: no .weight on %s for Q6_K dequant", type(parent).__name__)

    def _convert_q4_0_weights(self):
        """Convert all stored Q4_0 weights to NZ format on NPU."""
        if not _gguf_q4_0_available:
            return
        for mod in self.modules():
            # Handle sharded Q4_0 (qkv_proj, gate_up_proj)
            if hasattr(mod, '_qweight_raw_shards') and not hasattr(mod, '_qweight_nz_shards'):
                mod._qweight_nz_shards = {}
                mod._scales_shards = {}
                mod._qweight_shard_n = {}
                mod._qweight_shard_k = {}
                # Validate that N and K were stored for each shard before
                # conversion.  (k_eq_v layers may lack k/v shards.)
                valid_shards = {}
                for sid in list(mod._qweight_raw_shards.keys()):
                    if sid not in mod._qweight_shard_n:
                        continue
                    N = int(mod._qweight_shard_n.get(sid, 0))
                    K = int(mod._qweight_shard_k.get(sid, 0))
                    if N <= 0 or K <= 0:
                        continue
                    valid_shards[sid] = (N, K, mod._qweight_raw_shards[sid])
                for sid, (N, K, raw) in valid_shards.items():
                    nz_bytes = K * N // 2
                    qw_nz = np.zeros(nz_bytes, dtype=np.uint8)
                    scales = np.zeros(K // 32 * N, dtype=np.uint16)
                    convert_gguf_q4_0_qweight(raw, qw_nz, scales, N, K)
                    mod._qweight_nz_shards[sid] = torch.from_numpy(
                        qw_nz.ravel()).npu()
                    mod._scales_shards[sid] = torch.from_numpy(
                        scales.ravel().view(np.float16)).npu()
                    mod._qweight_shard_n[sid] = N
                    mod._qweight_shard_k[sid] = K
                del mod._qweight_raw_shards

            # Handle unsharded Q4_0 (down_proj, o_proj, etc.)
            if hasattr(mod, '_qweight_raw') and not hasattr(mod, '_qweight_nz'):
                raw = mod._qweight_raw
                N = mod._qweight_n
                K = mod._qweight_k
                if N <= 0 or K <= 0:
                    del mod._qweight_raw
                    continue
                nz_bytes = int(K) * int(N) // 2
                qw_nz = np.zeros(nz_bytes, dtype=np.uint8)
                scales = np.zeros(int(K) // 32 * int(N), dtype=np.uint16)
                convert_gguf_q4_0_qweight(raw, qw_nz, scales, int(N), int(K))
                mod._qweight_nz = torch.from_numpy(qw_nz.ravel()).npu()
                mod._scales = torch.from_numpy(
                    scales.ravel().view(np.float16)).npu()
                del mod._qweight_raw  # free CPU memory

    def _inject_missing_kv_weights(self):
        """Inject k_proj/v_proj for KV-shared layers (15-34) missing from GGUF."""
        import os, glob
        from safetensors import safe_open

        bf16_dir = '/ssd/models/gemma-4-E2B-it'
        if not os.path.isdir(bf16_dir):
            import logging
            logging.getLogger(__name__).warning(
                "GGUF: bf16 model not found at %s, cannot inject missing k/v",
                bf16_dir)
            return

        num_kv_shared = getattr(self.config, "num_kv_shared_layers", 0)
        if num_kv_shared <= 0:
            return
        first_kv_shared = self.config.num_hidden_layers - num_kv_shared

        safetensor_files = sorted(glob.glob(os.path.join(bf16_dir, '*.safetensors')))
        if not safetensor_files:
            return

        # Pre-load all k_proj AND v_proj weights from the bf16 model
        needed_k_weights = {}
        needed_v_weights = {}
        for f in safetensor_files:
            with safe_open(f, framework='pt') as sf:
                for key in sf.keys():
                    if 'self_attn.k_proj.weight' in key:
                        parts = key.split('.')
                        if 'layers' in parts:
                            idx = int(parts[parts.index('layers') + 1])
                            if idx >= first_kv_shared:
                                needed_k_weights[idx] = sf.get_tensor(key)
                    elif 'self_attn.v_proj.weight' in key:
                        parts = key.split('.')
                        if 'layers' in parts:
                            idx = int(parts[parts.index('layers') + 1])
                            if idx >= first_kv_shared:
                                needed_v_weights[idx] = sf.get_tensor(key)

        import logging
        _log = logging.getLogger(__name__)
        injected = 0
        for idx in range(first_kv_shared, self.config.num_hidden_layers):
            if idx >= len(self.layers):
                break
            layer = self.layers[idx]
            qkv = layer.self_attn.qkv_proj

            # Skip if k shard already exists
            if hasattr(qkv, '_qweight_nz_shards') and 'k' in qkv._qweight_nz_shards:
                continue

            k_tensor = needed_k_weights.get(idx)
            v_tensor = needed_v_weights.get(idx)
            if k_tensor is None or v_tensor is None:
                _log.warning("GGUF: layer %d k/v not found in bf16 model", idx)
                continue

            # Quantize and convert k_proj
            k_f32 = k_tensor.float().numpy()
            Nk, K = k_f32.shape
            gguf_k = _quantize_f32_to_q4_0(k_f32)
            nz_k = np.zeros(K * Nk // 2, dtype=np.uint8)
            sc_k = np.zeros(K // 32 * Nk, dtype=np.uint16)
            convert_gguf_q4_0_qweight(gguf_k, nz_k, sc_k, Nk, K)

            # Quantize and convert v_proj
            v_f32 = v_tensor.float().numpy()
            Nv, Kv = v_f32.shape
            gguf_v = _quantize_f32_to_q4_0(v_f32)
            nz_v = np.zeros(Kv * Nv // 2, dtype=np.uint8)
            sc_v = np.zeros(Kv // 32 * Nv, dtype=np.uint16)
            convert_gguf_q4_0_qweight(gguf_v, nz_v, sc_v, Nv, Kv)

            # Initialize shard dicts if not already present
            if not hasattr(qkv, '_qweight_nz_shards'):
                qkv._qweight_nz_shards = {}
                qkv._scales_shards = {}
                qkv._qweight_shard_n = {}
                qkv._qweight_shard_k = {}

            # Store k shard
            qkv._qweight_nz_shards['k'] = torch.from_numpy(nz_k.ravel()).npu()
            qkv._scales_shards['k'] = torch.from_numpy(
                sc_k.ravel().view(np.float16)).npu()
            qkv._qweight_shard_n['k'] = Nk
            qkv._qweight_shard_k['k'] = K

            # Store v shard
            qkv._qweight_nz_shards['v'] = torch.from_numpy(nz_v.ravel()).npu()
            qkv._scales_shards['v'] = torch.from_numpy(
                sc_v.ravel().view(np.float16)).npu()
            qkv._qweight_shard_n['v'] = Nv
            qkv._qweight_shard_k['v'] = Kv

            injected += 1
            _log.debug("GGUF: injected k/v Q4_0 for layer %d (Nk=%d Nv=%d K=%d)",
                       idx, Nk, Nv, K)

        if injected > 0:
            _log.info("GGUF: injected missing k/v Q4_0 for %d layers from bf16 model",
                      injected)

    def _post_load_process_weights(self):
        """Call process_weights_after_loading on non-Q4_0 modules.

        The GGUF loader does not call process_weights_after_loading, but NPU
        linear methods and embedding methods need it to transpose weights to NZ
        format.  Skip Q4_0 layers (they have _qweight_nz and use a custom kernel).
        """
        import logging
        _log = logging.getLogger(__name__)
        for name, mod in self.named_modules():
            # Only process LinearBase subclasses (ReplicatedLinear, etc.) —
            # NOT VocabParallelEmbedding.  Embeddings use layer.weight directly
            # in standard format; transposing them would break the lm_head apply()
            # path (which reads k from the transposed shape and tries to reshape
            # hidden_states incorrectly).
            meth = getattr(mod, 'quant_method', None)
            if meth is None:
                continue
            # Skip Q4_0 layers — they use a custom kernel via _qweight_nz or _qweight_nz_shards
            if hasattr(mod, '_qweight_nz') or hasattr(mod, '_qweight_nz_shards'):
                continue
            # Skip if the process method doesn't exist
            process_fn = getattr(meth, 'process_weights_after_loading', None)
            if process_fn is None:
                continue
            # Only process if weight was actually loaded
            if hasattr(mod, 'weight') and mod.weight is not None:
                if mod.weight.numel() == 0:
                    continue
                _log.debug("GGUF post-load process: %s (%s)", name,
                           type(meth).__name__)
                process_fn(mod)

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

            # Intercept GGUF quantized weights BEFORE stacked_params_mapping.
            # qweight_type tells us the GGML quantization type; qweight is the
            # raw quantized data.  These must be handled before the
            # stacked_params_mapping below, which would incorrectly remap
            # e.g. q_proj.qweight → qkv_proj.qweight (a name that doesn't
            # exist in params_dict).
            if name.endswith(".qweight_type"):
                # Store type on the parent module for use when processing qweight.
                # Apply _STACKED_REMAP to resolve stacked modules (qkv_proj, gate_up_proj).
                base = name.replace(".qweight_type", "")
                parent = self
                for part in base.split("."):
                    if part.isdigit():
                        parent = parent[int(part)]
                    else:
                        real_part = _STACKED_REMAP.get(part, part)
                        parent = getattr(parent, real_part, None)
                        if parent is None:
                            break
                if parent is not None:
                    parent._qweight_type = int(loaded_weight.item())
                continue
            if name.endswith(".qweight") and _gguf_q4_0_available:
                self._load_gguf_quant_weight(name, loaded_weight)
                continue

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
                # Names not matching any stacked param (norms, buffers, etc.)

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
                    if name.endswith(".layer_scalar"):
                        # The forward multiplies via mul_scalar_layer using this
                        # cached Python float; the NPU buffer's value is never
                        # read.  So SKIP default_weight_loader — its
                        # param.data.fill_(loaded_weight.item()) JIT-compiles a
                        # te_Fill TBE kernel under the NPU device — and just cache
                        # the float from the CPU checkpoint tensor (no D2H sync).
                        # loaded_weight is bf16 [1] on CPU, so float() is exact.
                        owner = self.get_submodule(name.rsplit(".", 1)[0])
                        owner._layer_scalar_f = float(loaded_weight)
                    else:
                        weight_loader = getattr(param, "weight_loader",
                                                default_weight_loader)
                        weight_loader(param, loaded_weight)

        # Post-load steps — only for GGUF loads.
        # Detect GGUF load by checking for _qweight_raw on any module.
        _is_gguf = any(hasattr(m, '_qweight_raw') or hasattr(m, '_qweight_raw_shards')
                       for m in self.modules())

        if _is_gguf:
            # Convert GGUF Q4_0 weights to NZ format
            self._convert_q4_0_weights()

            # Inject missing k/v for KV-shared layers (15-34) from bf16 model
            self._inject_missing_kv_weights()

            # Post-load: transpose non-Q4_0 weights to NZ format.
            # The GGUF loader doesn't call process_weights_after_loading,
            # but NPU linear methods need NZ-transposed weights.
            self._post_load_process_weights()


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

        # For GGUF models on NPU, force UnquantizedLinearMethod / UnquantizedEmbeddingMethod.
        # Quantization is handled in Gemma4Model.load_weights (_load_gguf_quant_weight).
        if quant_config is not None and quant_config.get_name() == "gguf":
            vllm_config.quant_config = None
            quant_config = None

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
