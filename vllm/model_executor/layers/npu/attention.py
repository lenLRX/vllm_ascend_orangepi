"""Attention layer."""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
import acl

from vllm.attention import AttentionMetadata, AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, AttentionType)
from .util import get_default_stream, get_pointer, DataType
from .py_npu_ops import (batch_matmul_qk_trans_causual_layer,
                         softmax_layer,
                         batch_matmul_trans_v_layer)


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            sliding_window = cache_config.sliding_window
            is_attention_free = cache_config.is_attention_free
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            sliding_window = None
            is_attention_free = False
        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.hidden_dim = self.num_heads * self.head_size
        self.scale = scale

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = 1.0
        self._v_scale = 1.0
        quant_method = quant_config.get_quant_method(
            self, prefix=prefix) if quant_config else None
        if quant_method is not None:
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        #print(f"q shape {query.shape} k shape {key.shape}, v shape {value.shape}, kv_cache shape {kv_cache.shape}")
        #print(f"attn_metadata {repr(attn_metadata)}")
        #print(f"offsets: {attn_metadata.offsets[0]}")
        assert len(attn_metadata.offsets) == 1
        copy_offset = attn_metadata.offsets[0]
        copy_bytes = attn_metadata.seq_lens[0] * kv_cache.dtype.itemsize * self.num_kv_heads * self.head_size
        k_cache_base = kv_cache[0, copy_offset, :]
        v_cache_base = kv_cache[1, copy_offset, :]
        k_cache = kv_cache[0, ...]
        v_cache = kv_cache[1, ...]
        # update k_cache
        ret = acl.rt.memcpy_async(k_cache_base.data_ptr(), copy_bytes, key.data_ptr(), copy_bytes, 3, get_default_stream())
        assert ret == 0, "failed to copy k cache"
        ret = acl.rt.memcpy_async(v_cache_base.data_ptr(), copy_bytes, value.data_ptr(), copy_bytes, 3, get_default_stream())
        assert ret == 0, "failed to copy v cache"

        cur_pos = attn_metadata.offsets[0] + attn_metadata.seq_lens[0]

        q_matmul_k = torch.empty((self.num_heads, cur_pos, attn_metadata.seq_lens[0]), dtype=torch.float16, device="npu")
        batch_matmul_qk_trans_causual_layer(get_pointer(q_matmul_k), get_pointer(query), get_pointer(k_cache),
                                            self.num_heads, attn_metadata.seq_lens[0], cur_pos, self.head_size, attn_metadata.offsets[0],
                                            self.scale, DataType.DT_FLOAT16, get_default_stream())
        hs = self.num_heads * attn_metadata.seq_lens[0]
        softmax_output = torch.empty((hs, cur_pos), dtype=torch.float16, device="npu")
        softmax_layer(get_pointer(softmax_output), get_pointer(q_matmul_k), hs, cur_pos, DataType.DT_FLOAT16, get_default_stream())

        tmp_output = torch.empty((attn_metadata.seq_lens[0], self.hidden_dim), dtype=torch.float16, device="npu")
        batch_matmul_trans_v_layer(get_pointer(tmp_output), get_pointer(softmax_output), get_pointer(v_cache),
                                   self.num_heads, attn_metadata.seq_lens[0], self.head_size, cur_pos, 1.0,
                                   DataType.DT_FLOAT16, get_default_stream())

        acl.rt.synchronize_stream(get_default_stream())
        return tmp_output


    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}"  # type: ignore
        s += f", num_heads={self.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.num_kv_heads}"  # type: ignore
        return s
