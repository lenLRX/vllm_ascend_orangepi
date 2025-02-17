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
        self.block_size = block_size

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
        #print(f"offsets: {attn_metadata.offsets}")
        #print(f"seq_lens: {attn_metadata.seq_lens}")
        #print(f"block_tables: {attn_metadata.block_tables}")
        batch_token_num, hidden_dim = query.shape

        tmp_output = torch.empty((batch_token_num, self.hidden_dim), dtype=torch.float16, device="npu")
        block_bytes = self.block_size * kv_cache.dtype.itemsize * self.num_kv_heads * self.head_size

        batch_size = len(attn_metadata.seq_lens)
        flat_seq_offset = 0
        for batch_i in range(batch_size):
            #print(f"key[{batch_i}]", key[batch_i].cpu())

            # update kv cache
            curr_seq_len = attn_metadata.seq_lens[batch_i]
            curr_offset = attn_metadata.offsets[batch_i]
            remain_seq_len = curr_seq_len
            curr_block_table = attn_metadata.block_tables[batch_i]
            offset_in_block = curr_offset % self.block_size
            block_table_i = curr_offset // self.block_size

            curr_seq_offset = 0
            while remain_seq_len > 0:
                copy_seq_len = min(self.block_size - offset_in_block, remain_seq_len)
                copy_bytes = copy_seq_len * kv_cache.dtype.itemsize * self.num_kv_heads * self.head_size

                k_cache_base = kv_cache[0, curr_block_table[block_table_i], offset_in_block, 0]
                v_cache_base = kv_cache[1, curr_block_table[block_table_i], offset_in_block, 0]

                #print(f"kv_cache base ptr: {kv_cache.data_ptr()}")
                #print(f"k_cache base ptr: {k_cache_base.data_ptr()}")

                #print(f"batch: {batch_i}, copy_bytes: {copy_bytes}, block_id: {curr_block_table[block_table_i]}, offset_in_block: {offset_in_block}")

                # update k_cache
                ret = acl.rt.memcpy_async(k_cache_base.data_ptr(), copy_bytes, key[flat_seq_offset + curr_seq_offset].data_ptr(), copy_bytes, 3, get_default_stream())
                assert ret == 0, "failed to copy k cache"
                ret = acl.rt.memcpy_async(v_cache_base.data_ptr(), copy_bytes, value[flat_seq_offset + curr_seq_offset].data_ptr(), copy_bytes, 3, get_default_stream())
                assert ret == 0, "failed to copy v cache"

                remain_seq_len -= copy_seq_len
                block_table_i += 1
                offset_in_block = (offset_in_block + copy_seq_len) % self.block_size
                curr_seq_offset += copy_seq_len


            curr_seq_len = attn_metadata.seq_lens[batch_i]
            curr_offset = attn_metadata.offsets[batch_i]
            curr_seq_block_num = len(curr_block_table)
            curr_k_cache = torch.empty((curr_seq_block_num, self.block_size, self.hidden_dim), dtype=torch.float16, device="npu")
            curr_v_cache = torch.empty((curr_seq_block_num, self.block_size, self.hidden_dim), dtype=torch.float16, device="npu")

            curr_pos = curr_offset + curr_seq_len

            for i, block_id in enumerate(curr_block_table):
                #print(f"i: {i}, block_id: {block_id}, block_bytes: {block_bytes}")
                #acl.rt.synchronize_stream(get_default_stream())
                #print(f"k_cache", kv_cache[0, block_id, 0, :].cpu())

                # update k_cache
                ret = acl.rt.memcpy_async(curr_k_cache[i, ...].data_ptr(), block_bytes, kv_cache[0, block_id, 0, 0].data_ptr(), block_bytes, 3, get_default_stream())
                assert ret == 0, "failed to copy k cache"
                ret = acl.rt.memcpy_async(curr_v_cache[i, ...].data_ptr(), block_bytes, kv_cache[1, block_id, 0, 0].data_ptr(), block_bytes, 3, get_default_stream())
                assert ret == 0, "failed to copy v cache"

            #acl.rt.synchronize_stream(get_default_stream())
            #print(f"batch: {batch_i} curr_k_cache", curr_k_cache.cpu())
            #print(f"batch: {batch_i} curr_v_cache", curr_v_cache.cpu())

            q_matmul_k = torch.empty((self.num_heads, curr_pos, curr_seq_len), dtype=torch.float16, device="npu")
            batch_matmul_qk_trans_causual_layer(get_pointer(q_matmul_k), get_pointer(query[flat_seq_offset]), get_pointer(curr_k_cache),
                                                self.num_heads, curr_seq_len, curr_pos, self.head_size, curr_offset,
                                                self.scale, DataType.DT_FLOAT16, get_default_stream())
            #acl.rt.synchronize_stream(get_default_stream())
            #print(f"batch: {batch_i} q_matmul_k", q_matmul_k.cpu())


            hs = self.num_heads * curr_seq_len
            softmax_output = torch.empty((hs, curr_pos), dtype=torch.float16, device="npu")
            softmax_layer(get_pointer(softmax_output), get_pointer(q_matmul_k), hs, curr_pos, DataType.DT_FLOAT16, get_default_stream())

            #acl.rt.synchronize_stream(get_default_stream())
            #print(f"batch: {batch_i} softmax_output", softmax_output.cpu())

            batch_matmul_trans_v_layer(get_pointer(tmp_output[flat_seq_offset]), get_pointer(softmax_output), get_pointer(curr_v_cache),
                                       self.num_heads, curr_seq_len, self.head_size, curr_pos, 1.0,
                                       DataType.DT_FLOAT16, get_default_stream())
            #acl.rt.synchronize_stream(get_default_stream())
            #print(f"batch: {batch_i} temp_output", tmp_output[batch_i].cpu())
            flat_seq_offset += curr_seq_len


        acl.rt.synchronize_stream(get_default_stream())
        return tmp_output


    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}"  # type: ignore
        s += f", num_heads={self.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.num_kv_heads}"  # type: ignore
        return s
