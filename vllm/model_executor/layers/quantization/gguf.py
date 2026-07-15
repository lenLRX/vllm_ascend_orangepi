from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.parameter import Parameter, UninitializedParameter

# GGML quant size table (block_size, type_size_bytes), keyed by GGML type id.
# Mirrored from vllm.model_executor.model_loader.gguf_reader to avoid a circular
# import through the model_loader package. GGML format constants (stable).
_GGML_QUANT_SIZES = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    24: (1, 1),    # I8
    25: (1, 2),    # I16
    26: (1, 4),    # I32
    27: (1, 8),    # I64
    28: (1, 8),    # F64
    30: (1, 2),    # BF16
}

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.utils import set_weight_attrs

# NPU imports (only used when device is NPU)
_npu_available = False
try:
    from vllm.model_executor.layers.npu.py_npu_ops import (
        matmul_gguf_q4_0_layer, convert_gguf_q4_0_qweight)
    from vllm.model_executor.layers.npu.util import get_default_stream, get_pointer, to_npu_dtype
    _npu_available = True
except Exception:
    pass

# CUDA ops — may not be available on NPU
try:
    from vllm import _custom_ops as ops
except Exception:
    ops = None


GGML_TYPE_Q4_0 = 2


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, ) -> None:
        pass

    def __repr__(self) -> str:
        return ("GGUFConfig()")

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            return GGUFEmbeddingMethod(self)
        return None


def _fuse_mul_mat_npu(x: torch.Tensor, qweight_nz: torch.Tensor,
                       scales: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """NPU Q4_0 matmul using custom CCE kernel."""
    M, K_in = x.shape
    assert K_in == K, f"K mismatch: input {K_in} != weight {K}"
    out = torch.empty(M * N, dtype=torch.float16, device='npu')
    matmul_gguf_q4_0_layer(
        get_pointer(out), get_pointer(x.reshape(-1)),
        get_pointer(qweight_nz), get_pointer(scales),
        M, N, K, to_npu_dtype(torch.float16), get_default_stream())
    return out.reshape(M, N)


def _convert_q4_0_on_device(layer: torch.nn.Module) -> None:
    """Convert Q4_0 GGUF weights to NZ format for NPU. Called once after loading."""
    qweight = layer.qweight
    N = qweight.shape[0]
    K = getattr(layer, '_q4_0_k', None)
    if K is None:
        raise ValueError("Q4_0 K not set on layer")

    # Handle nested tensor from GGUF loader
    if qweight.is_nested:
        # Unbind nested tensor to get the single (unsharded) weight tensor
        qw_tensors = qweight.unbind()
        if len(qw_tensors) == 1:
            qw_t = qw_tensors[0]
        else:
            raise RuntimeError(f"Expected 1 nested tensor, got {len(qw_tensors)}")
    else:
        qw_t = qweight

    # Move to CPU for conversion
    qw_cpu = qw_t.cpu()
    # Ensure contiguous flat array of uint8
    qw_flat = np.ascontiguousarray(qw_cpu.numpy()).view('uint8').flatten()

    nz_bytes = K * N // 2
    qweight_nz = np.zeros(nz_bytes, dtype=np.uint8)
    scales = np.zeros(K // 32 * N, dtype=np.uint16)

    convert_gguf_q4_0_qweight(qw_flat, qweight_nz, scales, N, K)

    # Store converted weights on layer (on NPU device)
    layer._qweight_nz = torch.from_numpy(qweight_nz).npu()
    layer._scales = torch.from_numpy(scales.view(np.float16)).npu()
    layer._q4_0_n = N
    layer._q4_0_k = K


def _fuse_mul_mat(x: torch.Tensor, qweight: torch.Tensor,
                  qweight_type: int, layer=None) -> torch.Tensor:
    """Dispatch matmul: NPU custom kernel for Q4_0, CUDA ops otherwise."""
    is_npu = x.device.type == 'npu'

    if is_npu and qweight_type == GGML_TYPE_Q4_0 and _npu_available:
        # Use NPU custom kernel — lazy convert on first call
        if layer is not None and not hasattr(layer, '_qweight_nz'):
            _convert_q4_0_on_device(layer)
        if layer is not None and hasattr(layer, '_qweight_nz'):
            return _fuse_mul_mat_npu(
                x, layer._qweight_nz, layer._scales,
                layer._q4_0_n, layer._q4_0_k)
        else:
            raise RuntimeError(
                "Q4_0 NPU weights not converted and layer not available.")

    if ops is None:
        raise RuntimeError(
            "vllm._custom_ops not available and NPU not available. "
            "Cannot run quantized matmul.")

    # Original CUDA path
    if x.shape[0] == 1:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type >= 16:
        block_size, type_size = _GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape)
        y = x @ weight.T
    else:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    return y


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            })
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(torch.empty(len(output_partition_sizes),
                                             dtype=torch.uint8),
                                 requires_grad=False)
        set_weight_attrs(
            qweight_type, {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True
            })
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

        # Store K dimension for Q4_0 NPU conversion
        layer._q4_0_k = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert Q4_0 weights to NZ format for NPU inference."""
        qweight_type = layer.qweight_type.weight_type
        if isinstance(qweight_type, torch.Tensor):
            qweight_type = qweight_type.item()
        if qweight_type == GGML_TYPE_Q4_0 and _npu_available:
            _convert_q4_0_on_device(layer)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        shard_id = getattr(layer.qweight, "shard_id", None)

        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight.unbind(0)
            result = []
            for sid in shard_id:
                q_idx = layer.qweight.shard_id_map[sid]
                qweight_type = layer.qweight_type.shard_weight_type[sid]
                result.append(_fuse_mul_mat(x, qweight[q_idx], qweight_type, layer))
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            if isinstance(qweight_type, torch.Tensor):
                qweight_type = qweight_type.item()
            out = _fuse_mul_mat(x, qweight, qweight_type, layer)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module,
                  x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        block_size, type_size = _GGML_QUANT_SIZES[qweight_type]
        hidden_size = qweight.shape[1] // type_size * block_size
        if qweight_type < 2:
            return torch.embedding(qweight, x)
        x_flat = x.flatten()
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ops.ggml_dequantize(quant, qweight_type, hidden_size,
                                      x_flat.shape[0])
        return dequant.view(*x.shape, hidden_size)


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: List[torch.Tensor]

    def materialize_nested(self) -> Parameter:
        nested_data = torch.nested.nested_tensor(self.data_container,
                                                 device=self.device,
                                                 dtype=torch.uint8)
        self.data_container.clear()
        param = torch.Tensor._make_subclass(self.cls_to_become,
                                            nested_data,
                                            require_grad=False)
        for k, v in self.__dict__.items():
            setattr(param, k, v)
        return param
