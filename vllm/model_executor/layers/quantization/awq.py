from typing import Any, Dict, List, Optional

import torch
import numpy as np
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)

from vllm.model_executor.layers.npu.util import get_default_stream, get_pointer, DataType
from vllm.model_executor.layers.npu.py_npu_ops import matmul_nz_awq_4bit_layer



class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert or []

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point}, "
                f"modules_to_not_convert={self.modules_to_not_convert})")

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None)
        return cls(weight_bits, group_size, zero_point, modules_to_not_convert)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["LinearMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_awq(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return AWQLinearMethod(self)
        return None


def is_layer_skipped_awq(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        weight_loader = extra_weight_attrs.get("weight_loader")
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            input_size_per_partition // self.quant_config.group_size,
            output_size_per_partition,
            dtype=params_dtype,
        ),
                                          input_dim=0,
                                          output_dim=1,
                                          weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        np_qweight = layer.qweight.data.cpu().numpy()
        np_qzeros = layer.qzeros.data.cpu().numpy()
        original_qweight_shape = np_qweight.shape
        original_qweight_dtype = np_qweight.dtype
        original_qzero_shape = np_qzeros.shape
        original_qzero_dtype = np_qzeros.dtype
        #print(f"qweight shape {np_qweight.shape}")
        #print(f"qzeros shape {np_qzeros.shape}")

        np_qweight = np_qweight.view("uint8")
        k_dim, n_dim = np_qweight.shape
        np_qweight = np_qweight.reshape(k_dim, n_dim, 1)
        np_qweight = np.repeat(np_qweight, 2, axis=-1)
        np_qweight[..., 0] = np_qweight[..., 0] & 0xf
        np_qweight[..., 1] = (np_qweight[..., 1] >> 4) & 0xf
        n_dim = n_dim * 2
        np_qweight = np_qweight.reshape(k_dim, n_dim//8, 2, 4)
        np_qweight = np.transpose(np_qweight, (0, 1, 3, 2))
        # transpose to (k, n)
        np_qweight = np_qweight.reshape(k_dim//16, 16, n_dim)
        np_qweight = np.transpose(np_qweight, (0, 2, 1))
        d1 = np_qweight.size // 512
        np_qweight  = np_qweight.reshape(d1, 4, 64, 2)
        np_qweight = np.transpose(np_qweight, (0, 2, 1, 3))
        np_qweight = (np_qweight + 8)&0xf
        np_qweight[..., 0] = np_qweight[..., 0] | (np_qweight[...,1] << 4)
        np_qweight = np.ascontiguousarray(np_qweight[..., 0]).view(original_qweight_dtype).reshape(original_qweight_shape)

        np_qzeros = np_qzeros.view("uint8")
        k_dim, n_dim = np_qzeros.shape
        np_qzeros = np_qzeros.reshape(k_dim, n_dim, 1)
        np_qzeros = np.repeat(np_qzeros, 2, axis=-1)
        np_qzeros[..., 0] = np_qzeros[..., 0] & 0xf
        np_qzeros[..., 1] = (np_qzeros[..., 1] >> 4) & 0xf
        np_qzeros = np_qzeros.astype("float16")
        np_qzeros = np_qzeros - 8.0
        n_dim = n_dim * 2
        np_qzeros = np_qzeros.reshape(k_dim, n_dim//8, 2, 4)
        np_qzeros = np.transpose(np_qzeros, (0, 1, 3, 2))
        np_qzeros = np_qzeros.reshape(k_dim, n_dim)
        np_qzeros = np.ascontiguousarray(np_qzeros)

        layer.qweight = torch.nn.Parameter(torch.from_numpy(np_qweight).npu(),
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(torch.from_numpy(np_qzeros).npu(),
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data,
                                          requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        print(f"input shape: {x.shape}, qweight shape: {qweight.shape}, qweight dtype {qweight.dtype}, pack_factor {pack_factor}")
        k, n = qweight.shape
        n *= pack_factor
        m, k = x.reshape(-1, k).shape


        matmul_nz_awq_4bit_layer(get_pointer(out), get_pointer(reshaped_x), get_pointer(qweight), get_pointer(qzeros), get_pointer(scales),
                                 m, n, k, DataType.DT_FLOAT16, get_default_stream())
        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
