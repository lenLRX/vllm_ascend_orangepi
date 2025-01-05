import torch

from .util import get_default_stream, get_pointer, DataType
from .py_npu_ops import silu_mul_layer_vllm


class SiluAndMul:
    def __init__(self):
        pass

    def __call__(self, x):
        #print(f"SiluAndMul input shape {x.shape}")
        output_shape = list(x.shape)
        output_shape[-1] = output_shape[-1] // 2
        last_dim = output_shape[-1]
        assert last_dim % 16 == 0
        assert x.dtype == torch.float16
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        shape2d = output.reshape(-1, last_dim).shape
        silu_mul_layer_vllm(get_pointer(output), get_pointer(x),
                            shape2d[0], shape2d[1],
                            DataType.DT_FLOAT16, get_default_stream())
        
        return output
        


