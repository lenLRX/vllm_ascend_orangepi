import torch
import torch_npu
from .py_npu_ops import NPUPtr, DataType


def get_default_stream():
    return torch.npu.default_stream().npu_stream


def get_pointer(x):
    return NPUPtr(x.data_ptr())

