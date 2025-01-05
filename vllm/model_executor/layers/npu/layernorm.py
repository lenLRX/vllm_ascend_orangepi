from typing import Optional, Tuple, Union

import acl
import torch
import torch.nn as nn

from .util import get_default_stream, get_pointer, DataType
from .py_npu_ops import rmsnorm_layer, add_layer


class RMSNorm(torch.nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO use fp32 add
        if residual is not None:
            add_out = torch.empty_like(x)
            add_layer(get_pointer(add_out), get_pointer(x), get_pointer(residual),
                      x.numel(), DataType.DT_FLOAT16, get_default_stream())
            residual = add_out


        assert x.dtype == torch.float16

        hidden_size = x.shape[-1]
        assert hidden_size % 16 == 0
        if hidden_size != self.hidden_size:
            raise ValueError("Expected hidden_size to be "
                             f"{self.hidden_size}, but found: {hidden_size}")

        assert self.variance_size_override is None

        first_dim = x.numel() // hidden_size

        output = torch.empty_like(x)
        rmsnorm_layer(get_pointer(output), get_pointer(self.weight),
                      get_pointer(x if residual is None else add_out), first_dim, hidden_size, self.variance_epsilon,
                      DataType.DT_FLOAT16, get_default_stream())

        acl.rt.synchronize_stream(get_default_stream())
        if residual is None:
            return output
        else:
            return output, residual

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s



