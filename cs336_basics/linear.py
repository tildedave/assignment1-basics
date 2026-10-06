from math import sqrt

import torch
from torch import nn

from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        w = torch.empty(out_features, in_features, **factory_kwargs)
        # trunc_normal_ is not yet validated by code
        stddev = sqrt(2 / (out_features + in_features))
        weights = torch.nn.init.trunc_normal_(w, mean=0.0, std=stddev, a=-3 * stddev, b=3 * stddev)
        self.weights = nn.Parameter(weights)

        nn.Linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
