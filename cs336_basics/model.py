from math import sqrt

import torch
from torch import nn, Tensor
from pytest import approx
from einops import einsum
from jaxtyping import Float


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")


def test_validate_default_weights():
    linear = Linear(128, 128)
    stddev = sqrt(2 / (128 + 128))
    assert linear.weights.std().item() == approx(stddev, rel=0.1)
    assert linear.weights.max().item() < 3 * stddev
    assert linear.weights.min().item() > -3 * stddev


class Embedding(nn.Module):
    """
    In the very first step, the Transformer embeds the (batched) sequence of token IDs into a sequence of
    vectors containing information on the token identity (red blocks in Figure 1).

    More specifically, given a sequence of token IDs, the Transformer language model uses a token embedding
    layer to produce a sequence of vectors. Each embedding layer takes in a tensor of integers of shape
    (batch_size, sequence_length) and produces a sequence of vectors of shape (batch_size,
    sequence_length, d_model)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,  # d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()

        nn.Embedding
        w = torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        weights = torch.nn.init.trunc_normal_(w, mean=0.0, std=1, a=-3, b=3)
        self.weights = nn.Parameter(weights)

    def forward(self, token_ids: Float[Tensor, " d_in"]) -> Float[Tensor, " embedding_dim d_in"]:
        # Apparently pytorch does what you'd want here
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = nn.Parameter(torch.tensor(eps, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = (((x * x).mean(dim=-1, keepdim=True)) + self.eps).sqrt()
        result = x * self.weights / rms

        return result.to(in_dtype)
