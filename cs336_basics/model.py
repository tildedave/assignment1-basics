from math import sqrt, inf

import torch
from torch import nn, Tensor, stack, sin, cos
from pytest import approx
from einops import einsum, rearrange
from jaxtyping import Float, Bool


class Linear(nn.Module):
    weight: Float[Tensor, " out_features in_features"]

    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        w = torch.empty(out_features, in_features, **factory_kwargs)
        # trunc_normal_ is not yet validated by code
        stddev = sqrt(2 / (out_features + in_features))
        weight = torch.nn.init.trunc_normal_(w, mean=0.0, std=stddev, a=-3 * stddev, b=3 * stddev)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


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

        w = torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        weights = torch.nn.init.trunc_normal_(w, mean=0.0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(weights)

    def forward(self, token_ids: Float[Tensor, " d_in"]) -> Float[Tensor, " embedding_dim d_in"]:
        # Apparently pytorch does what you'd want here
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = nn.Parameter(torch.tensor(eps, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = (((x * x).mean(dim=-1, keepdim=True)) + self.eps).sqrt()
        result = x * self.weight / rms

        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()

        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: Float[Tensor, "... d_model"]):
        a = self.w1(x)
        return self.w2(a * torch.sigmoid(a) * self.w3(x))


# NOTE: You should set 𝑑ff to approximately 8
# 3 × 𝑑_model in your implementation, while ensuring that the
# dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
# hardware.


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, "must have d_k even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        ks = torch.arange(1, d_k / 2 + 1, device=device)
        exp = (2 * ks - 2) / d_k
        frequencies = theta ** (-1 * exp)
        positions = torch.arange(0, max_seq_len)
        theta = torch.outer(positions, frequencies)

        top = stack(
            (
                cos(theta),
                -1 * sin(theta),
            ),
            dim=-1,
        )
        bottom = stack(
            (
                sin(theta),
                cos(theta),
            ),
            dim=-1,
        )
        rotations = stack(
            (
                top,
                bottom,
            ),
            dim=-2,
        )
        self.register_buffer("rotations", rotations, persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Float[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        split_x = rearrange(x, "... (a b) -> ... a b", b=2)
        # print(split_x)
        # print(self.rotations[token_positions])
        applied = einsum(self.rotations[token_positions], split_x, "... x z, ... z -> ... x")
        result = rearrange(applied, "... a b -> ... (a b)")
        return result


def test_rope():
    rope = RoPE(theta=10, d_k=4, max_seq_len=3)
    # Validate that R @ R^T = I
    assert torch.allclose(einsum(rope.rotations, rope.rotations, "... x y, ... b y -> ... x b"), torch.eye(2))
    # assert torch.allclose(result.forward(torch.ones(5, 4), torch.ones(5, dtype=int)), expected)


def softmax(x, dim):
    largest, _ = x.max(dim=dim, keepdim=True)
    xi = torch.exp(x - largest)
    return xi / xi.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " batch_size ... seq_len d_v"]:

    d_k = Q.shape[-1]
    QKT = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") * (d_k**-0.5)
    QKT = QKT.masked_fill(~mask, -1 * inf)
    return einsum(
        softmax(QKT, dim=-1),
        V,
        "... queries keys, ... keys d_z -> ... queries d_z",
    )
