import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    largest, _ = inputs.max(dim=-1, keepdim=True)
    denom = torch.exp(inputs - largest)
    tensor = torch.log(denom.sum(dim=-1, keepdim=True)) - (inputs - largest)
    return torch.gather(tensor, -1, rearrange(targets, "... -> ... 1")).mean()
